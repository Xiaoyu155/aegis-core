"""Stdlib dataset validation and explicit GPU preflight for optional trainers."""
import json
from pathlib import Path


def rows(path, *, supplements=()):
    values = [json.loads(line) for source in (path, *supplements)
              for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError("empty training dataset")
    return values


def quantized_model(model_id, *, rank=8):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: install an RTX-compatible CUDA torch wheel; no full precision fallback")
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    if major >= 12 and arch not in torch.cuda.get_arch_list():
        raise RuntimeError(f"torch wheel lacks {arch}; use the official CUDA 12.8+ wheel")
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=config,
                                                device_map={"": 0}, trust_remote_code=False)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(**lora_settings(rank)))
    return model, tokenizer, bf16


def lora_settings(rank: int) -> dict:
    if type(rank) is not int or rank not in {4, 8, 16}:
        raise ValueError("8 GB LoRA preflight supports ranks 4, 8 and 16")
    return {"r": rank, "lora_alpha": rank * 2, "lora_dropout": .05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "task_type": "CAUSAL_LM"}


def validate_budget(max_steps, *lengths):
    if max_steps < 1 or any(not 128 <= length <= 4096 for length in lengths):
        raise ValueError("positive max steps and token budgets in [128,4096] required")


def conversation(value, *, roles):
    if not isinstance(value, list) or any(not isinstance(message, dict)
                                         or type(message.get("content")) is not str
                                         or not message["content"].strip() for message in value):
        raise ValueError("expected a nonempty text conversation")
    if tuple(message.get("role") for message in value) not in roles:
        raise ValueError("invalid conversation roles")
    return value


class ByteTokenizer:
    """Deterministic local token IDs for a dependency-free data-path preflight."""

    name_or_path = "utf8-byte-preflight"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False):
        text = "".join(f"<{message['role']}>\n{message['content']}\n</{message['role']}>\n"
                       for message in messages)
        if add_generation_prompt:
            text += "<assistant>\n"
        return list(text.encode("utf-8")) if tokenize else text

    def encode(self, text, *, add_special_tokens=False):
        return list(text.encode("utf-8"))


def dry_tokenizer(path):
    if path is None:
        return ByteTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=False, local_files_only=True)


def chat_tokens(tokenizer, messages, *, generation=False):
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation,
                                            **({"return_dict": False} if not isinstance(tokenizer, ByteTokenizer) else {}))
    return list(encoded)


def verify_dataset(path):
    """Resolve and verify the training shards listed in an adjacent v3 manifest."""
    import hashlib

    path = Path(path)
    manifest_path = path.parent / "manifest_v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    supplement = manifest.get("cat_cafe_replay", {}).get("supplements", {}).get(path.name)
    if supplement is not None and (not isinstance(supplement, str) or Path(supplement).name != supplement
                                   or supplement == path.name):
        raise ValueError("invalid training supplement filename")
    supplements = (path.parent / supplement,) if supplement is not None else ()
    expected = manifest.get("sha256", {})
    digests = {}
    for source in (path, *supplements):
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if (source.name in expected or source in supplements) and expected.get(source.name) != digest:
            raise ValueError(f"dataset checksum mismatch for {source.name}")
        digests[source.name] = digest
    return digests, supplements


def export_preflight(output, kind, payload):
    """Atomically write and read back a portable preflight checkpoint."""
    import gzip
    import hashlib
    import os
    import tempfile

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}-preflight.checkpoint.json.gz"
    body = json.dumps({"schema_version": "aegis.training.preflight.v1", "kind": kind, **payload},
                      ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    digest = hashlib.sha256(body).hexdigest()
    checkpoint = json.dumps({"sha256": digest, "payload": json.loads(body)}, ensure_ascii=False,
                            sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(gzip.compress(checkpoint, mtime=0))
        if load_preflight(temporary_path) != json.loads(body):
            raise ValueError("checkpoint round-trip mismatch")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {"checkpoint": path.name, "checkpoint_sha256": digest, "checkpoint_verified": True}


def load_preflight(path):
    import gzip
    import hashlib

    checkpoint = json.loads(gzip.decompress(Path(path).read_bytes()))
    payload = checkpoint["payload"]
    if payload["schema_version"] != "aegis.training.preflight.v1":
        raise ValueError("unsupported preflight checkpoint")
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      allow_nan=False, separators=(",", ":")).encode()
    if hashlib.sha256(body).hexdigest() != checkpoint["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    return payload
