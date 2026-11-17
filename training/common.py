"""Stdlib dataset validation and explicit GPU preflight for optional trainers."""
import json
from pathlib import Path


def rows(path):
    values = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
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
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=.05,
                                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    return model, tokenizer, bf16
