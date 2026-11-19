"""QLoRA supervised fine-tuning with a CPU dataset and checkpoint preflight."""
import argparse
import json

from .common import chat_tokens, conversation, dry_tokenizer, export_preflight, lora_settings, rows, validate_budget, verify_dataset

_SYSTEM = "Return the requested structured repair contract or patch as JSON."


def load_conversations(dataset, *, supplements=()):
    data = []
    for row in rows(dataset, supplements=supplements):
        messages = conversation(row["messages"], roles={("user", "assistant"), ("system", "user", "assistant")})
        if len(messages) == 2:
            messages = [{"role": "system", "content": _SYSTEM}, *messages]
        if not isinstance(json.loads(messages[-1]["content"]), dict):
            raise ValueError("assistant content must be a JSON object")
        data.append(messages)
    return data


def encode_sft(data, tokenizer, max_length):
    encoded = []
    prompt_truncations = completion_truncations = 0
    for messages in data:
        prompt = chat_tokens(tokenizer, messages[:-1], generation=True)
        full = chat_tokens(tokenizer, messages)
        if full[:len(prompt)] != prompt or len(full) <= len(prompt):
            raise ValueError("chat template prefix mismatch or missing assistant labels")
        completion = full[len(prompt):]
        prompt_budget = max_length - min(len(completion), max_length // 4)
        if len(prompt) > prompt_budget:
            # Keep the system prefix and the assistant generation delimiter.
            prefix = prompt_budget // 3
            prompt = prompt[:prefix] + prompt[-(prompt_budget - prefix):]
            prompt_truncations += 1
        remaining = max_length - len(prompt)
        completion_truncations += len(completion) > remaining
        ids = prompt + completion[:remaining]
        encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                        "labels": [-100] * len(prompt) + ids[len(prompt):]})
    return encoded, {"prompt_truncations": prompt_truncations, "completion_truncations": completion_truncations,
                     "max_tokens": max(len(row["input_ids"]) for row in encoded)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/mediator_sft_train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--output", default="training/runs/sft")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--tokenizer-dir", help="Local tokenizer files for dry-run")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    dataset_sha256, supplements = verify_dataset(args.dataset)
    data = load_conversations(args.dataset, supplements=supplements)
    if args.validate_only:
        print(json.dumps({"validated": len(data), "gpu_used": False}))
        return 0
    validate_budget(args.max_steps, args.max_length)
    lora = lora_settings(args.rank)
    if args.dry_run:
        tokenizer = dry_tokenizer(args.tokenizer_dir)
        encoded, metrics = encode_sft(data, tokenizer, args.max_length)
        tokenizer_id = args.model if args.tokenizer_dir else tokenizer.name_or_path
        checkpoint = export_preflight(args.output, "sft", {"model": args.model, "tokenizer": tokenizer_id,
                                     "dataset_sha256": dataset_sha256, "lora": lora, "max_length": args.max_length,
                                     "max_steps": args.max_steps, "rows": encoded})
        print(json.dumps({"validated": len(data), "gpu_used": False, "tokenizer": tokenizer_id,
                          "mode": "dry-run", "lora_rank": lora["r"], **metrics, **checkpoint}))
        return 0
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from .common import quantized_model

    model, tokenizer, bf16 = quantized_model(args.model, rank=args.rank)
    encoded, _ = encode_sft(data, tokenizer, args.max_length)
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=args.output, max_steps=args.max_steps,
                      per_device_train_batch_size=1, gradient_accumulation_steps=8, gradient_checkpointing=True,
                      learning_rate=2e-4, bf16=bf16, fp16=not bf16, optim="paged_adamw_8bit", report_to=[],
                      save_strategy="no", logging_steps=1, seed=0), train_dataset=Dataset.from_list(encoded),
                      data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True))
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
