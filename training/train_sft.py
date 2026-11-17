"""Optional QLoRA SFT. --validate-only is stdlib and performs no model download."""
import argparse
import json
from .common import rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/mediator_sft_train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--output", default="training/runs/sft")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    data = rows(args.dataset)
    for row in data:
        if len(row["messages"]) != 3 or row["messages"][-1]["role"] != "assistant":
            raise ValueError("invalid SFT conversation")
        json.loads(row["messages"][-1]["content"])
    if args.validate_only:
        print(json.dumps({"validated": len(data), "gpu_used": False}))
        return 0
    if args.max_steps <= 0 or args.max_length < 128:
        raise ValueError("positive max_steps and max_length>=128 required")
    from .common import quantized_model
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
    model, tokenizer, bf16 = quantized_model(args.model)
    encoded = []
    for row in data:
        prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
        if full[:len(prompt)] != prompt:
            raise ValueError("chat template prefix mismatch; refuse incorrect label masking")
        if len(prompt) >= args.max_length:
            raise ValueError("prompt would truncate all assistant labels; increase --max-length")
        ids = full[:args.max_length]
        encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                        "labels": [-100] * len(prompt) + ids[len(prompt):]})
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
