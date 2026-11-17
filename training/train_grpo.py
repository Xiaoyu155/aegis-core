"""Optional online GRPO: reward physically evaluates generated offline patches."""
import argparse
import json
from pathlib import Path
import tempfile
from .common import rows


def score_patch(scenario, completion):
    from aegis_core.simulator import create_arena, ALL_FAULTS
    from aegis_core.sandbox import Sandbox
    from aegis_core.reward import compute_step_reward
    if scenario not in ALL_FAULTS:
        return -2.0
    manager = None
    try:
        text = completion[-1]["content"] if isinstance(completion, list) else completion
        patch = json.loads(text)["patch"]
        if not isinstance(patch, str):
            return -2.0
        with tempfile.TemporaryDirectory(prefix="aegis-grpo-") as directory:
            root = Path(directory) / "arena"
            contract = create_arena(root, scenario)
            manager = Sandbox(root)
            manager.create("sample")
            try:
                result = manager.repair("sample", patch, fail_before_cmd=contract["test_cmd"],
                                        pass_after_cmd=contract["test_cmd"], source_dirs=contract["source_dirs"], timeout=60)
                return compute_step_reward(baseline_failed=result["baseline_failed"], after_passed=result["after_passed"],
                    target_files=contract["target_files"], changed_files=result["changed_files"], diff_lines=result["diff_lines"],
                    elapsed_sec=result["elapsed_sec"], latency_budget_sec=60, cheat_detected=result["cheat_detected"])
            finally:
                manager.cleanup("sample")
    except (ValueError, KeyError, TypeError, OSError, __import__("subprocess").SubprocessError):
        return -2.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/grpo_repair_pairs.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--output", default="training/runs/grpo")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    data = rows(args.dataset)
    groups = {}
    for row in data:
        groups.setdefault(row["group_id"], []).append(row)
    if any(len(group) != 2 or not min(r["reward"] for r in group) < 0 < max(r["reward"] for r in group) for group in groups.values()):
        raise ValueError("expected positive/negative candidate pairs")
    if args.validate_only:
        print(json.dumps({"candidates": len(data), "groups": len(groups), "gpu_used": False}))
        return 0
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    from .common import quantized_model
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    model, tokenizer, bf16 = quantized_model(args.model, rank=4)
    tokenizer.padding_side = "left"
    train = Dataset.from_list([{"prompt": group[0]["prompt"], "scenario": group[0]["scenario"]} for group in groups.values()])
    def reward(completions, scenario, **kwargs):
        return [score_patch(name, text) for name, text in zip(scenario, completions)]
    trainer = GRPOTrainer(model=model, reward_funcs=reward, processing_class=tokenizer,
                          args=GRPOConfig(output_dir=args.output, max_steps=args.max_steps, per_device_train_batch_size=2,
                              gradient_accumulation_steps=4, num_generations=2, max_prompt_length=768, max_completion_length=256,
                              gradient_checkpointing=True, bf16=bf16, fp16=not bf16, beta=0., use_vllm=False,
                              report_to=[], save_strategy="no", logging_steps=1, seed=0), train_dataset=train)
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
