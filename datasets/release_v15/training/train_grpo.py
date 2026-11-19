"""Optional online GRPO: reward physically evaluates generated offline patches."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
import tempfile
from .common import chat_tokens, conversation, dry_tokenizer, export_preflight, lora_settings, rows, validate_budget, verify_dataset


def score_patch(scenario, completion):
    from aegis_core.simulator import create_arena, ALL_FAULTS
    from aegis_core.sandbox import Sandbox
    from aegis_core.reward import compute_step_reward
    from aegis_core.cat_cafe_replay import CASES, create_replay_arena
    if scenario not in ALL_FAULTS and scenario not in CASES:
        return -2.0
    try:
        text = completion[-1]["content"] if isinstance(completion, list) else completion
        patch = json.loads(text)["patch"]
        if not isinstance(patch, str):
            return -2.0
        with tempfile.TemporaryDirectory(prefix="aegis-grpo-") as directory:
            root = Path(directory) / "arena"
            contract = (create_arena if scenario in ALL_FAULTS else create_replay_arena)(root, scenario)
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


def load_groups(dataset, *, supplements=()):
    groups = {}
    for row in rows(dataset, supplements=supplements):
        for name in ("group_id", "scenario"):
            if type(row.get(name)) is not str or not row[name]:
                raise ValueError(f"GRPO row requires {name}")
        conversation(row["prompt"], roles={("user",), ("system", "user")})
        completion = json.loads(row["completion"])
        if not isinstance(completion, dict) or type(completion.get("patch")) is not str or not completion["patch"]:
            raise ValueError("GRPO completion requires a nonempty patch")
        patch_sha256 = hashlib.sha256(completion["patch"].encode()).hexdigest()
        if row.get("patch_sha256", patch_sha256) != patch_sha256:
            raise ValueError("GRPO patch checksum mismatch")
        if type(row.get("reward")) not in (int, float) or not math.isfinite(row["reward"]):
            raise ValueError("GRPO reward must be finite")
        groups.setdefault(row["group_id"], []).append(row)
    for group in groups.values():
        if (len(group) != 2 or not min(row["reward"] for row in group) < 0 < max(row["reward"] for row in group)
                or group[0]["prompt"] != group[1]["prompt"] or group[0]["scenario"] != group[1]["scenario"]):
            raise ValueError("expected matching prompts with positive/negative candidate pairs")
    return groups


def prepare_groups(groups, tokenizer, max_prompt_length, max_completion_length):
    prepared = []
    prompt_truncations = completion_truncations = 0
    for group_id, group in groups.items():
        prompt = chat_tokens(tokenizer, group[0]["prompt"], generation=True)
        prompt_truncations += len(prompt) > max_prompt_length
        rewards = [row["reward"] for row in group]
        mean, deviation = statistics.mean(rewards), statistics.stdev(rewards)
        candidates = []
        for row in group:
            full = chat_tokens(tokenizer, [*row["prompt"], {"role": "assistant", "content": row["completion"]}])
            if full[:len(prompt)] != prompt or len(full) <= len(prompt):
                raise ValueError("chat template prefix mismatch or missing GRPO completion")
            completion = full[len(prompt):]
            completion_truncations += len(completion) > max_completion_length
            candidates.append({"completion_ids": completion[:max_completion_length], "reward": row["reward"],
                               "advantage": (row["reward"] - mean) / (deviation + 1e-4)})
        prepared.append({"group_id": group_id, "scenario": group[0]["scenario"],
                         "prompt_ids": prompt[-max_prompt_length:], "candidates": candidates})
    return prepared, {"prompt_truncations": prompt_truncations, "completion_truncations": completion_truncations,
                      "reward_groups_normalized": len(prepared),
                      "max_prompt_tokens": max(len(group["prompt_ids"]) for group in prepared),
                      "max_completion_tokens": max(len(row["completion_ids"]) for group in prepared
                                                   for row in group["candidates"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/grpo_repair_pairs.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--output", default="training/runs/grpo")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--tokenizer-dir", help="Local tokenizer files for dry-run")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    dataset_sha256, supplements = verify_dataset(args.dataset)
    groups = load_groups(args.dataset, supplements=supplements)
    counts = {"candidates": sum(map(len, groups.values())), "groups": len(groups), "gpu_used": False}
    if args.validate_only:
        print(json.dumps(counts))
        return 0
    validate_budget(args.max_steps, args.max_prompt_length, args.max_completion_length)
    lora = lora_settings(args.rank)
    if args.dry_run:
        tokenizer = dry_tokenizer(args.tokenizer_dir)
        prepared, metrics = prepare_groups(groups, tokenizer, args.max_prompt_length, args.max_completion_length)
        tokenizer_id = args.model if args.tokenizer_dir else tokenizer.name_or_path
        checkpoint = export_preflight(args.output, "grpo", {"model": args.model, "tokenizer": tokenizer_id,
                                     "dataset_sha256": dataset_sha256, "lora": lora, "max_steps": args.max_steps,
                                     "max_prompt_length": args.max_prompt_length,
                                     "max_completion_length": args.max_completion_length, "groups": prepared})
        print(json.dumps({**counts, "tokenizer": tokenizer_id, "mode": "dry-run",
                          "lora_rank": lora["r"], **metrics, **checkpoint}))
        return 0
    from .common import quantized_model
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    model, tokenizer, bf16 = quantized_model(args.model, rank=args.rank)
    tokenizer.padding_side = "left"
    train = Dataset.from_list([{"prompt": group[0]["prompt"], "scenario": group[0]["scenario"]} for group in groups.values()])
    def reward(completions, scenario, **kwargs):
        return [score_patch(name, text) for name, text in zip(scenario, completions)]
    trainer = GRPOTrainer(model=model, reward_funcs=reward, processing_class=tokenizer,
                          args=GRPOConfig(output_dir=args.output, max_steps=args.max_steps, per_device_train_batch_size=2,
                              gradient_accumulation_steps=4, num_generations=2, max_prompt_length=args.max_prompt_length,
                              max_completion_length=args.max_completion_length,
                              gradient_checkpointing=True, bf16=bf16, fp16=not bf16, beta=0., use_vllm=False,
                              report_to=[], save_strategy="no", logging_steps=1, seed=0), train_dataset=train)
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
