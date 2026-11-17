"""Build synthetic medium SFT and template-verified paired repair datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import tempfile

from .ledger import Ledger
from .mediator import SYSTEM_PROMPT, compile_request
from .simulator import ALL_FAULTS, TESTS, REPAIR_SYSTEM_PROMPT, fixture_files, fixture_patch, run_arena


def _dump(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def build_datasets(destination: Path, *, seed: int = 0, verify_templates: bool = True) -> dict:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    if verify_templates:
        with tempfile.TemporaryDirectory(prefix="aegis-dataset-") as directory:
            ledger = Ledger(Path(directory) / "ledger.sqlite")
            for scenario in sorted(ALL_FAULTS):
                outcomes[scenario] = run_arena(Path(directory) / scenario, scenario, ledger)
    sft = []
    scenarios = sorted(ALL_FAULTS)
    for index in range(520):
        scenario = scenarios[index % len(scenarios)]
        source = fixture_files(scenario)
        target = ALL_FAULTS[scenario][0]
        # Distinct task parameters, not made-up measurements of real executions.
        cap = 100 + index * 7
        drawdown = round(.01 + (index % 40) / 1000, 3)
        request = {"ticket_id": f"medium-{index:04}", "wish": f"实盘交易风控：资金上限 {cap}，回撤阈值 {drawdown}" if scenario.startswith("quant_") else f"修复 {scenario}，验收编号 {index}",
                   "repo_skeleton": list(source), "proposal": {"target_files": [target], "contract_stub": "def repair() -> None:\n    ..."},
                   "constraints": {"execution_mode": "live", "credential_ref": "env:EXCHANGE_KEY", "credential_available": True,
                                   "max_notional": cap, "max_drawdown": drawdown, "position_mode": "one_way"} if scenario.startswith("quant_") else {}}
        if index % 3 == 0 and scenario.startswith("quant_"):
            request["constraints"].pop("max_notional")
        response = compile_request(**request)
        prompt = json.dumps(request, ensure_ascii=False, sort_keys=True)
        sft.append({"id": request["ticket_id"], "scenario": scenario, "provenance": "synthetic_compiler_validated",
                    "request": request, "expected": response, "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}, {"role": "assistant", "content": json.dumps(response, ensure_ascii=False, sort_keys=True)}]})
    random.Random(seed).shuffle(sft)
    _dump(destination / "mediator_sft_train.jsonl", sft[:468])
    _dump(destination / "mediator_sft_eval.jsonl", sft[468:])
    grpo = []
    instructions = ["最小化修复", "保持所有接口不变", "保留冻结验收断言", "只修复目标源文件", "避免无关重构"]
    for group_index in range(260):
        scenario = scenarios[group_index % len(scenarios)]
        prompt = [{"role": "system", "content": 'Return exactly JSON {"patch":"unified diff"}. Preserve frozen tests and interfaces.'},
                  {"role": "user", "content": json.dumps({"scenario": scenario, "instruction": instructions[group_index % len(instructions)],
                    "case_id": group_index, "source": fixture_files(scenario), "frozen_test": TESTS[scenario]}, ensure_ascii=False, sort_keys=True)}]
        for index, positive in enumerate((False, True)):
            patch = fixture_patch(scenario, positive=positive)
            # Never forge a physical measurement when verification was disabled.
            reward = outcomes[scenario][index]["reward"] if verify_templates else None
            grpo.append({"group_id": f"{scenario}-{group_index:04}", "scenario": scenario, "prompt": prompt,
                         "completion": json.dumps({"patch": patch}), "reward": reward, "positive": positive,
                         "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(), "template_verified": verify_templates,
                         "provenance": "synthetic_instruction_variant_of_executed_template" if verify_templates else "unverified_synthetic_template"})
    _dump(destination / "grpo_repair_pairs.jsonl", grpo)
    manifest = {"schema_version": "aegis.dataset.v3", "seed": seed, "fault_modes": scenarios,
                "sft_total": 520, "sft_train": 468, "sft_eval": 52, "grpo_candidates": 520, "grpo_groups": 260,
                "executed_templates": len(outcomes), "physical_template_attempts": len(outcomes) * 2,
                "provenance": "compiler_validated_synthetic_SFT_and_instruction_augmented_template_pairs",
                "sha256": {name: hashlib.sha256((destination / name).read_bytes()).hexdigest() for name in
                           ("mediator_sft_train.jsonl", "mediator_sft_eval.jsonl", "grpo_repair_pairs.jsonl")}}
    (destination / "manifest_v3.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def export_dataset(destination: Path, *, seed: int = 0, verify_templates: bool = True) -> dict:
    return build_datasets(destination, seed=seed, verify_templates=verify_templates)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(build_datasets(args.output_dir, seed=args.seed)))


if __name__ == "__main__":
    main()
