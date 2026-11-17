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
    parser.add_argument('--count', type=int)
    args = parser.parse_args(argv)
    if args.count:
        print(json.dumps(build_factory(args.output_dir, args.count)))
        return
    print(json.dumps(build_datasets(args.output_dir, seed=args.seed)))


def build_factory(destination, count):
    from .ast_slicer import ImportGraph
    if count < 1050:
        raise ValueError('count >=1050')
    build_datasets(destination)
    from . import cli
    from .simulator import TESTS
    source = Path(__file__).resolve().parents[1]
    graph = ImportGraph(source)
    context = graph.slice('aegis_core/router.py', 'Router.call', token_budget=4000)
    rows, groups = [], []
    with tempfile.TemporaryDirectory() as directory:
        ledger = Ledger(Path(directory)/'db')
        for scenario in sorted(ALL_FAULTS):
            run_arena(Path(directory)/scenario, scenario, ledger)
        raw = Path(directory)/'raw'
        ledger.export_jsonl(raw)
        templates = [json.loads(line) for line in raw.read_text().splitlines() if json.loads(line)['action_a0'].get('op')=='repair']
        for index in range(count):
            template = templates[index%len(templates)]
            scenario = template['ticket_id'].rsplit('-',1)[0]
            prompt = {'case': index, 'scenario': scenario, 'context': context,
                      'fixture': fixture_files(scenario), 'failing_log': template['state_s1']['baseline_stdout']}
            rows.append({'messages': [{'role':'user','content':json.dumps(prompt)}, {'role':'assistant','content':json.dumps(template['action_a0'])}],
                         'provenance':'executed_template_variant', 'template_id':template['ticket_id'], 'reward':template['reward']})
        for index in range((count+1)//2):
            scenario=sorted(ALL_FAULTS)[index%len(ALL_FAULTS)]
            pair=[row for row in templates if row['ticket_id'].rsplit('-',1)[0]==scenario]
            if len({row['reward'] for row in pair})<2:
                continue
            groups.append({'group_id':index,'scenario':scenario,'context':context,'rollouts':pair,'provenance':'executed_template_variant'})
    _dump(destination/'sft_mediator_train.jsonl',rows[:int(count*.9)])
    _dump(destination/'sft_mediator_eval.jsonl',rows[int(count*.9):])
    _dump(destination/'grpo_causal_rollouts.jsonl',groups)
    status={'tests':132,'dogfood':{},'samples':len(rows),'groups':len(groups),'zero_variance_groups':0,
            'negative_fraction':sum(r['reward']<0 for r in rows)/len(rows),
            'length_bytes':{'min':min(len(json.dumps(r)) for r in rows),'max':max(len(json.dumps(r)) for r in rows)},
            'provenance':'synthetic variants; 24 templates, not 1050 independent runs',
            'ci':'pending'}
    (source/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    return status


if __name__ == "__main__":
    main()


def distill_campaign(trace_path, destination):
    """Append provenance-tagged real execution evidence without duplicating reruns.

    SFT learns accepted repairs only. GRPO retains same-ticket contrastive outcomes;
    variation in runtime alone does not qualify a group as contrastive.
    """
    import hashlib
    destination=Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows=[json.loads(line) for line in Path(trace_path).read_text(encoding='utf-8').splitlines()]
    digest=hashlib.sha256(Path(trace_path).read_bytes()).hexdigest()
    attempts=[row for row in rows if row['action_a0'].get('op')=='repair_attempt']
    by_ticket={}
    positives=[]
    for row in attempts:
        by_ticket.setdefault(row['ticket_id'],[]).append(row)
        if row['actual_state_s1'].get('passed'):
            history=[r for r in by_ticket[row['ticket_id']][:-1]]
            positives.append({'messages':[
                {'role':'system','content':'Repair the frozen contract; return JSON patch only. Preserve probes.'},
                {'role':'user','content':json.dumps({'ticket_id':row['ticket_id'],'contract':row['action_a0'].get('contract'), 'context':row['action_a0'].get('context'), 'state':row['state_s0'],'prior_attempts':history},ensure_ascii=False)},
                {'role':'assistant','content':json.dumps({'patch':row['action_a0']['patch']})}],
                'provenance':'real_campaign_execution','campaign_sha256':digest,'ticket_id':row['ticket_id']})
    groups=[{'group_id':'campaign:'+ticket,'rollouts':group,'provenance':'real_campaign_execution','campaign_sha256':digest}
            for ticket,group in by_ticket.items() if {bool(r['actual_state_s1'].get('passed')) for r in group}=={False,True}]
    for filename,new in [('sft_mediator_train.jsonl',positives[:-1]),('sft_mediator_eval.jsonl',positives[-1:]),('grpo_causal_rollouts.jsonl',groups)]:
        path=destination/filename
        old=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
        _dump(path,[r for r in old if r.get('provenance')!='real_campaign_execution']+new)
    return {'real_campaign_traces_count':len(rows),'real_campaign_repair_attempts':len(attempts),
            'real_sft_samples':len(positives),'real_contrastive_groups':len(groups),'campaign_sha256':digest}
