"""Replay cat-cafe invariants against the repository's public Python example."""
from __future__ import annotations

from dataclasses import dataclass
import difflib
import hashlib
import json
from pathlib import Path
import tempfile

from .ast_slicer import ImportGraph
from .reward import compute_step_reward
from .sandbox import Sandbox
from .simulator import freeze_arena

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "cat_cafe_25d"
PROVENANCE = "public_cat_cafe_worktree_replay"


@dataclass(frozen=True)
class ReplayCase:
    path: str
    symbol: str
    intent: str
    old: str
    good: str
    bad: str
    probe: str


CASES = {
    "cat_replay_depth": ReplayCase(
        "depth_sorter.py", "DepthSorter.key", "Sort cat sprites by foot anchor and preserve the stable tie-break.",
        "cat.body_y*self.scale(cat)", "cat.foot_y", "-cat.foot_y",
        """from depth_sorter import DepthSorter
from entities import Cat

def test_stable_foot_anchor():
    cats = [Cat('z', 10, 1), Cat('b', 20, 0), Cat('a', 10, 999)]
    assert [cat.ident for cat in DepthSorter().order(cats)] == ['a', 'z', 'b']
"""),
    "cat_replay_history": ReplayCase(
        "render_loop.py", "RenderLoop.tick", "Bound retained draw history to the latest frame without replacing its list.",
        "self.history.append(tuple(self.draw_buffer))  # deliberately planted unbounded history",
        "self.history[:] = [tuple(self.draw_buffer)]", "self.history.append(tuple(self.draw_buffer[:1]))",
        """from entities import Cat
from render_loop import RenderLoop

def test_bounded_history():
    loop = RenderLoop([Cat('a', 1, 1)], capacity=2)
    history = loop.history
    for _ in range(100):
        loop.tick(.1)
    assert loop.history is history
    assert loop.history == [('a', None)]
"""),
    "cat_replay_interaction": ReplayCase(
        "cat_fsm.py", "CatFSM.update", "Keep cat state unchanged when the interaction clock pauses.",
        "if dt < 0: raise ValueError('negative dt')",
        "if dt < 0: raise ValueError('negative dt')\n        if dt == 0:\n            return",
        "if dt < 0: raise ValueError('negative dt')\n        if dt == 0 and cat.state == 'walking':\n            return",
        """import pytest
from cat_fsm import CatFSM
from entities import Cat

def test_paused_clock():
    machine = CatFSM()
    for state in ('idle', 'walking', 'resting'):
        cat = Cat('a', 1, 1, x=3, state=state)
        machine.update(cat, 0)
        assert (cat.state, cat.x) == (state, 3)
    with pytest.raises(ValueError):
        machine.update(cat, -1)
"""),
    "cat_replay_projection": ReplayCase(
        "geometry.py", "Viewport.project", "Reject nonfinite projection coordinates and keep finite projection round-trips.",
        "return (self.width/2+(x-y)*self.tile_width/2,",
        "if not math.isfinite(x) or not math.isfinite(y):\n            raise ValueError('nonfinite world coordinate')\n        return (self.width/2+(x-y)*self.tile_width/2,",
        "if not math.isfinite(x):\n            raise ValueError('nonfinite world coordinate')\n        return (self.width/2+(x-y)*self.tile_width/2,",
        """import math
import pytest
from geometry import Viewport

def test_projection_inputs():
    view = Viewport(320, 200)
    assert view.unproject(*view.project(-2, 7)) == (-2, 7)
    for position in ((math.inf, 0), (0, math.nan)):
        with pytest.raises(ValueError):
            view.project(*position)
"""),
}


def create_replay_arena(root, scenario):
    case = CASES[scenario]
    sources = {path.name: path.read_text(encoding="utf-8") for path in sorted(EXAMPLE.glob("*.py"))
               if not path.name.startswith("test_")}
    return freeze_arena(root, scenario, sources, case.probe, target=case.path, source_dirs=(".",))


def _candidate(source, case, positive):
    if source.count(case.old) != 1:
        raise ValueError("cat-cafe replay baseline changed")
    revised = source.replace(case.old, case.good if positive else case.bad, 1)
    return "".join(difflib.unified_diff(source.splitlines(keepends=True), revised.splitlines(keepends=True),
                                       fromfile="a/" + case.path, tofile="b/" + case.path))


def run_replays():
    """Run authored candidates against frozen probes and return trainable rows."""
    sft, grpo, evidence = [], [], []
    for scenario, case in CASES.items():
        with tempfile.TemporaryDirectory(prefix="aegis-cat-") as temporary:
            root = Path(temporary) / "arena"
            frozen = create_replay_arena(root, scenario)
            context = ImportGraph(root).slice(case.path, case.symbol, token_budget=1800)
            prompt = [{"role": "system", "content": 'Return JSON {"patch":"unified diff"}. Preserve the frozen tests.'},
                      {"role": "user", "content": json.dumps({"intent": case.intent, "context": context,
                                                              "frozen_probe": case.probe}, sort_keys=True)}]
            source = (root / case.path).read_text(encoding="utf-8")
            manager = Sandbox(root)
            for positive in (False, True):
                ticket = "positive" if positive else "negative"
                patch = _candidate(source, case, positive)
                manager.create(ticket)
                try:
                    result = manager.repair(ticket, patch, fail_before_cmd=frozen["test_cmd"],
                                            pass_after_cmd=frozen["test_cmd"], source_dirs=frozen["source_dirs"])
                finally:
                    manager.cleanup(ticket)
                if not result["baseline_failed"] or result["passed"] != positive or result["final"] is None:
                    raise AssertionError(f"physical probes disagreed with {scenario} {ticket}")
                # Keep training labels independent of machine speed, as the offline simulator does.
                reward = compute_step_reward(baseline_failed=result["baseline_failed"], after_passed=result["after_passed"],
                                             target_files=frozen["target_files"], changed_files=result["changed_files"],
                                             diff_lines=result["diff_lines"], elapsed_sec=0.0, latency_budget_sec=60,
                                             cheat_detected=result["cheat_detected"])
                completion = json.dumps({"patch": patch})
                observed = {"fail_before_exit": result["baseline"]["exit_code"],
                            "pass_after_exit": result["final"]["exit_code"], "reward": reward}
                grpo.append({"group_id": scenario, "scenario": scenario, "prompt": prompt, "completion": completion,
                             "reward": reward, "positive": positive, "physical_gate": observed,
                             "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(), "template_verified": True,
                             "provenance": PROVENANCE})
                evidence.append({"scenario": scenario, "candidate": ticket, **observed})
                if positive:
                    sft.append({"id": scenario, "scenario": scenario, "provenance": PROVENANCE,
                                "messages": [*prompt, {"role": "assistant", "content": completion}]})
    return sft, grpo, evidence


def update_dataset(destination):
    """Blend new measured rows with the v3 base, preserving its evaluation split."""
    from .dataset_builder import _dump

    destination = Path(destination)
    sft, grpo, evidence = run_replays()
    manifest_path = destination / "manifest_v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supplements = {"mediator_sft_train.jsonl": "cat_cafe_replay_sft.jsonl",
                   "grpo_repair_pairs.jsonl": "cat_cafe_replay_grpo.jsonl"}
    for name, additions in zip(supplements.values(), (sft, grpo)):
        path = destination / name
        _dump(path, additions)
        manifest["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["cat_cafe_replay"] = {"source": "examples/cat_cafe_25d", "scenarios": sorted(CASES),
                                   "physical_attempts": len(evidence), "sft_rows": len(sft), "grpo_rows": len(grpo),
                                   "supplements": supplements,
                                   "reward_policy": "deterministic_zero_latency_dual_probe_reward"}
    manifest["training_totals"] = {"sft_train": manifest["sft_train"] + len(sft),
                                   "sft_eval": manifest["sft_eval"],
                                   "grpo_candidates": manifest["grpo_candidates"] + len(grpo),
                                   "grpo_groups": manifest["grpo_groups"] + len(grpo) // 2}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest, evidence
