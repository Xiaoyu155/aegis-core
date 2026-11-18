"""Cross-repository predictions, native frontend review and frozen render probes."""
import difflib
import hashlib
import importlib.resources
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from aegis_core.ast_slicer import ImportGraph
from aegis_core.cli import main
from aegis_core.contract_lock import InterfaceContractLock
from aegis_core.experience_bank import ExperienceBank
from aegis_core.frontend import css_references, local_reference, parse_html
from aegis_core.ledger import Ledger
from aegis_core.mediator import is_business_source
from aegis_core.probes import classify_probe_failure, freeze_probes, verify_probe_files
from aegis_core.repo_adapter import attach_repository, load_tool_source_globs, validate_tool_source_globs
from aegis_core.reviewer import audit_candidate
from aegis_core.runner import diagnose
from aegis_core.sandbox import detect_source_cheat, run_exam_gate
from aegis_core.world_model import CausalWorldModel, encoder_digest, bundled_replay_anchors, load_repository_model


CONTROLLER = Path(__file__).resolve().parents[1]
ARTIFACTS = CONTROLLER / "artifacts" / "causal_world_model"


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, text=True,
                          encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def write(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def repository(root, files):
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "core.autocrlf", "false")
    git(root, "config", "user.name", "Aegis test")
    git(root, "config", "user.email", "aegis-test@localhost")
    for relative, content in files.items():
        write(root, relative, content)
    git(root, "add", "--all")
    git(root, "commit", "-qm", "baseline")
    return root


def call(capsys, *args):
    code = main([str(value) for value in args])
    output = capsys.readouterr()
    return code, json.loads(output.out or output.err)


def patch(directory, name, relative, before, after):
    delta = difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                 fromfile=f"a/{relative}", tofile=f"b/{relative}")
    return write(directory, name, "".join(delta))


@pytest.mark.parametrize("local,override,expected", [
    (False, None, "bundled"),
    (True, None, "repository"),
    (True, "valid", "environment"),
    (False, "valid", "environment"),
    (True, "missing", None),
])
def test_repository_model_selection(tmp_path, monkeypatch, local, override, expected):
    monkeypatch.delenv("AEGIS_WORLD_MODEL_DIR", raising=False)
    root = tmp_path / "external"
    root.mkdir()
    if local:
        local_dir = root / "artifacts" / "causal_world_model"
        local_dir.mkdir(parents=True)
        for name in ("model_weights.json", "feature_encoder.json"):
            shutil.copyfile(ARTIFACTS / name, local_dir / name)
    if override:
        monkeypatch.setenv("AEGIS_WORLD_MODEL_DIR", str(ARTIFACTS if override == "valid" else tmp_path / "absent"))
    model = load_repository_model(root)
    assert (model.model_source if model else None) == expected
    if model is not None:
        assert model.model_version == CausalWorldModel.load(ARTIFACTS).model_version
        assert model.predict({"status": "running"}, {"op": "repair_attempt"})["predicted_next_state"]["outcome"]


def test_package_contains_exact_shipped_seed_and_compatible_anchors():
    data = importlib.resources.files("aegis_core").joinpath("data")
    for name in ("feature_encoder.json", "model_weights.json"):
        assert data.joinpath(name).read_bytes() == (ARTIFACTS / name).read_bytes()
    model = CausalWorldModel.load(ARTIFACTS)
    pack = bundled_replay_anchors()
    assert pack["encoder_sha256"] == encoder_digest(model.encoder)
    assert pack["source_weights_sha256"] == hashlib.sha256(data.joinpath("model_weights.json").read_bytes()).hexdigest()
    assert 0 < len(pack["training"]) <= 24 * len(model.network.state_labels)
    assert 0 < len(pack["validation"]) <= 96
    assert all(len(row["features"]) == model.encoder.input_size for row in pack["training"] + pack["validation"])


@pytest.mark.parametrize("attribute,executed", [
    ("", True), ('type="module"', True), ('TYPE="Text/JavaScript; charset=utf-8"', True),
    ('type="application/ecmascript"', True), ('type="application/json"', False),
    ('type="importmap"', False), ('type="text/template"', False), ('src="./external.js"', False),
])
def test_frontend_script_types_and_lines(attribute, executed):
    html = f"<!doctype html>\n<script\n  {attribute}>\nfunction render() {{}}\n</script>\n"
    blocks, references = parse_html(html)
    assert bool(blocks) == executed
    if executed:
        assert blocks[0].start_line == 3
        assert blocks[0].source.startswith("\nfunction")
    if "src" in attribute:
        assert (references[0].value, references[0].line) == ("./external.js", 2)


@pytest.mark.parametrize("value,expected", [
    ("../styles/base.css?v=1#main", "styles/base.css"), ("/ui/render.js", "ui/render.js"),
    ("../assets/cat%20cafe.png", "assets/cat cafe.png"), ("../../outside.js", None),
    ("https://cdn.example.com/script.js", None), ("//cdn.example.com/script.js", None),
    ("#room", None),
])
def test_browser_local_reference_resolution(value, expected):
    assert local_reference("ui/preview.html", value) == expected


def test_frontend_graph_and_locks_share_the_html_entry(tmp_path):
    write(tmp_path, "render.js", "export function render(a) {return a}\n")
    write(tmp_path, "theme/base.css", "body {color: blue}")
    write(tmp_path, "theme/app.css", '@import "base.css";\nbody { background: url("/assets/room.png") }')
    write(tmp_path, "assets/room.png", "fixture")
    page = write(tmp_path, "preview.html", """<!doctype html>
<!-- <script src="ignored.js"></script> -->
<link rel="stylesheet" href="/theme/app.css?theme=1">
<script src="./render.js"></script>
<script type="module">
import { render } from './render.js';
function publicRender(value) { return render(value); }
window.room = publicRender;
</script>
""")
    graph = ImportGraph(tmp_path)
    assert graph.file_dependencies["preview.html"] == ["render.js", "theme/app.css"]
    assert graph.file_dependencies["theme/app.css"] == ["assets/room.png", "theme/base.css"]
    symbol = graph.slice("preview.html", line=7)["items"][0]
    assert symbol["start_line"] == 7 and symbol["file"] == "preview.html"
    lock = InterfaceContractLock.capture(graph, "baseline", ["preview.html", "render.js"])
    assert {("preview.html", "room", "signature"), ("render.js", "render", "signature")} <= {
        (item.path, item.symbol, item.kind) for item in lock.entries}
    page.write_text(page.read_text().replace("publicRender(value)", "publicRender(value, zoom)"), encoding="utf-8")
    assert any(item["symbol"] == "room" and item["kind"] == "signature"
               for item in lock.violations(tmp_path, ["preview.html"]))


def test_css_reference_parser_ignores_comments_and_string_literals():
    refs = list(css_references('/* @import "fake.css"; */\n@import "real.css";\n'
                               'p::after { content: "url(fake.png)"; color: red; }\n'
                               'main { background: url("assets/room.png"); }'))
    assert [(ref.value, ref.line) for ref in refs] == [("real.css", 2), ("assets/room.png", 4)]


@pytest.mark.parametrize("path,expected", [
    ("tools/bake.py", True), ("tools/render/pack.cjs", True), ("宠物小程序/tools/bake.html", True),
    ("tools/render/site.css", True), ("tools/test_bake.py", False), ("tools/pack.test.cjs", False),
    ("tools/manifest.json", False), ("tools/config/settings.js", False), ("scripts/bake.py", False),
])
def test_default_tool_source_policy(path, expected):
    assert is_business_source(path) == expected


@pytest.mark.parametrize("pattern", ["/outside/**/*.py", "../outside.py", "tools\\*.py", "tools//*.py"])
def test_tool_source_profile_rejects_ambiguous_paths(pattern):
    with pytest.raises(ValueError, match="repository-relative"):
        validate_tool_source_globs([pattern])


def test_custom_tool_source_profile_does_not_authorize_tests(tmp_path):
    root = repository(tmp_path / "external", {"scripts/build/pack.py": "COUNT = 0\n", "preview.html": "<script>1</script>\n",
                                            "scripts/build/test_pack.py": "VALUE = 1\n"})
    profile = attach_repository(root, tool_source_globs=["scripts/build/**/*.py"])["profile"]
    assert profile["entry_modules"] == ["preview.html"]
    assert attach_repository(root)["profile"]["tool_source_globs"] == ["scripts/build/**/*.py"]
    assert is_business_source("scripts/build/pack.py", tool_source_globs=load_tool_source_globs(root))
    assert not is_business_source("scripts/build/test_pack.py", tool_source_globs=load_tool_source_globs(root))
    # Profiles are local configuration and should not themselves count as patch scope.
    git(root, "add", ".aegis/repo_profile.json")
    git(root, "commit", "-qm", "record local scope")
    write(root, "scripts/build/pack.py", "COUNT = 1\n")
    assert detect_source_cheat(root, ("scripts",))["source_changed_lines"] == 2
    write(root, "scripts/build/test_pack.py", "VALUE = 2\n")
    assert detect_source_cheat(root, ("scripts",))["cheat_detected"]


@pytest.mark.parametrize("kind,code,output,expected", [
    ("unit", 1, "AssertionError: expected y", "ASSERTION_FAILED"),
    ("browser_visual", 1, "node:assert ERR_ASSERTION", "ASSERTION_FAILED"),
    ("unit", 1, "ModuleNotFoundError: No module named 'game'", "MODULE_NOT_IMPLEMENTED"),
    ("browser_visual", 1, "TypeError: player.start is not a function", "MODULE_NOT_IMPLEMENTED"),
    ("browser_visual", 1, "page.waitForFunction: Timeout 45000ms exceeded.", "BROWSER_WAIT_TIMEOUT"),
    ("unit", 1, "TimeoutError", "EXECUTION_ERROR"),
    ("browser_visual", 1, "browserType.launch: Executable doesn't exist", "ENVIRONMENT_UNAVAILABLE"),
    ("browser_visual", 1, "Error: Cannot find module 'playwright'", "ENVIRONMENT_UNAVAILABLE"),
    ("unit", 127, "command not found", "ENVIRONMENT_UNAVAILABLE"),
    ("browser_visual", 124, "process group killed", "TIMEOUT_KILLED"),
])
def test_probe_failure_codes_are_in_diagnoses(tmp_path, kind, code, output, expected):
    assert classify_probe_failure(code, output, kind=kind) == expected
    result = diagnose(tmp_path, {"final": {"probe_kind": kind, "exit_code": code, "stderr": output}}, attempt=1)
    assert result["failure_code"] == expected and result["probe_kind"] == kind


def test_browser_probe_requires_script_and_freezes_selectors(tmp_path):
    script = write(tmp_path, "tests/render.py", "def test_view(): pass\n")
    with pytest.raises(ValueError, match="at least one frozen"):
        freeze_probes(tmp_path, [["runner", "-c", "code"]], kind="browser_visual")
    arg = str(script) + "::test_view"
    commands, hashes = freeze_probes(tmp_path, [["runner", arg]], kind="browser_visual")
    assert commands == [["runner", "./tests/render.py::test_view"]]
    assert hashes == {"tests/render.py": hashlib.sha256(script.read_bytes()).hexdigest()}
    verify_probe_files(tmp_path, hashes)
    script.unlink()
    with pytest.raises(ValueError, match="PROBE_INTEGRITY_VIOLATION"):
        verify_probe_files(tmp_path, hashes)


def test_gate_checks_external_script_after_execution(tmp_path):
    root = tmp_path / "worktree"
    root.mkdir()
    script = write(tmp_path, "external.py", "from pathlib import Path\nPath(__file__).write_text('changed')\n")
    [argv], hashes = freeze_probes(root, [[sys.executable, str(script)]], kind="browser_visual")
    result = run_exam_gate(root, argv, probe_kind="browser_visual", probe_script_sha256=hashes)
    assert result["exit_code"] == 0 and result["failure_code"] == "PROBE_INTEGRITY_VIOLATION"
    assert result["passed"] is False


def test_tracked_script_lock_survives_git_crlf_checkout_and_pins_runtime_bytes(tmp_path):
    relative = "tests/render.py"
    root = repository(tmp_path / "external", {relative: "print('render probe')\n"})
    git(root, "config", "core.autocrlf", "true")
    [argv], hashes = freeze_probes(root, [[sys.executable, relative]], kind="browser_visual")
    worktree = tmp_path / "candidate"
    git(root, "worktree", "add", "--detach", str(worktree), "HEAD")
    before_bytes = (root / relative).read_bytes()
    assert b"\r\n" not in before_bytes
    assert (worktree / relative).read_bytes() == before_bytes.replace(b"\n", b"\r\n")
    before = run_exam_gate(worktree, argv, probe_kind="browser_visual", probe_script_sha256=hashes)
    assert before["passed"] and before["probe_runtime_sha256"][relative] != hashes[relative]
    (worktree / relative).write_bytes(before_bytes)
    after = run_exam_gate(worktree, argv, probe_kind="browser_visual", probe_script_sha256=hashes,
                          probe_runtime_sha256=before["probe_runtime_sha256"])
    assert after["failure_code"] == "PROBE_INTEGRITY_VIOLATION"
    assert after["stdout"] == ""


def test_external_script_lock_does_not_normalize_line_endings(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    script = tmp_path / "visual.py"
    script.write_bytes(b"print('render probe')\r\n")
    _, hashes = freeze_probes(root, [[sys.executable, str(script)]], kind="browser_visual")
    assert hashes[str(script.resolve())] == hashlib.sha256(script.read_bytes()).hexdigest()
    script.write_bytes(b"print('render probe')\n")
    with pytest.raises(ValueError, match="PROBE_INTEGRITY_VIOLATION"):
        verify_probe_files(root, hashes)


def test_cli_rejects_probe_changed_after_wish(tmp_path, capsys):
    root = repository(tmp_path / "external", {"preview.html": "<script>function render() {}</script>\n"})
    script = write(tmp_path, "browser-check.py", "assert False\n")
    manifest = tmp_path / "manifest.json"
    db = tmp_path / "ledger.sqlite"
    argv = json.dumps(["{python}", str(script)])
    code, frozen = call(capsys, "wish", "--repo", root, "--no-auto-probe", "--probe-kind", "browser_visual",
                        "--fail-before", argv, "--pass-after", argv, "--target", "preview.html",
                        "--out", manifest, "--db", db, "--patch", tmp_path / "unused.diff", "repair browser entry")
    assert code == 0 and str(script.resolve()) in frozen["probe_script_sha256"]
    script.write_text("assert True\n", encoding="utf-8")
    code, result = call(capsys, "run", "--contract", manifest, "--db", db)
    assert code == 1 and "PROBE_INTEGRITY_VIOLATION" in result["message"]
    assert result["telemetry"]["predictions_made"] == 0
    assert Ledger(db).get_ticket(frozen["ticket_id"])["status"] == "PRECHECK_FAILED"


def test_native_frontend_review_reports_original_lines_and_only_new_findings(tmp_path):
    before = """<!doctype html>
<script type="application/json">this.ignore.push(1)</script>
<script>const prior = []; function previous() { prior.push(1); }</script>
<script>
class Loop { tick() {} }
</script>
"""
    root = repository(tmp_path / "repo", {"tools/bake_all_clips.html": before})
    path = "tools/bake_all_clips.html"
    write(root, path, before.replace("tick() {}", "tick() { this.frames.push(1); }"))
    review = audit_candidate(root, git(root, "rev-parse", "HEAD"), [path])
    assert review["audited_files"] == [path]
    assert [(item["code"], item["symbol"], item["line"]) for item in review["findings"]] == [
        ("UNBOUNDED_GROWTH", "frames", 5)]


def test_native_css_audit_reports_a_new_unsafe_reference(tmp_path):
    root = repository(tmp_path / "repo", {"ui.css": "main { color: blue; }\n"})
    write(root, "ui.css", 'main { color: blue; }\n@import "javascript:demo";\n')
    result = audit_candidate(root, git(root, "rev-parse", "HEAD"), ["ui.css"])
    assert result["findings"][0]["code"] == "UNSAFE_FRONTEND_URL"
    assert result["findings"][0]["line"] == 2


def test_cross_repository_attachment_plan_dag_review_and_recovery(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("AEGIS_WORLD_MODEL_DIR", raising=False)
    entry = "tools/bake_all_clips.html"
    before = """<!doctype html>
<body data-ready="no">
<script>
class Room { draw(frame) { return frame; } }
</script>
"""
    root = repository(tmp_path / "external", {entry: before, "preview.html": "<!doctype html>\n",
                                            "draw.js": "const ready = true;\n", "style.css": "main{color:blue}\n"})
    base = git(root, "rev-parse", "HEAD")
    assert call(capsys, "attach", "--repo", root)[0] == 0
    plan_file = tmp_path / "dag.json"
    db = tmp_path / "ledger.sqlite"
    code, planned = call(capsys, "plan", f"Fix {entry} ready state", "--repo", root, "--out", plan_file)
    assert code == 0 and planned["telemetry"]["model_source"] == "bundled"
    dag = json.loads(plan_file.read_text(encoding="utf-8"))
    task = next(task for task in dag["tasks"] if task["kind"] == "repair")
    assert task["contract"]["target_files"] == [entry]
    assert task["prediction"]["predicted_next_state"]
    probe = write(tmp_path, "probe.py", f"from pathlib import Path\n"
                  f"assert 'data-ready=\"yes\"' in Path({entry!r}).read_text(encoding='utf-8')\n")
    bad = before.replace('data-ready="no"', 'data-ready="yes"').replace(
        "return frame;", "this.frames.push(frame); return frame;")
    good = bad.replace("this.frames.push(frame);", "this.frames.push(frame); if(this.frames.length > 4) this.frames.shift();")
    task["contract"]["target_files"] += ["preview.html", "draw.js", "style.css"]  # Four targets require native L3.
    task.update(source_dirs=["tools"], fail_before_cmd=["{python}", "-B", str(probe)],
                pass_after_cmd=["{python}", "-B", str(probe)],
                patches=[str(patch(tmp_path, "bad.diff", entry, before, bad)),
                         str(patch(tmp_path, "good.diff", entry, before, good))])
    dag["tasks"] = [task]
    plan_file.write_text(json.dumps(dag), encoding="utf-8")
    code, run = call(capsys, "run", "--repo", root, "--dag", plan_file, "--db", db)
    assert code == 0 and run["telemetry"]["predictions_made"] == 2
    rejected, accepted = run["tasks"][0]["aegis_result"]["results"]
    assert rejected["baseline_failed"] and rejected["after_passed"] and not rejected["passed"]
    assert rejected["gate_failure"] == "ADVERSARIAL_AUDIT_REJECTED"
    assert rejected["review"]["findings"][0]["code"] == "UNBOUNDED_GROWTH"
    assert rejected["review"]["findings"][0]["line"] == 4
    assert accepted["source_changed_lines"] > 0 and accepted["review"]["passed"]
    assert accepted["probe_script_sha256"][str(probe.resolve())]
    assert accepted["predicted_next_state"] and git(root, "rev-parse", "HEAD") == base
    code, learned = call(capsys, "learn", "--repo", root, "--db", db, "--min-surprise", "100")
    assert code == 0 and learned["selected_traces"] == 0
    assert learned["telemetry"]["predictions_made"] == learned["telemetry"]["below_threshold_count"] == 2
    assert learned["experience_bank"]["recovery_added"] == 1
    hits = ExperienceBank(db).search("UNBOUNDED_GROWTH", min_similarity=0.01)
    assert hits[0]["source_kind"] == "verified_recovery"
    pairs = learned["telemetry"]["verified_recoveries"]
    assert hits[0]["trace_id"] == pairs[0]["failed_trace_id"]
    assert hits[0]["resolution"] == f"dual_probe_recovered:{pairs[0]['passed_trace_id']}"
    assert call(capsys, "learn", "--repo", root, "--db", db, "--min-surprise", "100")[1]["experience_bank"]["added"] == 0
    assert not (root / "artifacts").exists()


def test_learn_counts_legacy_no_prediction_and_can_report_no_model(tmp_path, monkeypatch, capsys):
    root = repository(tmp_path / "external", {"source.py": "PROBE = 0\n"})
    db = tmp_path / "ledger.sqlite"
    ledger = Ledger(db)
    ledger.create_ticket("legacy", "legacy", {}, "pass")
    lease = ledger.acquire_lease("legacy", "test")
    ledger.transition("legacy", lease, "queued", "running", state_s0={"status": "queued"},
                      action_a0={"op": "start_run"}, actual_state_s1={"status": "running"}, reward=0)
    ledger.transition("legacy", lease, "running", "succeeded", state_s0={"status": "running"},
                      action_a0={"op": "repair_attempt"}, actual_state_s1={"status": "succeeded", "passed": True},
                      reward=1)
    monkeypatch.setenv("AEGIS_WORLD_MODEL_DIR", str(tmp_path / "missing-override"))
    code, result = call(capsys, "learn", "--repo", root, "--db", db)
    assert code == 0 and result["status"] == "no_surprises"
    assert result["telemetry"]["no_prediction_count"] == 1 and not result["telemetry"]["model_loaded"]
    assert result["telemetry"]["below_threshold_count"] == 0
