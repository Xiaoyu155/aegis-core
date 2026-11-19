"""Physical contract probe edge cases discovered during business-repo gating."""
import subprocess

from aegis_core.probe_synth import compile_structured_probe


def _repo(path, source):
    path.mkdir()
    target = path / "app.py"
    target.write_text(source, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Probe Test"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "probe@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "add", "app.py"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=path, check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True,
                          capture_output=True, text=True).stdout.strip()


def test_contract_probe_wrong_exception_type_is_a_true_red_assertion(tmp_path):
    root = tmp_path / "wrong_exception"
    baseline = _repo(root, "def max_stake(payload):\n    return payload['notional'] / payload['leverage']\n")
    result = compile_structured_probe({
        "kind": "contract_schema",
        "path": "app.py",
        "symbol": "max_stake",
        "seed": 15,
        "input_schema": {"type": "object", "required": ["notional", "leverage"],
                         "properties": {"notional": {"type": "number"}, "leverage": {"type": "number"}}},
        "return_schema": {"type": "object", "required": ["max_stake"],
                          "properties": {"max_stake": {"type": "number"}}},
        "valid_args": [[{"notional": 100, "leverage": 2}]],
        "invalid_args": [[{"notional": 100, "leverage": 0}]],
        "invalid_exception": "ValueError",
    }, root, baseline, timeout=10)

    assert result["red"]["failure_code"] == "ASSERTION_FAILED"
    assert "AEGIS_ACCEPTANCE" in result["red"]["stdout"]
    assert "ZeroDivisionError" in result["red"]["stdout"]
