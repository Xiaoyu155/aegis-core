import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from luna.cli import main
from luna.initialize import _key_location, initialize, provision_keys


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".git").mkdir()
    return root


@pytest.mark.parametrize("name", ["keys", "signed keys", "签名密钥", ".local-keys", "Mixed_Case-3", "other/nested"])
def test_external_path_is_resolved(repo, name):
    path = repo.parent / name
    assert _key_location(repo, path) == path.resolve()


@pytest.mark.parametrize("name", ["", "keys", ".hidden", "nested/deep", "../project/other", "secrets/.."])
def test_rejects_in_repository_paths(repo, name):
    with pytest.raises(ValueError, match="Git"):
        _key_location(repo, repo / name)


@pytest.mark.parametrize("marker", ["directory", "worktree", "bare"])
@pytest.mark.parametrize("subdir", ["keys", "nested/keys"])
def test_rejects_other_git_worktrees_without_writing(repo, marker, subdir):
    other = repo.parent / "another-checkout"
    other.mkdir()
    if marker == "directory":
        (other / ".git").mkdir()
        (other / ".git" / "HEAD").touch()
    elif marker == "worktree":
        (other / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
    else:
        (other / "HEAD").touch()
        (other / "objects").mkdir()
    with pytest.raises(ValueError, match="Git"):
        provision_keys(repo, other / subdir)
    assert not (other / subdir).exists()


@pytest.mark.parametrize("new_name", ["keys", "签名 密钥", "nested/two levels"])
def test_key_generation_and_idempotency(repo, new_name):
    path = repo.parent / new_name
    first = provision_keys(repo, path)
    key = path / "ed25519-private.pem"
    original = key.read_bytes()
    assert original.startswith(b"-----BEGIN PRIVATE KEY-----")
    assert first["fingerprint"] == hashlib.sha256((path / "ed25519-public.pem").read_bytes()).hexdigest()
    second = provision_keys(repo, path)
    assert first["created"] and not second["created"]
    assert second["fingerprint"] == first["fingerprint"]
    assert key.read_bytes() == original
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
        assert stat.S_IMODE(key.stat().st_mode) == 0o600


@pytest.mark.parametrize("variable", ["AEGIS_SIGNING_KEY_PATH", "AEGIS_VERIFYING_KEY_PATH"])
def test_rejects_key_file_overrides(repo, monkeypatch, variable):
    monkeypatch.setenv(variable, str(repo / "material.pem"))
    with pytest.raises(ValueError, match="--keys-dir"):
        provision_keys(repo, repo.parent / "keys")
    assert not (repo.parent / "keys").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode checks")
@pytest.mark.parametrize("mode", [0o755, 0o710, 0o704, 0o770])
def test_unsafe_directory_rejected_before_writing(repo, mode):
    path = repo.parent / "keys"
    path.mkdir()
    path.chmod(mode)
    with pytest.raises(ValueError, match="0700"):
        provision_keys(repo, path)
    assert not list(path.iterdir())


@pytest.mark.parametrize("artifact", ["ed25519-private.pem", "ed25519-public.pem", "key-metadata.json"])
def test_partial_keyset_kept_for_manual_recovery(repo, artifact):
    path = repo.parent / "keys"
    provision_keys(repo, path)
    (path / artifact).unlink()
    remaining = {child.name: child.read_bytes() for child in path.iterdir()}
    with pytest.raises(ValueError, match="密钥材料缺失"):
        provision_keys(repo, path)
    assert {child.name: child.read_bytes() for child in path.iterdir()} == remaining


@pytest.mark.parametrize("failure", ["environment", "ed25519_key", "git_permissions", "sandbox_isolation",
                                     "ci", "model_gateway", None])
def test_cli_init_writes_template_and_reports_preflight(repo, failure, monkeypatch, capsys):
    capture = []
    def check(root, *, models, keys_dir):
        capture.append((root, models, keys_dir))
        return {"status": "BLOCKED" if failure else "READY", "blockers": [failure] if failure else []}
    monkeypatch.setattr("luna.initialize.combat_check", check)
    path = repo.parent / "keys"
    code = main(["init", "--repo", str(repo), "--keys-dir", str(path), "--models", "custom", "--json"])
    result = json.loads(capsys.readouterr().out)
    assert code == (2 if failure else 0)
    assert result["keys"]["created"]
    assert list(result["remediation"]) == ([failure] if failure else [])
    assert capture == [(repo.resolve(), "custom", str(path.resolve()))]
    template = Path(result["template"])
    assert "LUNA_REASONING_API_KEY=" in template.read_text(encoding="utf-8")
    assert "PRIVATE KEY" not in template.read_text(encoding="utf-8")
    assert not (repo / ".env.luna").exists()


def test_existing_template_edits_preserved(repo, monkeypatch):
    monkeypatch.setattr("luna.initialize.combat_check", lambda *a, **k: {"status": "READY", "blockers": []})
    template = repo / ".env.luna.example"
    template.write_text("# user edits", encoding="utf-8")
    initialize(repo, keys_dir=repo.parent / "keys")
    assert template.read_text(encoding="utf-8") == "# user edits"
