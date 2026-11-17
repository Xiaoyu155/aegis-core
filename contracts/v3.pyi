"""Frozen combined stage 3/4 APIs; preserves all V1/V2 behavior."""
from pathlib import Path

def build_datasets(output_dir: Path, *, count: int = 520, seed: int = 42, verify_templates: bool = True) -> dict: ...
def main(argv: list[str] | None = None) -> int: ...
# CLI wish: repo/task/constraints/proposal -> compilation + immutable repository
# baseline reference. run: validate manifest -> lease -> isolated repair attempts
# -> route/breaker -> terminal WAL trace; never auto-merges user business code.
# JSON manifests bind repo_root, base_commit, compilation, probes, source_dirs,
# local patch candidates OR explicit model scheduling. Test command strings use
# shell-free argument arrays. All paths and changed files remain scope checked.
# Optional training imports are deferred; --help and dataset validation are stdlib.
