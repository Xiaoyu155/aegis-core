"""Reproducible repository-scale gate latency and memory profile."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import tracemalloc

from .security_gate import scan_files


def run_performance_profile(*, output_path: str | Path | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="aegis-v20-perf-") as temporary:
        root = Path(temporary)
        tracemalloc.start()
        all_files = []
        changed_files = []
        diff_chunks = []
        for index in range(500):
            relative = f"src/module_{index:03d}.py"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = [f"def calculate_{index}(value):\n", "    total = 0\n"]
            lines.extend(f"    total += value + {line}\n" for line in range(8))
            lines.append("    return total\n")
            path.write_text("".join(lines), encoding="utf-8")
            all_files.append(relative)
            if index < 50:
                changed_files.append(relative)
                for line_number in range(100):
                    diff_chunks.append(f"+    total += value + {line_number}  # stress line\n")
        synthetic_diff = ("diff --git a/src/module_000.py b/src/module_000.py\n"
                          "--- a/src/module_000.py\n+++ b/src/module_000.py\n@@ -1,0 +1,5000 @@\n"
                          + "".join(diff_chunks)).encode("utf-8")
        started = time.perf_counter()
        report = scan_files(root, changed_files, diff=synthetic_diff)
        elapsed_ms = (time.perf_counter() - started) * 1000
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result = {"schema_version": "aegis.performance.v1", "simulated_repository_files": len(all_files),
                  "changed_files": len(changed_files), "diff_lines": len(diff_chunks),
                  "files_scanned": report["files_scanned"], "p95_ms": report["p95_ms"],
                  "total_scan_ms": report["total_ms"], "wall_ms": elapsed_ms,
                  "peak_memory_bytes": peak, "peak_memory_mib": peak / (1024 * 1024),
                  "findings": len(report["findings"]), "degraded_files": len(report["degraded_files"]),
                  "thresholds": {"p95_ms": 80, "total_50_file_ms": 3000, "peak_memory_bytes": 256 * 1024**2},
                  "passed": (report["p95_ms"] < 80 and elapsed_ms < 3000 and peak < 256 * 1024**2)}
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result
