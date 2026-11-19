"""Incremental, deduplicated preference distillation and release bundle auditing."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tempfile

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
_TRAIN_FILES = ("__init__.py", "common.py", "train_sft.py", "train_qlora_sft.py", "train_grpo.py",
                "requirements-train.txt")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(data: bytes): return hashlib.sha256(data).hexdigest()


class DistillPipeline:
    """Consume immutable Aegis evidence and build a portable local training release."""

    def __init__(self, repo_root: str | Path, *, model_id: str = MODEL_ID, tokenizer=None,
                 tokenizer_id: str | None = None, tokenizer_path: str | Path | None = None):
        self.root = Path(repo_root).resolve()
        if type(model_id) is not str or not model_id.strip(): raise ValueError("model_id is required")
        self.model_id = model_id
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as error:
                raise RuntimeError("official Qwen tokenizer requires the optional transformers dependency") from error
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path or model_id), trust_remote_code=False,
                                                      local_files_only=tokenizer_path is not None)
            tokenizer_id = tokenizer_id or getattr(tokenizer, "name_or_path", model_id)
        self.tokenizer = tokenizer
        self.tokenizer_id = tokenizer_id or getattr(tokenizer, "name_or_path", model_id)
        if not hasattr(tokenizer, "encode"):
            raise TypeError("tokenizer must expose encode(text, add_special_tokens=False)")

    def _tokens(self, value: str) -> int:
        encoded = self.tokenizer.encode(value, add_special_tokens=False)
        if not isinstance(encoded, (list, tuple)):
            raise TypeError("tokenizer.encode must return token ids")
        return len(encoded)

    def build_bundle(self, output_dir: str | Path, *, sft_paths=(), tournament_paths=(),
                     ledger_paths=(), experience_paths=(), max_tokens: int = 4096,
                     min_quality: float = .35, release_version: int = 14) -> dict:
        if type(release_version) is not int or release_version not in {14, 15}:
            raise ValueError("release_version must be a supported integer release (14 or 15)")
        if type(max_tokens) is not int or not 128 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be in [128, 4096] for the 8GB local training profile")
        if type(min_quality) not in (int, float) or not math.isfinite(min_quality) or not 0 <= min_quality <= 1:
            raise ValueError("min_quality must be finite and between zero and one")
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        data_dir = output / "datasets"; data_dir.mkdir(parents=True, exist_ok=True)
        sft_sources = [Path(path).resolve() for path in sft_paths]
        tournament_sources = [Path(path).resolve() for path in tournament_paths]
        ledger_sources = [Path(path).resolve() for path in ledger_paths]
        experience_sources = [Path(path).resolve() for path in experience_paths]
        sft_sources = sft_sources or [self.root / "datasets" / "mediator_sft_train.jsonl"]
        tournament_sources = tournament_sources or [self.root / "datasets" / "grpo_tournament_pairs.jsonl"]
        if not experience_sources and ledger_sources: experience_sources = ledger_sources[:]

        counters = Counter()
        sft_by_fingerprint = {}
        preference_by_fingerprint = {}
        token_lengths_sft, token_lengths_preference = [], []
        for source in sft_sources:
            if source.is_file():
                for line_number, row in _jsonl(source):
                    counters["sft_rows_seen"] += 1
                    normalized = self._normalize_sft(row, source)
                    if normalized is None:
                        counters["invalid_sft_rows"] += 1; continue
                    fingerprint = _digest(_canonical({"messages": normalized["messages"]}).encode("utf-8"))
                    if fingerprint in sft_by_fingerprint:
                        counters["duplicate_sft_rows"] += 1; continue
                    length = self._sft_length(normalized)
                    score = _sft_quality(normalized)
                    if length > max_tokens:
                        counters["overlength_sft_rows"] += 1; continue
                    if score < min_quality:
                        counters["low_quality_sft_rows"] += 1; continue
                    normalized.update({"token_count": length, "quality_score": round(score, 6),
                                       "fingerprint": fingerprint, "source": source.name})
                    sft_by_fingerprint[fingerprint] = normalized
                    token_lengths_sft.append(length)
        for source in tournament_sources:
            if source.is_file():
                rows = list(_jsonl(source))
                labeled_groups = defaultdict_list()
                for line_number, pair in rows:
                    if _is_labeled_rollout(pair):
                        prompt = _prompt_text(pair.get("prompt"))
                        group_id = pair.get("group_id")
                        if prompt and type(group_id) is str and group_id:
                            labeled_groups[(group_id, prompt)].append(pair)
                        counters["labeled_rollout_rows_seen"] += 1
                        continue
                    counters["tournament_pairs_seen"] += 1
                    normalized = self._normalize_pair(pair, source)
                    if normalized is None:
                        counters["invalid_tournament_pairs"] += 1; continue
                    fingerprint = _digest(_canonical({key: normalized[key] for key in
                        ("prompt", "chosen_patch", "rejected_patch")}).encode("utf-8"))
                    if fingerprint in preference_by_fingerprint:
                        counters["duplicate_tournament_pairs"] += 1; continue
                    length = self._pair_length(normalized)
                    score = _pair_quality(normalized)
                    if length > max_tokens:
                        counters["overlength_tournament_pairs"] += 1; continue
                    if score < min_quality:
                        counters["low_quality_tournament_pairs"] += 1; continue
                    normalized.update({"token_count": length, "quality_score": round(score, 6),
                                       "fingerprint": fingerprint, "source_file": source.name})
                    preference_by_fingerprint[fingerprint] = normalized
                    token_lengths_preference.append(length)
                for (group_id, _), candidates in sorted(labeled_groups.items()):
                    normalized = _preference_from_labeled_group(candidates, source.name)
                    if normalized is None:
                        counters["unpaired_labeled_rollout_groups"] += 1
                        continue
                    counters["tournament_pairs_seen"] += 1
                    counters["labeled_rollout_groups_paired"] += 1
                    fingerprint = _digest(_canonical({key: normalized[key] for key in
                        ("prompt", "chosen_patch", "rejected_patch")}).encode("utf-8"))
                    if fingerprint in preference_by_fingerprint:
                        counters["duplicate_tournament_pairs"] += 1; continue
                    length = self._pair_length(normalized)
                    score = _pair_quality(normalized)
                    if length > max_tokens:
                        counters["overlength_tournament_pairs"] += 1; continue
                    if score < min_quality:
                        counters["low_quality_tournament_pairs"] += 1; continue
                    normalized.update({"token_count": length, "quality_score": round(score, 6),
                                       "fingerprint": fingerprint, "source_file": source.name})
                    preference_by_fingerprint[fingerprint] = normalized
                    token_lengths_preference.append(length)
        for source in ledger_sources:
            if source.is_file():
                ledger_sft, ledger_pairs, counts = _ledger_examples(source)
                counters.update(counts)
                for row in ledger_sft:
                    fingerprint = _digest(_canonical(row).encode())
                    if fingerprint not in sft_by_fingerprint:
                        length = self._sft_length(row); score = _sft_quality(row)
                        if length <= max_tokens and score >= min_quality:
                            row.update({"token_count": length, "quality_score": round(score, 6),
                                        "fingerprint": fingerprint, "source": "sqlite_ledger"})
                            sft_by_fingerprint[fingerprint] = row; token_lengths_sft.append(length)
                for pair in ledger_pairs:
                    fingerprint = _digest(_canonical({key: pair[key] for key in
                        ("prompt", "chosen_patch", "rejected_patch")}).encode())
                    if fingerprint not in preference_by_fingerprint:
                        length = self._pair_length(pair); score = _pair_quality(pair)
                        if length <= max_tokens and score >= min_quality:
                            pair.update({"token_count": length, "quality_score": round(score, 6),
                                         "fingerprint": fingerprint, "source_file": "sqlite_ledger"})
                            preference_by_fingerprint[fingerprint] = pair; token_lengths_preference.append(length)
        for source in experience_sources:
            if source.is_file():
                for row in _experience_examples(source):
                    counters["experience_rows_seen"] += 1
                    fingerprint = _digest(_canonical(row).encode())
                    if fingerprint in sft_by_fingerprint:
                        counters["duplicate_experience_rows"] += 1; continue
                    length = self._sft_length(row); score = _sft_quality(row)
                    if length <= max_tokens and score >= min_quality:
                        row.update({"token_count": length, "quality_score": round(score, 6),
                                    "fingerprint": fingerprint, "source": "experience_bank"})
                        sft_by_fingerprint[fingerprint] = row; token_lengths_sft.append(length)
                    else:
                        counters["filtered_experience_rows"] += 1

        sft_rows = [sft_by_fingerprint[key] for key in sorted(sft_by_fingerprint)]
        preference_rows = [preference_by_fingerprint[key] for key in sorted(preference_by_fingerprint)]
        # Every preference record contains exactly one accepted and one rejected
        # completion.  Export all pairs in deterministic order, retaining an exact
        # 1:1 chosen/rejected balance even when the available sources are uneven.
        sft_path = data_dir / "sft_train.jsonl"
        grpo_path = data_dir / "grpo_preference_pairs.jsonl"
        _atomic_write(sft_path, _jsonl_bytes(sft_rows))
        _atomic_write(grpo_path, _jsonl_bytes(preference_rows))

        training_dir = output / "training"; training_dir.mkdir(parents=True, exist_ok=True)
        for name in _TRAIN_FILES:
            source = self.root / "training" / name
            if source.is_file(): shutil.copy2(source, training_dir / name)
        scripts_dir = output / "scripts"; scripts_dir.mkdir(parents=True, exist_ok=True)
        bootstrap = self.root / "scripts" / "bootstrap_local.ps1"
        if bootstrap.is_file(): shutil.copy2(bootstrap, scripts_dir / bootstrap.name)
        runbook = (f"# Aegis v{release_version} local training bundle\n\n"
                   "Windows 11 + RTX 5060 8GB: run `scripts\\bootstrap_local.ps1` from the repository root.\n"
                   f"SFT dataset: `datasets\\sft_train.jsonl`; GRPO preferences: `datasets\\grpo_preference_pairs.jsonl`.\n"
                   f"Base model and tokenizer: `{self.model_id}`. The release uses the tokenizer recorded in the manifest.\n")
        _atomic_write(output / "RUN_LOCAL_WINDOWS.md", runbook.encode("utf-8"))
        file_entries = []
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name not in {"manifest.json", f"release_v{release_version}_bundle_manifest.json"}:
                file_entries.append({"path": path.relative_to(output).as_posix(),
                                     "bytes": path.stat().st_size, "sha256": _digest(path.read_bytes())})
        tokenizer_info = {"model_id": self.model_id, "tokenizer_id": self.tokenizer_id,
                          "official_tokenizer": self.tokenizer_id == MODEL_ID,
                          "zero_truncation": True, "max_tokens": max_tokens}
        def source_label(item: Path) -> str:
            try:
                return item.relative_to(self.root).as_posix()
            except ValueError:
                return item.name

        manifest = {"schema_version": f"aegis.training.release.v{release_version}", "base_model": self.model_id,
                    "tokenizer": tokenizer_info, "created_from": {"ledger_files": [source_label(item) for item in ledger_sources],
                       "experience_files": [source_label(item) for item in experience_sources],
                       "tournament_files": [source_label(item) for item in tournament_sources],
                       "sft_files": [source_label(item) for item in sft_sources]},
                    "quality_policy": {"min_quality": float(min_quality), "deduplicate": True,
                                       "preference_balance": "one chosen positive / one rejected negative per pair",
                                       "truncation_policy": "reject overlength rows; never truncate"},
                    "counts": {**dict(sorted(counters.items())), "sft_examples": len(sft_rows),
                               "preference_pairs": len(preference_rows), "chosen_positive": len(preference_rows),
                               "rejected_negative": len(preference_rows), "truncated_rows": 0},
                    "token_length_distribution": {"sft": _distribution(token_lengths_sft),
                                                  "preference": _distribution(token_lengths_preference)},
                    "files": file_entries,
                    "windows_training": {"bootstrap": "scripts/bootstrap_local.ps1",
                                         "python": "3.11 or 3.12", "gpu": "RTX 5060 8GB",
                                         "precision": "4-bit NF4 QLoRA", "device_map": "cuda:0"}}
        manifest_path = output / "manifest.json"
        _atomic_write(manifest_path, (_canonical(manifest) + "\n").encode())
        sibling_manifest = output.parent / f"release_v{release_version}_bundle_manifest.json"
        _atomic_write(sibling_manifest, (_canonical(manifest) + "\n").encode())
        verification = self.verify_bundle(manifest_path)
        return {"status": "verified" if verification["valid"] else "invalid",
                "manifest": str(sibling_manifest), "bundle_root": str(output),
                "tokenizer": tokenizer_info, "counts": manifest["counts"],
                "token_length_distribution": manifest["token_length_distribution"],
                "file_count": len(file_entries), "bundle_sha256": _digest(manifest_path.read_bytes()),
                "verification": verification}

    @staticmethod
    def verify_bundle(manifest_path: str | Path) -> dict:
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        schema_version = manifest.get("schema_version", "")
        if schema_version not in {"aegis.training.release.v14", "aegis.training.release.v15"}:
            raise ValueError("unsupported release bundle manifest")
        release_version = schema_version.rsplit("v", 1)[1]
        root = path.parent if path.name == "manifest.json" else path.parent / f"release_v{release_version}"
        verified = []
        for entry in manifest.get("files", ()):
            relative = PurePosixPath(entry.get("path", ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("release manifest contains an unsafe path")
            target = root.joinpath(*relative.parts)
            if not target.is_file(): raise FileNotFoundError(target)
            if target.stat().st_size != entry["bytes"] or _digest(target.read_bytes()) != entry["sha256"]:
                raise ValueError(f"release file digest mismatch: {relative}")
            verified.append(relative.as_posix())
        data_entries = {PurePosixPath(item["path"]).as_posix(): item for item in manifest["files"]}
        expected_rows = {"datasets/sft_train.jsonl": manifest["counts"]["sft_examples"],
                         "datasets/grpo_preference_pairs.jsonl": manifest["counts"]["preference_pairs"]}
        for filename, count in expected_rows.items():
            if filename not in data_entries: raise ValueError(f"missing dataset artifact {filename}")
            actual_count = sum(1 for _ in _jsonl(root / filename))
            if actual_count != count: raise ValueError(f"row-count mismatch: {filename}")
        if manifest["counts"].get("truncated_rows") != 0 or not manifest["tokenizer"].get("zero_truncation"):
            raise ValueError("release must have a zero-truncation token audit")
        return {"valid": True, "verified_files": len(verified), "verified_paths": verified,
                "sft_rows": expected_rows["datasets/sft_train.jsonl"],
                "preference_pairs": expected_rows["datasets/grpo_preference_pairs.jsonl"]}

    def _normalize_sft(self, row, source):
        messages = row.get("messages") if type(row) is dict else None
        if not isinstance(messages, list) or len(messages) < 2:
            return None
        if any(type(item) is not dict or item.get("role") not in {"system", "user", "assistant"}
               or type(item.get("content")) is not str or not item["content"].strip() for item in messages):
            return None
        if messages[-1]["role"] != "assistant": return None
        try: payload = json.loads(messages[-1]["content"])
        except (ValueError, TypeError): return None
        if type(payload) is not dict: return None
        return {"messages": [{"role": item["role"], "content": item["content"]} for item in messages],
                "source_name": source.name}

    def _normalize_pair(self, row, source):
        if type(row) is not dict: return None
        prompt = row.get("prompt")
        chosen, rejected = row.get("chosen_patch"), row.get("rejected_patch")
        if type(prompt) is not str or not prompt.strip() or len(prompt) > 20000:
            return None
        if type(chosen) is not str or not chosen.strip() or type(rejected) is not str or not rejected.strip():
            return None
        chosen_score, rejected_score = row.get("chosen_score"), row.get("rejected_score")
        if type(chosen_score) not in (int, float) or type(rejected_score) not in (int, float): return None
        if not math.isfinite(chosen_score) or not math.isfinite(rejected_score) or chosen_score <= rejected_score:
            return None
        chosen_hash, rejected_hash = _digest(chosen.encode()), _digest(rejected.encode())
        if row.get("chosen_sha256") not in (None, chosen_hash): return None
        if row.get("rejected_sha256") not in (None, rejected_hash): return None
        return {"prompt": prompt.strip(), "chosen_patch": chosen, "rejected_patch": rejected,
                "chosen_score": float(chosen_score), "rejected_score": float(rejected_score),
                "chosen_sha256": chosen_hash, "rejected_sha256": rejected_hash,
                "source": row.get("source", "candidate_tournament")}

    def _sft_length(self, row):
        if hasattr(self.tokenizer, "apply_chat_template"):
            rendered = self.tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                                          add_generation_prompt=False)
            return len(rendered)
        return sum(self._tokens(item["content"]) for item in row["messages"])

    def _pair_length(self, row):
        return self._tokens(row["prompt"]) + max(self._tokens(row["chosen_patch"]), self._tokens(row["rejected_patch"]))


def _jsonl(path):
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try: value = json.loads(line)
            except json.JSONDecodeError as error: raise ValueError(f"invalid JSONL {path}:{line_number}: {error.msg}") from error
            yield line_number, value


def _is_labeled_rollout(row) -> bool:
    return (type(row) is dict and type(row.get("positive")) is bool
            and type(row.get("completion")) is str and "group_id" in row)


def _prompt_text(prompt):
    if type(prompt) is str:
        return prompt.strip() if prompt.strip() else None
    if type(prompt) is list and prompt:
        parts = []
        for message in prompt:
            if type(message) is not dict or type(message.get("role")) is not str \
                    or type(message.get("content")) is not str:
                return None
            parts.append(f"{message['role']}: {message['content']}")
        return "\n".join(parts).strip()
    return None


def _preference_from_labeled_group(rows, source_name):
    """Pair the strongest positive and weakest negative verified completion."""
    positives, negatives = [], []
    for row in rows:
        try:
            content = json.loads(row["completion"])
        except (ValueError, TypeError):
            continue
        patch = content.get("patch") if type(content) is dict else row["completion"]
        digest = row.get("patch_sha256")
        reward = row.get("reward")
        prompt = _prompt_text(row.get("prompt"))
        if type(patch) is not str or not patch.strip() or type(digest) is not str:
            continue
        if not prompt:
            continue
        if _digest(patch.encode("utf-8")) != digest:
            continue
        if type(reward) not in (int, float) or not math.isfinite(reward):
            continue
        target = positives if row["positive"] else negatives
        target.append((float(reward), digest, patch, prompt))
    if not positives or not negatives:
        return None
    chosen = min(positives, key=lambda item: (-item[0], item[1]))
    rejected = min(negatives, key=lambda item: (item[0], item[1]))
    if chosen[0] <= rejected[0] or chosen[3] != rejected[3]:
        return None
    return {"prompt": chosen[3], "chosen_patch": chosen[2], "rejected_patch": rejected[2],
            "chosen_score": chosen[0], "rejected_score": rejected[0],
            "chosen_sha256": chosen[1], "rejected_sha256": rejected[1],
            "source": f"verified_labeled_rollout:{source_name}"}


def _jsonl_bytes(rows):
    return ("".join(_canonical(row) + "\n" for row in rows)).encode("utf-8")


def _sft_quality(row):
    assistant = row["messages"][-1]["content"]
    user_text = " ".join(item["content"] for item in row["messages"] if item["role"] == "user")
    try: payload = json.loads(assistant)
    except (ValueError, TypeError): return 0.0
    score = .35 + min(.25, len(user_text.strip()) / 400) + min(.25, len(assistant.strip()) / 2000)
    if isinstance(payload, dict) and payload: score += .15
    return min(1.0, score)


def _pair_quality(row):
    margin = max(0.0, min(1.0, (row["chosen_score"] - row["rejected_score"])))
    patch_score = min(1.0, len(row["chosen_patch"]) / 1000)
    prompt_score = min(1.0, len(row["prompt"]) / 160)
    return min(1.0, .25 + .35 * margin + .2 * patch_score + .2 * prompt_score)


def _distribution(lengths):
    if not lengths: return {"count": 0, "min": 0, "p50": 0, "p90": 0, "max": 0}
    ordered = sorted(lengths)
    def percentile(q): return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]
    return {"count": len(ordered), "min": ordered[0], "p50": percentile(.5),
            "p90": percentile(.9), "max": ordered[-1]}


def _ledger_examples(path):
    uri = f"file:{Path(path).as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=30) as db:
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "causal_traces" not in tables: return [], [], Counter()
        traces = []
        columns = {row[1] for row in db.execute("PRAGMA table_info(causal_traces)")}
        ticket_status = "LEFT JOIN tickets USING(ticket_id)" if "tickets" in tables else ""
        status = "tickets.status AS ticket_status" if "tickets" in tables else "NULL AS ticket_status"
        for row in db.execute(f"SELECT causal_traces.*, {status} FROM causal_traces {ticket_status} ORDER BY trace_id"):
            item = dict(row)
            for key in ("state_s0", "action_a0", "actual_state_s1", "predicted_next_state"):
                item[key] = json.loads(item[key]) if item.get(key) else {}
            traces.append(item)
    attempts = [row for row in traces if (row["action_a0"] or {}).get("op") == "repair_attempt"]
    sft, preferences = [], []
    grouped = defaultdict_list()
    for row in attempts: grouped[row["ticket_id"]].append(row)
    counts = Counter({"ledger_traces_seen": len(traces), "ledger_repair_attempts": len(attempts)})
    for ticket, rows in grouped.items():
        successes = [row for row in rows if (row["actual_state_s1"] or {}).get("passed")
                     and (row["actual_state_s1"] or {}).get("after_passed")]
        failures = [row for row in rows if not (row["actual_state_s1"] or {}).get("passed")
                    and (row["actual_state_s1"] or {}).get("gate_failure")]
        for failed in failures:
            action = failed["action_a0"]
            diagnosis = (failed["actual_state_s1"] or {}).get("diagnosis") or {}
            prompt = action.get("contract") or f"Repair ticket {ticket}; diagnosis {diagnosis.get('failure_code', 'gate rejection')}"
            for passed in successes:
                good, bad = passed["action_a0"], action
                if good.get("patch_sha256") == bad.get("patch_sha256") or good.get("contract") != bad.get("contract"):
                    continue
                preferences.append({"prompt": prompt, "chosen_patch": good.get("patch", ""),
                    "rejected_patch": bad.get("patch", ""),
                    "chosen_score": float(good.get("tournament_score", 1.0)),
                    "rejected_score": float(bad.get("tournament_score", 0.0)),
                    "chosen_sha256": good.get("patch_sha256"), "rejected_sha256": bad.get("patch_sha256"),
                    "source": "ledger_verified_recovery"})
        for passed in successes:
            action = passed["action_a0"]
            patch = action.get("patch")
            contract = action.get("contract")
            if type(patch) is not str or not patch.strip() or type(contract) is not str or not contract.strip():
                continue
            expected_hash = action.get("patch_sha256")
            if expected_hash and expected_hash != _digest(patch.encode("utf-8")):
                continue
            sft.append({"messages": [
                {"role": "system", "content": "Return a JSON object containing the verified repair patch."},
                {"role": "user", "content": contract.strip()},
                {"role": "assistant", "content": _canonical({"patch": patch})},
            ]})
            counts["ledger_verified_repair_examples"] += 1
    return sft, preferences, counts


def defaultdict_list():
    from collections import defaultdict
    return defaultdict(list)


def _experience_examples(path):
    uri = f"file:{Path(path).as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=30) as db:
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "experience_records" not in tables: return []
        rows = db.execute("SELECT pattern,outcome,cause,resolution,human_rule FROM experience_records ORDER BY experience_id").fetchall()
    result = []
    for item in rows:
        content = json.dumps({"outcome": item["outcome"], "cause": item["cause"],
                              "resolution": item["resolution"], "rule": json.loads(item["human_rule"])},
                             ensure_ascii=False, sort_keys=True)
        result.append({"messages": [{"role": "user", "content": "Apply this verified engineering precedent: " + item["pattern"]},
                                    {"role": "assistant", "content": content}]})
    return result


def _atomic_write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
