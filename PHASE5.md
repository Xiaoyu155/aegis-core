# Phase 5 acceptance

`runner.retrieve` combines a repository-wide AST index with deterministic weighted
identifier/docstring retrieval and a bilingual concept lexicon. It cites real file
and line ranges, excludes tests from business candidates, and stops on unresolved
invariants or close candidate scores. This is static retrieval, not unrestricted
natural-language understanding. Explicit approved targets remain authoritative.

`runner.diagnose` captures assertion text, resolved traceback frames and bounded
AST slices before the failing worktree is destroyed. The next model dispatch
receives these records in `prior_attempts`. Exact rejected patch hashes still
trip the existing replay breaker; tests and probes remain frozen.

The eight-case campaign executes real pytest subprocesses through the existing
CLI, SQLite WAL ledger and isolated worktrees. The business projects and repair
patches are authored offline fixtures. The self-healing provider is a deterministic
mock that requires assertion diagnosis before its second response. This verifies
feedback plumbing and gate behavior, not live LLM reasoning or production trading.
The render test verifies bounded retained state and draw-buffer identity; it does
not assert zero Python allocations or actual Canvas/GPU visual correctness.

From the activated core environment:

```
python scripts/run_real_world_campaign.py
python -m pytest -v
```

The default campaign directory must be new. For repeat runs use
`--output <new-directory>`; custom-output runs leave the checked-in datasets alone.
The default run appends five accepted SFT repairs and one same-ticket contrastive
GRPO group, preserving the 1,050 synthetic variants. All 47 physical transitions,
including negative attempts and non-repair states, remain in
`datasets/campaign/campaign_traces.jsonl`. Each real row has explicit provenance;
SFT excludes rejected patches. Same-input failure/success groups are selected by
gate outcome, not by small variations in elapsed-time reward. Distillation replaces
previous campaign-tagged rows on rerun, so evidence cannot inflate by appending
identical executions. Campaign samples are a small integration corpus, not a
statistically independent production or held-out generalization benchmark.

PR #5 is stacked on the unmerged PR #4 branch. Existing Windows bootstrap,
training dependencies, prior datasets, and the 136-test baseline remain present.

The SFT preflight accepts both legacy two-message synthetic rows and three-message
accepted campaign rows, adding the system role in memory for legacy rows. Check
with `python -m training.train_sft --dataset datasets/sft_mediator_train.jsonl --validate-only`.
The grouped causal rollout file is a reference/distillation corpus. The existing
online GRPO trainer still uses its separate executable simulator candidate-pair
format; it does not replay campaign groups against a real model automatically.
