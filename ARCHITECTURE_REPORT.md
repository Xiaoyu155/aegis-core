# Aegis Core Architecture Report — PR #9

Updated 2026-11-18. This report summarizes implemented gates and measured runs; it does not claim production deployment or unrestricted model autonomy.

## Delivery lineage

| Delivery | Implemented capability |
|---|---|
| PR #1–#2 | SQLite WAL ticket/trace ledger, strict JSON contracts, Python AST slices, isolated Git worktrees, and lifecycle gates. |
| PR #3–#5 | Deterministic wish compilation and clarification, dual probes, source/test protection, replay detection, model routing, bounded reward, offline business arenas, and the eight-work-order physical campaign. PR #3 was not a separate release branch. |
| PR #6 | CPU-trained causal state, reward, and risk heads; predictions and observed outcomes are recorded in the ledger. |
| PR #7 | Reward calibration, ticket/template-group stratified splits, `plan`/`run --dag`, `learn`, mixed-language attach profiles, and JavaScript/TypeScript symbol extraction. |
| PR #8 | Explicit new-source DAG planning, process-group timeout kill and reaping, cascade skip codes, append-only versioned `ExperienceBank`, and validated private-repository worktree runs. |
| PR #9 | Repeated external cat-cafe work orders, bounded projection reuse, save integrity recovery, v3 experience replay, and mediator retrieval from fresh traces. |

## Runtime path

`wish` freezes a repository-relative contract and probes against a Git baseline. `run` acquires a ledger lease, optionally predicts the proposed outcome, then runs fail-before and pass-after probes in a disposable worktree. The source-scope and protected-test checks run before a result can pass. `run --dag` commits each passing task only inside its disposable staging worktree, schedules independent tasks by dependency layer, and marks descendants of failed tasks `SKIPPED_UPSTREAM_FAILED`. `attach` writes a language, entry-point, and test-runner profile under `.aegis/`. `learn` selects high-surprise real traces, replays them with historical anchors, and appends trace summaries to the versioned experience bank.

Timeouts now start a new process group/session. A timeout kills the full group, waits for the direct child, reaps adopted Linux descendants, and records `TIMEOUT_KILLED` with a negative reward. Windows uses `taskkill /T /F`. A Git worktree is source isolation, not an OS security boundary; gate commands still inherit host permissions.

## Measured evidence

| Run | Result |
|---|---|
| Existing world-model holdout | 1,069 examples; train/validation/test 704/182/183; no ticket/template leakage. State accuracy 48.63% → 97.27%; reward MAE 0.7432 → 0.3049; learned-risk precision 0.20 → 0.667 and recall 0.50 → 1.00. Deterministic pre-block precision/recall was 1.00/1.00 on support 2. |
| Private `palm-cat-cafe` attach | 1,327 files: 283 Python, 238 JavaScript, 0 TypeScript. Detected `node --test`, `npm test`, and `pytest`; profile at `.aegis/repo_profile.json`. |
| Cross-repository DAG | New `interaction.js` primitive → `core.js` integration → existing `npm test`; 3/3 DAG tasks passed. Headless assertions covered zero-distance cat separation, deterministic symmetry, render-priority restoration, and core wiring. Target main worktree remained clean. |
| Four-node chaos drill | Diagnosis-driven repair passed on attempt 2; test-edit patch was `HIGH_RISK_PRE_BLOCK`; dependent command was `SKIPPED_UPSTREAM_FAILED`; infinite-loop probe was `TIMEOUT_KILLED` (reward −1.4575). Spawned child was reaped; no residual worktrees remained. |
| Online replay | 12 high-surprise traces; replay loss 2.053460 → 2.041790. Anchor validation loss 0.610334 → 0.613454, within the 0.02 guard. Model version `bde814c83bd926cf67540dcf64774da0f24bed5a15f15abff52f440c47114692`; experience bank version 2 contains 12 distilled records. |
| PR #8 tests and CI | 275 local tests passed (234 existing plus 41 PR #8 cases). See `STATUS.json` for the final eight Linux/Windows × Python 3.11/3.12 Actions checks. |

## PR #9 — external cat-cafe iteration and ExperienceBank v3

The external `main` ref remained at `22000d36ade5ee0166c96237834fd61179180568` with a clean original worktree. The gated outputs were preserved on the local branch `feat/pr9-catcafe-pr9` (`f52e469013bd710cd7041046a2d8a2a138a9f767`) in a separate clean worktree. Work order A passed 3/3 DAG tasks; work order B passed 4/4, including both headless probes and the repository's `npm test`. The final branch contains both tickets in order.

A adds a transaction-style interaction state module for feed, pet, and play. It clamps affection and needs, applies independent cooldowns, and wires those events into the existing game loop and web bundle. B adds a bounded TTL/LRU cache for recurring 2.5D projection pairs and a versioned save envelope with structural/range validation, checksum checks, legacy migration, and safe default recovery. The checksum detects accidental or casual edits; it is not cryptographic authentication against a client that can recompute it. The first B attempt exposed an off-by-one assumption in the authored TTL probe; the diagnosis showed the cache behavior was correct, and the probe was corrected to advance the required third frame before rerunning the full DAG.

Both external DAGs ran with the shipped causal model loaded via `AEGIS_WORLD_MODEL_DIR`. Across five new repair traces, the model produced composite errors from 1.3750 to 3.8549; the actual code gates passed, so these are recorded calibration surprises rather than model successes. `learn --min-surprise 0.25` selected 17 traces, including five new external traces. Replay loss moved 2.711768 → 2.697695; anchor validation loss moved 0.608742 → 0.611992 (increase 0.003251 against a 0.02 tolerance). The model version advanced from `541eb833…` to `5c64ee29…`. ExperienceBank now has 17 deduplicated records at version 3, hash `4c2491992e1e70b4e7356ffdb56d0b19bf1fb28b0b8e15a980b774762d3da383`. Mediator retrieval returned five matching v3 records (similarity 0.333–0.429) and did not fill safety constraints automatically.

The full Python suite passed 294/294. Nineteen PR #9 cases cover versioned memory search, mediator hits, report/anchor synchronization, invalid DAG shapes, and two sequential real worktree-gated work orders. The external repository's original `main` worktree remains clean. Eight GitHub Actions checks are pending for the PR branch at report time; `STATUS.json` is updated with the final run links after they finish.

## Model and data boundaries

The portable inference path uses the shipped NumPy-trained dense network and standard-library forward pass. Online learning requires NumPy. The local container did not train or validate on a GPU. The model can be explicitly selected for an attached repository with `AEGIS_WORLD_MODEL_DIR`; otherwise Aegis loads weights from that repository's own `artifacts/causal_world_model` directory. Deterministic policy gates remain authoritative over learned risk estimates.

The cross-repository and chaos runs use real Git repositories, real patches, and real child processes. They validate in disposable worktrees and do not merge changes into `palm-cat-cafe`. The headless JavaScript gate validates the new interaction primitives and their wiring; it does not claim a browser/WebGL visual regression run. The eight-case campaign and synthetic datasets remain offline fixtures, not production trading data.

## Windows 11 + RTX 5060 handoff

The existing `training/` tree and `scripts/bootstrap_local.ps1` are preserved. On the Windows host, fetch/checkout the merged PR #9 branch, then run PowerShell from the repository root:

```powershell
& .\scripts\bootstrap_local.ps1
& .\.venv\Scripts\python.exe -m training.train_sft --validate-only
& .\.venv\Scripts\python.exe -m training.train_grpo --validate-only
& .\scripts\bootstrap_local.ps1 -InstallTraining -Train sft -MaxSteps 1
```

The bootstrap creates/reuses a Python 3.11+ virtual environment, installs the editable core plus tests, and runs the suite. `-InstallTraining` separately installs a CUDA 12.8 PyTorch wheel and the pinned QLoRA/GRPO stack, then requires `torch.cuda.is_available()` before training. `-Train grpo` selects physical-reward GRPO. The default SFT model is Qwen2.5-Coder-3B-Instruct; the script does not claim the model fits 8 GB without the host-side CUDA preflight. Model downloads and actual RTX 5060 throughput must be measured on that machine.
