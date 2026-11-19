# Aegis Core Architecture Report — PR #10

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
| PR #10 | Parallel interface contracts, complexity routing and independent review, requirement expansion, and CPU training preflight. |

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

The full Python suite passed 294/294. Nineteen PR #9 cases cover versioned memory search, mediator hits, report/anchor synchronization, invalid DAG shapes, and two sequential real worktree-gated work orders. The external repository's original `main` worktree remains clean. All eight GitHub Actions checks passed on the PR commit (Linux/Windows × Python 3.11/3.12 for both push and pull_request). Push run: https://github.com/Xiaoyu155/aegis-core/actions/runs/44240248287; pull_request run: https://github.com/Xiaoyu155/aegis-core/actions/runs/44240327631. `STATUS.json` records the matrix results.

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

## PR #10 — Autonomous Contractor

PR #10 is stacked on `feat/world-model-v10` at `c03e76d9489c6f5ee8f48aca71598c928aba3131`.

### Fan-out and review gates

`aegis plan` records an interface-lock preview for each parallel repair layer. `run --dag` freezes a new lock against that layer's staged Git commit before scheduling the repair worktrees. The lock tracks statically visible cross-module Python signatures, type annotations, shared dictionary keys, TypedDict and dataclass fields, plus indexed JavaScript and TypeScript signatures and state declarations. Each worker receives the same lock digest in its frozen manifest. The gate reports the path, symbol, previous declaration, and observed declaration for drift. Overlapping source targets require a dependency before execution.

The complexity router combines the transitive production caller radius, import and call coupling, and the loaded causal model's risk probability. L1, L2, and L3 map to the local, worker, and architect provider tiers in `config/models.yaml`. Locally authored patches are verified through the same gates. Supplying `aegis run --dag ... --models config/models.yaml` enables configured providers for repair nodes that have frozen probes and lack patch artifacts.

An L3 candidate reaches the independent reviewer after the real Pass-After probe. The reviewer compares changed files with their Git baseline and checks for new unguarded divisors, unbounded retained collections, silent exception handlers, hard-coded credentials, and removed policy checks. The audit covers recognized Python and JavaScript/TypeScript patterns. Reflection, computed exports, dynamic keys, and general semantic equivalence are outside static coverage. Audit failures and interface violations carry negative rewards and prevent the candidate from staging. The hosted model providers were outside this drill. Codex authored the exercised diffs directly from the AST slices.

### Requirements and experience

`aegis spec "Keep frame cache history bounded" --repo .` produces an `ExpandedSpec` with normal-path, boundary, and failure-recovery acceptance criteria. It cites the AST slice and searches the selected WAL ledger's ExperienceBank. Existing behavior and requested limits guide edge cases. `aegis plan` attaches the spec to the plan. A missing business value produces a question and exit code 2. Run the same command with `--answers answers.json` to resolve it. The drill built a version 3 bank from four physical repair attempts, retrieved two records, and resolved the sole outstanding drawdown parameter with an explicit value of 0.08.

### Dataset and 8 GB handoff

The v3 manifest links the existing SFT and GRPO data to supplemental shards. Four public cat-cafe scenarios from `examples/cat_cafe_25d` produced four SFT examples and four GRPO pairs after eight physical dual-probe attempts. Training reads 472 SFT rows and 528 GRPO candidates in 264 groups. The historical split and files are preserved byte for byte. PR #8 and PR #9 report the previous private cat-cafe runs and PR #9 trace IDs. Their raw external ledger and traces were unavailable in this environment. The new rows are labeled as replays of the public example.

Run the CPU preflight with `python -m training.train_qlora_sft --dry-run` and `python -m training.train_grpo --dry-run`. Adding `--tokenizer-dir` selects local Hugging Face tokenizer files. The preflight validates dataset hashes, verifies masking and truncation, normalizes paired GRPO rewards, validates the LoRA settings used by the CUDA path, and writes checksummed checkpoints that are read back before the command succeeds. The checkpoints contain the prepared data and configuration. The original `training.train_sft` entry point and `scripts/bootstrap_local.ps1` still work for the Windows handoff.

The drill used the official Qwen2.5-Coder-3B-Instruct tokenizer on CPU. Default budgets accommodated the longest SFT sequence at 784 tokens and the longest GRPO prompt and completion at 665 and 167 tokens. Smaller budgets exercised prompt and completion truncation in both trainers. CUDA, GPU allocation, training throughput, and adapter weights require the RTX 5060 run.

### Reproduce the drill

```sh
python scripts/run_whitepaper_v10_drill.py --tokenizer-dir /path/to/qwen-tokenizer
```

The script runs the production CLI against temporary Git repositories and a shared WAL ledger. It verifies the contract rejection and downstream skip, two L3 outcomes after green probes, ExperienceBank retrieval and business clarification, the v3 dataset, and four CPU checkpoint exports. It checks that both original Git worktrees stay clean. See `STATUS.json` for the recorded test count and Actions matrix.

### Verification

The complete suite passed 385/385 tests, including the 294 existing cases. The implementation diff also passed an independent Fail-Before/Pass-After probe in a disposable worktree. The full matrix passed all eight push and pull_request jobs for code commit `3b06ee683caebdfd5328b114175590e2d4e6a94c`. [Push](https://github.com/Xiaoyu155/aegis-core/actions/runs/44247523641) · [Pull request](https://github.com/Xiaoyu155/aegis-core/actions/runs/44247529339)

A Windows 3.11 run exposed a Git config read racing with worktree metadata updates. The CLI, planner, scope gate and reviewer now acquire the same lock around Git commands. The concurrent worktree test passed in both Windows 3.11 matrix jobs after the fix.

## PR #13 — symbol impact, candidate tournament, and self-healing autopilot

PR #13 is stacked on PR #12 at baseline `e32c80469104e2bcaa42b19dd57bccc6d401a9c9`, branch `feat/world-model-v12`. It preserves all 507 baseline tests, the `training/` tree, and `scripts/bootstrap_local.ps1`. The final suite collected and passed 656 tests: 507 preserved and 149 added across the four implementation test modules.

### Symbol impact and budget-aware context

`aegis_core/impact_analyzer.py` builds a bidirectional symbol graph over Python, JavaScript, TypeScript, and HTML inline JavaScript. It resolves Python imports and aliases, local JS/TS imports, script references, calls, and visible type/reference edges. Caller and callee traversal is cycle safe and has a bounded depth. Reflection and computed property calls that cannot be resolved are surfaced as unresolved references instead of guessed edges.

`ImpactAnalyzer.compress_context()` keeps each directly changed file complete, uses signatures and annotations for one-hop dependencies, and summarizes farther symbols. It reports the lexical-token count, source-tree baseline, compression ratio, and recall of affected symbols; it fails explicitly when the changed files alone exceed the budget. The physical scale check indexed a temporary 1,000-file repository with 1,000 symbols and 250 resolved edges in 0.117 seconds. The 41 new tests cover cycles, aliased cross-file imports, HTML inline calls, computed properties, depth boundaries, and budget/recall behavior.

### Candidate tournament and structured probes

`CandidateTournament` evaluates each patch in a detached Git worktree with static, dynamic, mutation, and independent L3 review gates. It computes weighted score and Pareto eligibility from gate results, complexity growth, and token/time cost. Rejected patch pairs are appended with stable identities under a cross-process file lock to `datasets/grpo_tournament_pairs.jsonl`.

The expanded `probe_synth` compiler handles numeric bounds, finite-state transitions, idempotency/replay, bounded collections, async timeouts/fallbacks, and interface schema assertions. Each generated probe combines legal inputs, invalid inputs, and fixed-seed randomized cases. The compiler executes its probe against the frozen Git baseline and accepts it only after a real acceptance assertion fails with `ASSERTION_FAILED`. The 48 new probe cases and the tournament tests run against real temporary repositories and worktrees.

### `aegis autopilot` and replay learning

The new `aegis autopilot --config autopilot.json --repo . --out report.json` path expands the contract, retrieves experience hints, compresses impact context, compiles and red-checks a structured probe, runs candidate tournaments and records diagnosis before a retry. Token and wall-clock limits stop additional candidate processes when the budget fuse trips. Successful repairs paired with prior rejected traces are ingested as verified replay recoveries.

The two-round physical recovery case rejected the first candidates, returned a dynamic-probe diagnosis and a matching ExperienceBank hint to round two, then accepted the targeted repair. Convergence took two rounds and produced one verified recovery. The suite covers the CLI parser, wrong-patch diagnostics, empty budgets, ticket closure, patch/data provenance, and clean worktree recovery.

### Endurance evidence

`scripts/run_endurance_v13_stress.py` physically launches eight detached Git worktrees that concurrently update one SQLite WAL ledger and one `ExperienceBank`. All 8 workers completed; the ledger has 8 tickets and 24 traces; the bank has 8 records; the database reports `integrity_check=ok`; `database is locked` remained zero; and no worktree, temporary path, or main-tree change remained.

The same script launches the `aegis autopilot` CLI against a 12-work-order Python/JavaScript/HTML/build-tool project. The project includes a guarded Python order state machine, asynchronous replay-safe JavaScript events with bounded history, an HTML submission contract, and a Python build check. The tournament physically evaluated 47 candidates in isolated worktrees: 11 targeted repairs were accepted and 36 invalid or adversarial candidates were rejected. The planted faults included illegal state transitions, broken event signatures/replay behavior, test-cheating patches, a broken HTML contract, a missing build dependency, unbounded event history, an infinite timeout loop, and a token-budget overrun. Eleven work orders converged; the twelfth tripped `BUDGET_EXCEEDED_TOKENS` before starting candidates. All campaign worktrees were removed and the campaign repository ended clean.

The 36 real chosen/rejected patch pairs have SHA-256 `8cf50f972735c08eae5e1d5831b3b20cec442615315ad294fc58e1ab8dbbe9a3`. Full receipts and exact stdout are saved at `artifacts/endurance_v13/report.json` and `artifacts/endurance_v13/stdout.txt`. `STATUS.json` holds the machine-readable metrics. The 656-test local regression passed on Python 3.12.14 with Node v24.22.1; Linux/Windows Python 3.11/3.12 Actions are recorded after the PR run completes.

The cloud run had no commercial model API key, so candidate code was authored offline and evaluated through the production tournament gates. The Windows 11 RTX 5060 handoff remains intact; no GPU training claim is made for this Linux run.
