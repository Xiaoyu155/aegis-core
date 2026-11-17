# Causal world model

`CausalWorldModel` estimates the result of a proposed repair action before the
runner creates an execution worktree. It encodes the current ledger state,
patch/action and AST slice, then predicts a five-class next state, scalar
Reward and a calibrated Reward interval. The inference path is standard-library
only; model training uses NumPy on CPU.

## Train and evaluate

```bash
python scripts/train_and_eval_world_model.py
```

Artifacts are written under `artifacts/causal_world_model/`:

- `model_weights.npz`: trained numeric arrays.
- `model_weights.json`: portable inference weights, including Reward scale and
  validation-calibrated interval radius.
- `feature_encoder.json`: training-only vocabulary and numeric normalization.
- `training_metrics.json`: source hashes, hardware, epoch losses, before/after
  holdout metrics and risk supports.
- `split_manifest.json`: group assignment and sample IDs for audit.

The source accounting contains 1,050 executed synthetic action outcomes and all
47 physical campaign transitions. Of the physical rows, 11 contain a labeled
action outcome and 36 are lifecycle records; those lifecycle-only rows are
accounted for but excluded from supervised examples. The 1,061 labeled examples
are split by whole template family or ticket into 619 train, 177 validation and
265 test rows. The split manifest records groups and asserts there is no group
overlap. Vocabulary and feature scaling are fit only on the training split;
validation selects the best epoch and calibrates the Reward interval. The test
split is used for the before/after report only.

Training performs Adam updates for the state softmax, Reward regression and
risk sigmoid heads. Early stopping selects the best validation-loss weights.
The baseline is the same architecture at seeded random initialization. Risk
metrics on this dataset have only one high-risk holdout example, so they are a
smoke check, not a reliable population estimate. The exact pre-block guard is
reported separately from the learned risk head: it identifies patches that
edit tests or out-of-scope paths and exact patches repeated twice before
execution.

## Runtime traces

When trained artifacts are present, `python -m aegis_core run` stores a
`world_model_prediction` trace before probe/worktree execution, with the
predicted outcome, Reward, Reward interval and confidence. After execution, a
second append-only trace records `actual_state_s1`, the causal error and whether
the observed Reward fell inside the predicted interval. Exact pre-blocks are
recorded as `HIGH_RISK_PRE_BLOCK`; the model outcome is made one-hot for these
deterministic policy decisions. Exported JSONL keeps the legacy state/action/
reward keys and includes the additional fields.
