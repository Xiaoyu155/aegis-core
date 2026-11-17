# Frozen V1 contracts

Freeze point: this file and v1.pyi precede implementation in Git history.

Ledger: ticket creation records step 0. Every subsequent lifecycle status change
requires a live opaque lease token, expected status, and a causal tuple in one
SQLite transaction. Lease operations are coordination metadata, not world-state
transitions. JSONL uses state_s1 as a training alias for actual_state_s1.
Terminal states are succeeded/failed; allowed edges are queued -> running and
running -> succeeded/failed. Expired leases can be reacquired with a new token.

Slicer: deterministic static analysis of Python modules in the target directory;
resolve same-module functions, imported function aliases, module aliases, classes
and methods. Report unresolved/dynamic calls explicitly; never import target code.
Result keys: target, downstream, upstream, unresolved_calls, compression_ratio.

Sandbox: Git worktrees isolate file changes, not hostile code execution. Tests run
without shell expansion. Gates return stdout/stderr/exit_code/reward and before/
after tree hashes. Only an unchanged, passing tree may be merged. git apply uses
check then atomic apply; directory/branch lifecycle is scoped to validated IDs.

Protocol: strict JSON discriminated by kind; reject unknown fields, wrong types,
nonfinite numbers, missing DAG dependencies/cycles and duplicate ticket IDs.
Plans include initial_state, steps with state_s0/action_a0/predicted_next_state,
and sub_tickets with ticket_id/intent/contract_stub/depends_on.
