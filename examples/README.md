# Authored business rehearsal projects

These are executable offline engineering fixtures, not copied production systems.
Their default heads deliberately contain four defects: asymmetric leverage cap,
absorbing HALTED state, sprite-center depth ordering, and unbounded frame history.
The campaign freezes independent backtest/geometry assertions, runs each failing
baseline in a Git worktree, and applies repairs only to production source files.

`quant_engine` links market/account policy models, an 8-hour scheduler, funding
and leverage limits, recovery gating and deterministic fill accounting. No real
orders, keys or capital are involved. `cat_cafe_25d` links entity state, behavior,
container/depth ordering, bounded perspective, reusable draw buffers and a CPU
geometry/variance oracle. Buffer reuse is verified; Python allocations and GPU
pixels are not claimed to be zero or visually verified.

Run from the repository virtual environment:
`python scripts/run_real_world_campaign.py --output /tmp/aegis-campaign`
Use a new output directory on every run. The report distinguishes physical
execution from the authored offline patch policy. The self-healing case injects
a deterministic mock provider that requires the previous assertion diagnosis
before producing its second patch; no live model capability is inferred.
