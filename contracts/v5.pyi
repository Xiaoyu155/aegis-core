from pathlib import Path

def retrieve(root: Path, wish: str, *, token_budget: int = 4000) -> dict: ...
def diagnose(root: Path, feedback: dict, *, attempt: int, token_budget: int = 4000) -> dict: ...
# Campaign executes 8 offline work orders. Evidence includes actual CLI return
# codes, ledger transitions and retained patches; no production capital/network.
# Examples model geometry/state, not full GPU rendering or zero allocation.
