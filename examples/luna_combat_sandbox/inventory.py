"""Minimal stock-allocation business rule for isolated acceptance exercises."""


def clamp_stock(value: object) -> int | None:
    """Return a sellable stock quantity; this baseline has a planted lower-bound defect."""
    if type(value) is not int:
        return 0
    return min(10, value)
