"""Deterministic net-position fill accounting, without exchange side effects."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Fill:
    cycle: int
    quantity: float
    fee: float
    reduce_only: bool

class ExecutionBook:
    def __init__(self, fee_rate=.0004):
        if not math.isfinite(fee_rate) or fee_rate<0:
            raise ValueError('fee rate')
        self.fee_rate=fee_rate
        self.fills=[]
        self.cycles=set()

    def submit(self, portfolio, market, target, cycle, *, reduce_only=False):
        if not all(math.isfinite(value) for value in (target,market.price,portfolio.notional)) or market.price<=0:
            raise ValueError('invalid execution inputs')
        if cycle in self.cycles:
            return None
        if reduce_only and (abs(target)>abs(portfolio.notional) or target*portfolio.notional<0):
            raise ValueError('reduce-only exposure increase')
        delta=target-portfolio.notional
        fill=Fill(cycle,delta/market.price,abs(delta)*self.fee_rate,reduce_only)
        self.fills.append(fill)
        self.cycles.add(cycle)
        return fill
