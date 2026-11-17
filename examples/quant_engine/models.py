from dataclasses import dataclass, field

@dataclass(frozen=True)
class Market:
    timestamp: int
    price: float
    funding_rate: float
    volatility: float
    liquidity: float

@dataclass(frozen=True)
class Policy:
    leverage_cap: float = 2.0
    funding_cap: float = .001
    recovery_ticks: int = 2

@dataclass
class Portfolio:
    equity: float
    notional: float = 0.0
    last_cycle: int = -1
    orders: list = field(default_factory=list)
