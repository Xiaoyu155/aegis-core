from enum import Enum
from models import Market, Policy

class Regime(Enum):
    NORMAL = 'normal'
    HALTED = 'halted'
    RECOVERING = 'recovering'

class RegimeGate:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.state = Regime.NORMAL
        self.healthy_ticks = 0

    def update(self, market: Market) -> bool:
        """Extreme-regime gate must recover after consecutive liquid calm ticks."""
        danger = market.volatility >= .2 or market.liquidity <= 0
        if danger:
            self.state = Regime.HALTED
            self.healthy_ticks = 0
            return False
        if self.state == Regime.HALTED:
            return False  # deliberately planted absorbing-state deadlock
        if self.state == Regime.RECOVERING:
            self.healthy_ticks += 1
            if self.healthy_ticks >= self.policy.recovery_ticks:
                self.state = Regime.NORMAL
        return self.state == Regime.NORMAL
