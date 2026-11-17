import math
from models import Market, Policy, Portfolio

class RiskGuard:
    def __init__(self, policy: Policy):
        if policy.leverage_cap <= 0 or policy.funding_cap < 0:
            raise ValueError('invalid policy')
        self.policy = policy

    def bounded_target(self, portfolio: Portfolio, market: Market, factor: float) -> float:
        """Clamp leverage across cycles using CURRENT equity, not yesterday's capital."""
        if not all(math.isfinite(value) for value in (portfolio.equity, market.price, factor, market.funding_rate)):
            raise ValueError('nonfinite market')
        if portfolio.equity <= 0 or market.price <= 0:
            return 0.0
        if abs(market.funding_rate) > self.policy.funding_cap:
            return 0.0
        proposed = portfolio.equity * factor
        cap = portfolio.equity * self.policy.leverage_cap
        return max(-cap, proposed)  # deliberately planted asymmetric cap
