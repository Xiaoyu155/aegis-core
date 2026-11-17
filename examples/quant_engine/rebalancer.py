from models import Market, Portfolio
from risk_guard import RiskGuard
from regime_gate import RegimeGate
from execution import ExecutionBook

class Rebalancer:
    PERIOD = 28800
    def __init__(self, risk: RiskGuard, gate: RegimeGate):
        self.risk, self.gate = risk, gate
        self.book = ExecutionBook()

    def step(self, portfolio: Portfolio, market: Market, factor: float):
        """8-hour rebalance; regime evaluated on every tick, executions once/cycle."""
        if market.timestamp < 0:
            raise ValueError('negative timestamp')
        allowed = self.gate.update(market)
        cycle = market.timestamp // self.PERIOD
        if cycle <= portfolio.last_cycle or not allowed:
            return None
        target = self.risk.bounded_target(portfolio, market, factor)
        fill = self.book.submit(portfolio, market, target, cycle)
        order = {'fill': fill, 'cycle':cycle,'delta':target-portfolio.notional, 'target':target}
        portfolio.notional = target
        portfolio.last_cycle = cycle
        portfolio.orders.append(order)
        return order
