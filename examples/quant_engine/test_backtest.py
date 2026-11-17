from models import Policy, Market, Portfolio
from rebalancer import Rebalancer
from regime_gate import Regime, RegimeGate
from risk_guard import RiskGuard

def test_leverage():
    policy=Policy(); portfolio=Portfolio(1000)
    engine=Rebalancer(RiskGuard(policy),RegimeGate(policy))
    for cycle,equity,factor in [(0,1000,1),(1,600,8),(2,500,-8),(3,200,3)]:
        portfolio.equity=equity
        engine.step(portfolio,Market(cycle*28800,100,.0001,.01,10),factor)
        assert abs(portfolio.notional) <= equity*policy.leverage_cap

def test_recovery():
    gate=RegimeGate(Policy(recovery_ticks=2))
    assert not gate.update(Market(0,100,0,.3,10))
    assert not gate.update(Market(1,100,0,.01,10))
    assert gate.update(Market(2,100,0,.01,10))
