"""Boundary assertions against executable authored business projects."""
import importlib
import json
from pathlib import Path
import shutil
import sys
import pytest
from aegis_core.runner import retrieve, diagnose
from aegis_core.mediator import compile_wish
from scripts.run_real_world_campaign import FIXES, campaign

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def project(tmp_path, monkeypatch):
    loaded=[]
    def load(name):
        path=tmp_path/name
        shutil.copytree(ROOT/'examples'/name,path)
        for file,(old,good,_) in FIXES.items():
            target=path/file
            if target.exists(): target.write_text(target.read_text(encoding='utf-8').replace(old,good))
        monkeypatch.syspath_prepend(str(path))
        for file in path.glob('*.py'):
            if file.stem in sys.modules:
                monkeypatch.delitem(sys.modules,file.stem)
            loaded.append(file.stem)
        return lambda module: importlib.import_module(module)
    yield load
    for name in loaded: sys.modules.pop(name,None)

@pytest.mark.parametrize('factor',[-100,-3,-2,0,2,3,100])
def test_current_equity_caps(project,factor):
    m=project('quant_engine'); models=m('models')
    guard=m('risk_guard').RiskGuard(models.Policy())
    for equity in (1000,500,10):
        assert abs(guard.bounded_target(models.Portfolio(equity),models.Market(0,100,0,.01,1),factor))<=equity*2

@pytest.mark.parametrize('funding',[-.1,-.002,.002,.1])
def test_funding_shutdown(project,funding):
    m=project('quant_engine'); models=m('models')
    assert m('risk_guard').RiskGuard(models.Policy()).bounded_target(models.Portfolio(100),models.Market(0,1,funding,0,1),2)==0

@pytest.mark.parametrize('value',[float('inf'),float('nan'),float('-inf')])
def test_nonfinite_market(project,value):
    m=project('quant_engine'); models=m('models')
    with pytest.raises(ValueError): m('risk_guard').RiskGuard(models.Policy()).bounded_target(models.Portfolio(100),models.Market(0,value,0,0,1),1)

@pytest.mark.parametrize('ticks',[1,2,3,5])
def test_regime_recovery_and_relapse(project,ticks):
    m=project('quant_engine'); models=m('models'); gate=m('regime_gate').RegimeGate(models.Policy(recovery_ticks=ticks))
    danger=models.Market(0,1,0,.3,1);calm=models.Market(1,1,0,0,1)
    assert not gate.update(danger)
    for _ in range(ticks-1): assert not gate.update(calm)
    assert gate.update(calm)
    assert not gate.update(danger)


def test_eight_hour_cycle_idempotency(project):
    m=project('quant_engine'); models=m('models');p=models.Portfolio(100)
    engine=m('rebalancer').Rebalancer(m('risk_guard').RiskGuard(models.Policy()),m('regime_gate').RegimeGate(models.Policy()))
    for second in (0,1,28799,28800,28801,57600):engine.step(p,models.Market(second,1,0,0,1),1)
    assert [order['cycle'] for order in p.orders]==[0,1,2]

@pytest.mark.parametrize('foot,expected',[(-1000,.5),(-50,.5),(0,1),(50,1.5),(1000,2)])
def test_perspective_bounds(project,foot,expected):
    m=project('cat_cafe_25d')
    assert m('depth_sorter').DepthSorter().scale(m('entities').Cat('a',foot,1))==expected


def test_container_anchor_stable_tie(project):
    m=project('cat_cafe_25d');Cat=m('entities').Cat
    cats=[Cat('b',5,99),Cat('a',5,-99),Cat('overlay',-100,0,1)]
    assert [cat.ident for cat in m('depth_sorter').DepthSorter().order(cats)]==['a','b','overlay']


def test_buffer_reuse_and_removed_entities(project):
    m=project('cat_cafe_25d');cats=[m('entities').Cat('a',1,1)]
    loop=m('render_loop').RenderLoop(cats,2);buffer=loop.draw_buffer
    for _ in range(500):loop.tick(.01)
    cats.clear();assert loop.tick(0)==0
    assert buffer is loop.draw_buffer and buffer==[None,None] and len(loop.history)<=1

@pytest.mark.parametrize('wish,expected', [('猫咪走动时层级遮挡错乱','depth_sorter.py'),('极端行情状态机死锁','regime_gate.py'),('调仓杠杆失控','risk_guard.py'),('帧循环内存泄漏','render_loop.py')])
def test_retrieval_business_phenomena(wish,expected):
    domain='cat_cafe_25d' if '猫' in wish or '帧' in wish else 'quant_engine'
    result=retrieve(ROOT/'examples'/domain,wish)
    assert result['candidates'][0]['file']==expected
    assert result['context']['items'] and result['candidates'][0]['line']>0
    assert not any(row['file'].startswith('test_') for row in result['candidates'])


def test_unsupported_semantics_requires_invariant():
    result=compile_wish(ROOT/'examples/cat_cafe_25d','猫咪走动时层级遮挡错乱')
    assert result['missing'] and result['context']['retrieval']
    assert result['context']['retrieval'][0]['symbol'].split('.')[-1] in result['contract_stub']


def test_stack_diagnosis(tmp_path):
    (tmp_path/'calc.py').write_text('def add(x):\n    return x\n')
    result=diagnose(tmp_path,{'final':{'stdout':'calc.py:2: AssertionError\nassert 2 == 3','stderr':''}},attempt=1)
    assert result['frames']==[{'file':'calc.py','line':2}]
    assert result['category']=='assertion' and result['contexts'] and result['assertions']


def test_eight_actual_cli_orders(tmp_path):
    report=campaign(tmp_path/'campaign')
    assert report['all_passed'] and len(report['cases'])==8
    assert report['stats']['cheat_interceptions']==report['stats']['replay_interceptions']==1
    rows=[json.loads(line) for line in (tmp_path/'campaign/campaign_traces.jsonl').read_text(encoding='utf-8').splitlines()]
    attempts=[row for row in rows if row['action_a0'].get('op')=='repair_attempt']
    healing=[row for row in attempts if row['ticket_id']=='self_heal']
    assert len(healing)==2 and healing[0]['reward']<0<healing[1]['reward']
    assert healing[0]['actual_state_s1']['diagnosis']['contexts']
    assert all('No module named pytest' not in json.dumps(row) for row in rows)

@pytest.mark.parametrize('point',[(-2,3),(0,0),(5,7)])
def test_projection_round_trip(project,point):
    m=project('cat_cafe_25d'); view=m('geometry').Viewport(800,600)
    assert view.unproject(*view.project(*point))==pytest.approx(point)


def test_blank_frame_gate(project):
    m=project('cat_cafe_25d');metrics=m('geometry').FrameMetrics()
    for _ in range(100):metrics.sample(1)
    assert not metrics.healthy()
    metrics.sample(255);assert metrics.healthy()


def test_reduce_only_refuses_flip(project):
    m=project('quant_engine'); models=m('models'); book=m('execution').ExecutionBook()
    p=models.Portfolio(100,10);market=models.Market(0,1,0,0,1)
    with pytest.raises(ValueError):book.submit(p,market,-10,0,reduce_only=True)
    assert book.submit(p,market,0,0,reduce_only=True).quantity==-10
    assert book.submit(p,market,0,0,reduce_only=True) is None


def test_distillation_provenance_idempotency(tmp_path):
    from aegis_core.dataset_builder import distill_campaign
    rows=[]
    for passed,reward in [(False,-1),(True,1)]:
        rows.append({'ticket_id':'repair','state_s0':{'status':'running'},'action_a0':{'op':'repair_attempt','patch':'good' if passed else 'bad'},'actual_state_s1':{'passed':passed},'reward':reward})
    path=tmp_path/'raw.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows))
    first=distill_campaign(path,tmp_path);second=distill_campaign(path,tmp_path)
    assert first==second and first['real_contrastive_groups']==1
    assert len((tmp_path/'sft_mediator_eval.jsonl').read_text(encoding='utf-8').splitlines())==1
    assert len((tmp_path/'grpo_causal_rollouts.jsonl').read_text(encoding='utf-8').splitlines())==1
