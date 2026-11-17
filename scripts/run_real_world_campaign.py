"""Offline authored projects: real executions, not real exchange/GPU deployments."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import contextlib
import io
import json
import difflib
import shutil
import subprocess
import tempfile
from aegis_core import cli
from aegis_core.ledger import Ledger
from aegis_core.ast_slicer import ImportGraph, signature
from unittest.mock import patch as mock_patch

ROOT=Path(__file__).resolve().parents[1]
FIXES={
 'risk_guard.py':('return max(-cap, proposed)','return max(-cap, min(cap, proposed))','return proposed'),
 'regime_gate.py':('return False  # deliberately planted absorbing-state deadlock','self.state = Regime.RECOVERING\n            self.healthy_ticks = 0','return False'),
 'depth_sorter.py':('cat.body_y*self.scale(cat)','cat.foot_y','-cat.foot_y'),
 'render_loop.py':('self.history.append(tuple(self.draw_buffer))  # deliberately planted unbounded history','self.history[:] = [None]','self.history.append(None)'),
}

def git(root,*args):
    return subprocess.run(['git',*args],cwd=root,capture_output=True,check=True).stdout.decode()

def invoke(*args):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        code=cli.main(list(args))
    return code,json.loads(out.getvalue())

def campaign(output):
    __import__('pytest')  # refuse an infrastructure failure masquerading as a red assertion
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    db=output/'campaign.sqlite'
    if db.exists():raise ValueError('use a fresh output directory')
    summaries=[]
    with tempfile.TemporaryDirectory(prefix='campaign-') as d:
        work=Path(d)
        cases=[('leverage','risk_guard.py','test_backtest.py::test_leverage'),('regime','regime_gate.py','test_backtest.py::test_recovery'),
               ('depth','depth_sorter.py','test_geometry.py::test_depth'),('memory','render_loop.py','test_geometry.py::test_memory'),
               ('ambiguity','depth_sorter.py','test_geometry.py::test_depth'),('self_heal','risk_guard.py','test_backtest.py::test_leverage'),
               ('cheat','risk_guard.py','test_backtest.py::test_leverage'),('replay','risk_guard.py','test_backtest.py::test_leverage')]
        for case,target,test in cases:
            root=work/case;shutil.copytree(ROOT/'examples'/('quant_engine' if target in ('risk_guard.py','regime_gate.py') else 'cat_cafe_25d'),root)
            git(root,'init','-b','main');git(root,'add','.');git(root,'-c','user.name=t','-c','user.email=t@localhost','commit','-m','frozen campaign')
            if case=='ambiguity':
                code,result=invoke('wish','猫咪走动时层级遮挡错乱','--repo',str(root),'--db',str(db),'--ticket-id',case)
                hit=code==2 and result['compilation']['status']=='MISSING_CONSTRAINT'
                summaries.append({'case':case,'expected':'MISSING_CONSTRAINT','actual':result['compilation']['status'],'hit':hit})
                continue
            old,good,bad=FIXES[target];source=(root/target).read_text()
            def patch(replacement):return ''.join(difflib.unified_diff(source.splitlines(True),source.replace(old,replacement).splitlines(True),fromfile='a/'+target,tofile='b/'+target))
            graph=ImportGraph(root)
            key=next(key for key in graph.symbols if key.endswith({'risk_guard.py':':RiskGuard.bounded_target','regime_gate.py':':RegimeGate.update','depth_sorter.py':':DepthSorter.key','render_loop.py':':RenderLoop.tick'}[target]))
            proposal=work/(case+'.proposal');proposal.write_text(json.dumps({'target_files':[target],'contract_stub':signature(graph.symbols[key])}))
            candidates=[patch(good)]
            if case=='self_heal':candidates=[patch(bad),patch(good)]
            if case=='replay':candidates=[patch(bad)]*3
            if case=='cheat':
                name='test_backtest.py';before=(root/name).read_text()
                candidates=[''.join(difflib.unified_diff(before.splitlines(True),['def test_fake():\n','    assert True\n'],fromfile='a/'+name,tofile='b/'+name))]
            manifest=work/(case+'.json');args=['wish','repair '+target,'--no-auto-probe','--repo',str(root),'--db',str(db),'--ticket-id',case,'--out',str(manifest),'--proposal',str(proposal)]
            probe=json.dumps(['{python}','-B','-m','pytest','-q','--tb=short',test])
            args+=['--fail-before',probe,'--pass-after',probe]
            for i,p in enumerate(candidates):
                file=work/(case+str(i)+'.diff');file.write_text(p);args+=['--patch',str(file)]
            if case=='self_heal':
                config=work/'offline_models.json';config.write_text(json.dumps(cli._offline_router().config))
                args+=['--allow-model','--models',str(config)]
            code,compiled=invoke(*args)
            if code: raise AssertionError(compiled)
            def feedback_policy(node,messages,**kwargs):
                context=json.loads(messages[-1]['content'])
                prior=context['prior_attempts']
                if not prior: return json.dumps({'patch':patch(bad)})
                diagnosis=prior[-1]['diagnosis']
                assert diagnosis['category']=='assertion' and diagnosis['contexts']
                return json.dumps({'patch':patch(good)})
            with mock_patch('aegis_core.router.request_provider',side_effect=feedback_policy) if case=='self_heal' else contextlib.nullcontext():
                code,result=invoke('run','--contract',str(manifest),'--db',str(db))
            attempts=result['results']
            for attempt in attempts:
                if attempt.get('baseline'):
                    assert attempt['baseline']['exit_code']==1 and 'assert ' in attempt['baseline']['stdout'], attempt['baseline']
            hit=(result['status']=='succeeded') if case not in ('cheat','replay') else (
                 attempts[0]['cheat_detected'] if case=='cheat' else result['reason']=='blocked_needs_human')
            if case=='self_heal':hit=hit and len(attempts)==2 and attempts[0]['reward']<0 and 'diagnosis' in attempts[0]
            if list((root/'.worktrees').iterdir()):raise AssertionError('unclean')
            summaries.append({'case':case,'actual':result['status'],'hit':hit,'rewards':[r['reward'] for r in attempts],
                              'probes':[{'before_exit':(r.get('baseline') or {}).get('exit_code'),'after_exit':(r.get('final') or {}).get('exit_code'),'diagnosis':r.get('diagnosis',{}).get('category')} for r in attempts]})
    ledger=Ledger(db)
    count=ledger.export_jsonl(output/'campaign_traces.jsonl')
    report={'cases':summaries,'all_passed':all(c['hit'] for c in summaries),'stats':ledger.stats(),
            'real_campaign_traces_count':count,'provenance':'authored offline projects; physically executed CLI/worktree probes; self_heal uses a diagnosis-driven deterministic mock provider, not a live LLM'}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='datasets/campaign');args=parser.parse_args()
    report=campaign(args.output)
    if report['all_passed'] and Path(args.output).resolve()==(ROOT/'datasets/campaign').resolve():
        from aegis_core.dataset_builder import distill_campaign
        report['distillation']=distill_campaign(Path(args.output)/'campaign_traces.jsonl',ROOT/'datasets')
        (Path(args.output)/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));sys.exit(0 if report['all_passed'] else 1)
