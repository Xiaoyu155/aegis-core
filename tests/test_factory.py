import json
import time
from pathlib import Path
import pytest
from aegis_core.ast_slicer import ImportGraph
from aegis_core.mediator import compile_wish
from aegis_core.router import repair_json, Router
from aegis_core.cli import _offline_router
from aegis_core.sandbox import validate_probe_patch

@pytest.mark.parametrize('encoding', ['utf-8','utf-8-sig','gbk'])
def test_encoding(encoding,tmp_path):
    (tmp_path/'x.py').write_bytes('# 中文\ndef value(x: int) -> int:\n    return x\n'.encode(encoding))
    graph=ImportGraph(tmp_path)
    assert 'x:value' in graph.symbols

@pytest.mark.parametrize('budget',[256,300,500,800,1000,2000,4000,8000])
def test_budget(budget,tmp_path):
    (tmp_path/'x.py').write_text('def value(x):\n    return x\n'+'#padding\n'*100)
    result=ImportGraph(tmp_path).slice('x.py','value',token_budget=budget)
    assert result['token_upper_bound'] <= budget

@pytest.mark.parametrize('kind',['sync','async','nested','decorator','types'])
def test_graph(kind,tmp_path):
    (tmp_path/'pkg').mkdir(); (tmp_path/'pkg/__init__.py').write_text('')
    (tmp_path/'pkg/model.py').write_text('class Model:\n    pass\ndef helper(x):\n    return x\n')
    source='from .model import helper, Model\ndef target(x: Model):\n    return helper(x)\n'
    if kind=='async': source=source.replace('def target','async def target')
    if kind=='nested': source='class Outer:\n    class Inner:\n        def target(self):\n            return 1\n'
    if kind=='decorator': source='def decorate(f):\n    return f\n@decorate\n'+source.split('def target')[1].join(['def target',''])
    if kind=='decorator': source='def decorate(f):\n    return f\n@decorate\ndef target(x):\n    return x\n'
    (tmp_path/'pkg/main.py').write_text(source)
    graph=ImportGraph(tmp_path); symbol='Outer.Inner.target' if kind=='nested' else 'target'
    result=graph.slice('pkg/main.py',symbol)
    assert result['items']
    if kind in ('sync','async','types'):
        assert any('helper' in item['symbol'] for item in result['items'])

@pytest.mark.parametrize('text',['{}','{"patch":"ok"}','```json\n{"a":1}\n```','{"a":1,}','{"a":1'])
def test_json_repair(text):
    assert isinstance(repair_json(text),dict)

@pytest.mark.parametrize('text',['{"a":"half','junk','{a:1}','[','{"a":NaN}'])
def test_bad_json(text):
    with pytest.raises(ValueError): repair_json(text)

@pytest.mark.parametrize('word',['回撤','亏损','平仓'])
def test_risk_missing(word,tmp_path):
    (tmp_path/'risk.py').write_text('def guard(peak: float, equity: float) -> bool:\n    return False\n')
    result=compile_wish(tmp_path,word)
    assert 'max_drawdown' in result['missing']
    assert 'unlock_condition' in result['missing']
    assert 'close_method' in result['missing']

@pytest.mark.parametrize('value',[0,1,2,5,10,20,30,99])
def test_literal(value,tmp_path):
    (tmp_path/'calc.py').write_text('def add(x: int) -> int:\n    return x\n')
    result=compile_wish(tmp_path,f'calc.py add({value}) returns {value+1}')
    assert not result['missing']
    validate_probe_patch(result['probe_patch'])
    assert 'x: int' in result['contract_stub']

@pytest.mark.parametrize('patch',['','garbage','--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b\n'])
def test_bad_probe(patch):
    with pytest.raises((ValueError,SyntaxError)): validate_probe_patch(patch)


def test_real_socket(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from aegis_core.router import request_provider
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert data['model']=='local'
            self.send_response(200);self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"content":"{\\"ok\\":true}"}}]}')
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    node={'model':'local','model_env':'','base_url':f'http://127.0.0.1:{server.server_port}/v1','base_url_env':'','api_key_env':'','provider':'openai_compatible'}
    try: assert json.loads(request_provider(node,[],timeout=2,deadline=time.monotonic()+3))['ok']
    finally: server.shutdown();server.server_close()


def test_precheck_and_auto_end(tmp_path):
    from aegis_core import cli
    import contextlib,io,subprocess
    root=tmp_path/'repo';root.mkdir()
    def git(*args):subprocess.run(['git',*args],cwd=root,capture_output=True,check=True)
    (root/'calc.py').write_text('def add(x: int) -> int:\n    return x\n')
    git('init','-b','main');git('add','.');git('-c','user.name=t','-c','user.email=t@x','commit','-m','x')
    db=tmp_path/'db';out=tmp_path/'contract';patch=tmp_path/'patch';patch.write_text('not a patch')
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli.main(['wish','calc.py add(2) returns 2','--repo',str(root),'--db',str(db),'--out',str(out),'--patch',str(patch)])==0
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli.main(['run','--contract',str(out),'--db',str(db)])==1
    from aegis_core.ledger import Ledger
    assert Ledger(db).stats()['statuses']=={'PRECHECK_FAILED':1}

@pytest.mark.parametrize('workers',[0,-1,33])
def test_queue_invalid(workers):
    from aegis_core.queue_worker import run_contracts
    with pytest.raises(ValueError):run_contracts([],db='unused',workers=workers)

@pytest.mark.parametrize('code',[429,500,503])
def test_retry(code):
    import urllib.error
    router=_offline_router();router.sleep=lambda _:None;router.jitter=lambda:0
    calls=[]
    def transport(*args,**kwargs):
        calls.append(1)
        if len(calls)<2:raise urllib.error.HTTPError('http://localhost',code,'transient',{'Retry-After':'0'},None)
        return '{"ok":true}'
    result=router.call('offline',[],deadline=time.monotonic()+10,transport=transport,require_json=True)
    assert result['status']=='admitted' and len(calls)==2


def test_parallel_wal_worktrees(tmp_path):
    from aegis_core import cli
    from aegis_core.queue_worker import run_contracts
    from aegis_core.ledger import Ledger
    import subprocess,io,contextlib,difflib
    root=tmp_path/'repo';root.mkdir()
    (root/'calc.py').write_text('def add(x):\n    return x\n')
    def git(*args): subprocess.run(['git',*args],cwd=root,capture_output=True,check=True)
    git('init','-b','main');git('add','.');git('-c','user.name=t','-c','user.email=t@x','commit','-m','x')
    patch=tmp_path/'patch';patch.write_text(''.join(difflib.unified_diff(['def add(x):\n','    return x\n'],['def add(x):\n','    return x+1\n'],fromfile='a/calc.py',tofile='b/calc.py')))
    db=tmp_path/'db';paths=[]
    for i in range(6):
        path=tmp_path/f'{i}.json';paths.append(path)
        with contextlib.redirect_stdout(io.StringIO()):
            assert cli.main(['wish','calc.py add(2) returns 3','--repo',str(root),'--db',str(db),'--out',str(path),'--patch',str(patch)])==0
    results=run_contracts(paths,db=db,workers=3)
    assert all(code==0 for _,code in results),results
    assert Ledger(db).stats()['success_rate']==1
    assert not list((root/'.worktrees').iterdir())

@pytest.mark.parametrize('failure',['timeout','429','bad_json'])
def test_gateway_failover(failure):
    import urllib.error
    router=_offline_router();node=dict(router.nodes['offline']);node['id']='backup'
    config=router.config;config['tiers']['offline'].append({k:v for k,v in node.items() if k!='tier'})
    router=Router(config,sleep=lambda x:None,jitter=lambda:0)
    calls=[]
    def transport(node,messages,**kwargs):
        calls.append(node['id'])
        if node['id']=='backup':return '{"ok":true}'
        if failure=='timeout':raise TimeoutError()
        if failure=='429':raise urllib.error.HTTPError('x',429,'busy',{'Retry-After':'0'},None)
        return '{"broken":'
    # Fairness initially selects backup alphabetically; force the failed primary first.
    router._served['backup']=100
    result=router.call('offline',[],deadline=time.monotonic()+10,transport=transport)
    assert result['slot_id']=='backup'
    assert router.events


def test_graph_line_and_callers(tmp_path):
    (tmp_path/'b.py').write_text('def callee(x):\n    return x\n')
    (tmp_path/'a.py').write_text('from b import callee\ndef caller(x):\n    return callee(x)\n')
    graph=ImportGraph(tmp_path)
    result=graph.slice('b.py',line=2)
    assert any(i['relation']=='caller' for i in result['items'])
