"""Run A-D on this repository, without any paid model or source mutation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contextlib
import io
import json
import difflib
import tempfile
from aegis_core import cli


def exercise(root, directory):
    db = directory / 'ledger.sqlite'
    def command(*args):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = cli.main([*args, '--db', str(db)])
        return code, json.loads(out.getvalue())
    source = (root / 'aegis_core/router.py').read_text()
    def patch(value):
        after = source.replace('ALIASES = {', 'ALIASES = {"factory_alias": ' + repr(value) + ', ', 1)
        return ''.join(difflib.unified_diff(source.splitlines(True), after.splitlines(True), fromfile='a/aegis_core/router.py', tofile='b/aegis_core/router.py'))
    good, bad = directory / 'good.diff', directory / 'bad.diff'
    good.write_text(patch('codex_cli'))
    bad.write_text(patch('anthropic'))
    wish = "aegis_core/router.py normalize_provider('factory_alias') returns 'codex_cli'"
    results = {}
    code, a = command('wish', '亏损大了自动平仓', '--repo', str(root), '--ticket-id', 'scenario-a')
    results['A'] = code == 2 and a['compilation']['status'] == 'MISSING_CONSTRAINT'
    for name, candidates in [('b', [bad, good]), ('c', []), ('d', [bad, bad, bad])]:
        manifest = directory / (name + '.json')
        args = ['wish', wish, '--repo', str(root), '--ticket-id', 'scenario-' + name, '--out', str(manifest)]
        if name == 'c':
            probe = a  # The generated test is reconstructed below, not trusted from a model.
            code, compiled = command(*args)
            added = '\n'.join(line[1:] for line in compiled['probe_patch'].splitlines() if line.startswith('+') and not line.startswith('+++')) + '\n'
            cheat = directory / 'cheat.diff'
            cheat.write_text(''.join(difflib.unified_diff(added.splitlines(True), ['def test_fake():\n', '    assert True\n'], fromfile='a/tests/test_aegis_probe.py', tofile='b/tests/test_aegis_probe.py')))
            # New ticket is needed after immutable compilation.
            args[args.index('scenario-c')] = 'scenario-c-final'
            candidates = [cheat]
        for path in candidates:
            args += ['--patch', str(path)]
        code, compiled = command(*args)
        code, result = command('run', '--contract', str(manifest))
        results[name.upper()] = (result['status'] == 'succeeded' if name == 'b' else
                                result['reason'] == 'blocked_needs_human' if name == 'd' else result['results'][0]['cheat_detected'])
        results[name + '_evidence'] = {'status': result['status'], 'reason': result['reason'],
                                      'rewards': [r['reward'] for r in result['results']]}
    code, stats = command('stats', '--repo', str(root))
    results['stats'] = stats
    results['all_passed'] = all(results[key] for key in 'ABCD')
    return results


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='dogfood-') as directory:
        report = exercise(root, Path(directory))
    print(json.dumps(report, ensure_ascii=False))
    if not report['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
