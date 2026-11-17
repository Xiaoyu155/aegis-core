"""Evidence-driven retrieval and failure attribution; never a model-based judge."""
import ast
from collections import Counter
import math
import re
from .ast_slicer import ImportGraph

CONCEPTS = {
 'depth': ('depth','sort','occlusion','perspective','anchor','层级','遮挡','透视'),
 'motion': ('cat','walk','position','frame','猫','走动'),
 'leverage': ('leverage','margin','notional','exposure','杠杆','调仓'),
 'regime': ('regime','halt','crash','extreme','lock','极端','行情','死锁'),
 'memory': ('buffer','history','allocation','leak','gc','内存','帧'),
 'risk': ('risk','drawdown','funding','limit','风控','亏损'),
}


def terms(text):
    lowered = re.sub(r'([a-z])([A-Z])', r'\1 \2', text).lower()
    words = re.findall(r'[a-z_]+', lowered)
    result = set(part for word in words for part in word.split('_') if part)
    for concept, synonyms in CONCEPTS.items():
        if any(word in lowered for word in synonyms):
            result.add('concept_' + concept)
    return result


def retrieve(root, wish, *, token_budget=4000):
    graph = ImportGraph(root)
    query = terms(wish)
    documents = {}
    for key, node in graph.symbols.items():
        filename, source, _ = graph.modules[graph.symbol_modules[key]]
        if filename.split('/')[-1].startswith('test_') or any(part in {'tests','training','scripts'} for part in filename.split('/')):
            continue
        # Names, docstrings, signatures and referenced field names carry meaning.
        text = key + ' ' + (ast.get_docstring(node) or '') + ' ' + ast.unparse(node)
        documents[key] = terms(text)
    counts = Counter(term for values in documents.values() for term in values)
    ranked = []
    for key, values in documents.items():
        overlap = query & values
        if not overlap:
            continue
        score = sum(math.log(1 + len(documents) / counts[term]) for term in overlap) / math.log2(2+len(values))
        node = graph.symbols[key]
        score += .5 * len(query & terms((ast.get_docstring(node) or '') + ' ' + key))
        if isinstance(node, ast.ClassDef):
            score *= .5
        if key.endswith('.__init__'):
            score *= .5
        filename, _, _ = graph.modules[graph.symbol_modules[key]]
        if filename in wish or key.split(':')[-1] in wish:
            score += 10
        node = graph.symbols[key]
        ranked.append({'file':filename,'symbol':key.split(':',1)[1], 'line':node.lineno,
                       'end_line':node.end_lineno,'score':score, 'matched_terms':sorted(overlap)})
    ranked.sort(key=lambda row:(-row['score'],row['file'],row['line']))
    context = graph.slice(ranked[0]['file'],ranked[0]['symbol'],token_budget=token_budget) if ranked else {}
    return {'candidates':ranked[:5], 'context':context,
            'method':'static_multilingual_concept_retrieval', 'unresolved': not bool(ranked),
            'ambiguity_margin': (ranked[0]['score']-ranked[1]['score'] if len(ranked)>1 else 1.0)}


def diagnose(root, feedback, *, attempt, token_budget=4000):
    graph = ImportGraph(root)
    final = feedback.get('final') or feedback.get('baseline') or {}
    output = final.get('stdout','') + '\n' + final.get('stderr','')
    frames=[]
    for filename, line in re.findall(r'([\w./\\-]+\.py):(\d+)',output):
        normalized=filename.replace('\\','/')
        matches=[path for path in graph.paths if normalized.endswith(path)]
        if matches:
            path=max(matches,key=len)
            frames.append({'file':path,'line':int(line)})
    contexts=[]
    for frame in frames[:3]:
        try: contexts.append(graph.slice(frame['file'],line=frame['line'],token_budget=max(256,token_budget//3)))
        except (KeyError,ValueError): pass
    category = ('cheat' if feedback.get('cheat_detected') else 'replay' if feedback.get('repeated_patch') else
                'timeout' if final.get('exit_code')==124 else 'assertion' if 'AssertionError' in output or 'assert ' in output else 'execution')
    return {'kind':'FailureDiagnosis','attempt':attempt,'category':category,'frames':frames[:10],
            'assertions':[line for line in output.splitlines() if 'assert ' in line or 'Error' in line][-12:],
            'contexts':contexts,'evidence_excerpt':output[-4000:],
            'next_action':'change the failed causal assumption; preserve frozen tests; do not repeat the rejected patch'}
