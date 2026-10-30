#!/usr/bin/env python3
"""Verify immutable candidate-era evidence and replay both promoted-model matches."""
import hashlib,importlib.util,json,sys,tarfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def archived(manifest):
    m=read(manifest);assert sha(ROOT/m['archive'])==m['sha256'],manifest
    with tarfile.open(ROOT/m['archive'],'r:gz') as t:
        assert len(t.getmembers())==len(m['files'])
        for name,digest in m['files'].items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==digest,name
    return m

def main():
    saved=archived(ROOT/'models/archive/v2-promotion-preserved.json')
    # All candidate-era models, protocols, reports and earlier baseline records stay immutable.
    for name,digest in saved['files'].items():
        if name.startswith('models/') and name not in ('models/default-model.json','models/README.md'):assert sha(ROOT/name)==digest,name
    for name in ['v1.49-global-policy-native-sources','v1.49-v1.9-200ms-50-sources']:
        m=archived(ROOT/f'models/archive/{name}.json')
        for dependency,digest in m.get('engineSourceManifests',{}).items():
            assert sha(ROOT/dependency)==digest;archived(ROOT/dependency)
    with tempfile.TemporaryDirectory(prefix='otello-v2-history-') as folder:
        root=Path(folder);scripts=root/'scripts';scripts.mkdir()
        with tarfile.open(ROOT/saved['archive'],'r:gz') as t:
            for name in ['compare-nnue-candidate.py','compare-nnue-search.py','compare-resnet.py','opening_policy.py']:
                (scripts/name).write_bytes(t.extractfile('scripts/'+name).read())
        sys.path.insert(0,str(scripts))
        spec=importlib.util.spec_from_file_location('v2_archived_match_auditor',scripts/'compare-nnue-candidate.py')
        comparator=importlib.util.module_from_spec(spec);sys.modules[spec.name]=comparator;spec.loader.exec_module(comparator)
        match=comparator.match;total=0
        for report,summary,protocol,n,ms,wdl in [
            ('v1.49-global-policy-vs-v1.37-2s-20','v1.49-global-policy-2s-20-comparison','v1.49-global-policy-2s-20-protocol',20,2000,(10,6,4)),
            ('v1.49-global-policy-vs-v1.9-200ms-50','v1.49-v1.9-200ms-50-comparison','v1.49-v1.9-200ms-50-protocol',50,200,(48,1,1))]:
            path=ROOT/f'models/{report}.json';r=read(path);s=read(ROOT/f'models/{summary}.json');p=read(ROOT/f'models/{protocol}.json')
            assert r['complete'] and r['games']==len(r['records'])==n and r['millisecondsPerMove']==ms and not r['useBook']
            assert sha(path)==s['reportSha256'] and r['protocolSha256']==sha(ROOT/f'models/{protocol}.json')
            assert r['sourceArchiveManifestSha256']==sha(ROOT/r['sourceArchiveManifest'])
            for name,digest in p['preservedSha256'].items():
                if name in ('models/default-model.json','src/game/wasm/othello.mjs'):assert saved['files'][name]==digest
                elif not name.startswith('.build/'):assert sha(ROOT/name)==digest,name
            match.CONFIGS=r['configs'];actions=0
            for i,game in enumerate(r['records']):
                assert game['number']==i+1
                assert (game['black'],game['white'])==(('baseline','candidate') if i%2==0 else ('candidate','baseline'))
                assert game['seed']==(20261029^((i//2+1)*0x9e3779b9))&0x7fffffff
                actions+=match.replay(game)
                for d in game['decisions']:
                    expected=2097152 if n==50 and d['engine']=='baseline' else 4194304
                    assert d['valueTarget']=='outcome' and d['mcts']['maxNodes']==expected
            results,matrix=match.aggregate(r['records'])
            assert results==r['results'] and matrix==r['matrix'] and actions==r['replayedActions']==s['replayedActions']
            c=results['candidate'];assert (c['win'],c['draw'],c['loss'])==wdl
            assert s['candidateScoreRate']==c['scoreRate'] and s['candidateMeanDiscDifference']==c['meanDiscDifference']
            total+=actions;print(f'{report}: {n} games, {actions} actions, every recorded decision audited')
    print(f'v2 candidate-era evidence preserved; {total} actions replayed')

if __name__=='__main__':main()
