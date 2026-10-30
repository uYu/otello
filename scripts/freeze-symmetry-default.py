#!/usr/bin/env python3
"""Freeze the exact user-approved v1.37 symmetry model and delivery sources."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'models/v1.37-default.json'
ARCHIVE=ROOT/'models/archive/v1.37-default-sources.tar.gz'
MANIFEST=ROOT/'models/archive/v1.37-default-sources.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def record(validated=False):
    assert not ARCHIVE.exists(),'Baseline already frozen'
    active=read(ROOT/'models/default-model.json');assert active['version']=='v1.37'
    model=ROOT/active['defaultModel'];metadata=read(model.with_suffix('.json'))
    assert model.name=='pattern32-symmetry-v1.37.otpat' and model.read_bytes()[:8]==b'OTPAT4\0\0'
    assert sha(model)==active['modelSha256']==metadata['sha256']
    release=read(ROOT/'models/v1.36-default.json')
    release.update(version='v1.37',status='frozen',date='2026-10-30',userApproved=True,
        approval='User explicitly requested: 把这个作为基线：对称共享',previousBaseline='models/v1.36-default.json',
        defaultModel=active['defaultModel'],modelSha256=active['modelSha256'],modelFormat='OTPAT4',modelVariant=1,
        parameters=916750,bytes=1833692,
        architecture='38 ternary line/corner patterns -> D4-tied 28,503 x 32 INT16 table -> edge/interior/corner means + empty/60 -> 97->32 clipped ReLU -> outcome/WDL/original globally conditioned per-square policy',
        sourceManifest=str(MANIFEST.relative_to(ROOT)),referenceExperiments=[])
    release['native']['evaluator']='othello-pattern32-symmetry-outcome-v4'
    for ms in [100,2000]:
        path=ROOT/f'models/v1.37-symmetry-vs-v1.36-{ms}ms-20.json';r=read(path)
        assert r['complete'] and r['games']==20 and r['configs'][1]['modelSha256']==active['modelSha256']
        release['referenceExperiments'].append(dict(path='models/v1.37-pattern-structure-protocol.json',sha256=sha(ROOT/'models/v1.37-pattern-structure-protocol.json'),
            report=str(path.relative_to(ROOT)),reportSha256=sha(path),millisecondsPerMove=ms,games=20,candidateResults=r['results']['candidate']))
    paths=set()
    for directory in ['cpp','src','benchmarks']:
        paths.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.cpp','.hpp','.h','.inc','.md','.ts','.tsx','.mts','.css'))
    names=['AGENTS.md','CMakeLists.txt','README.md','Dockerfile','compose.yaml','compose.prebuilt.yaml','package.json','package-lock.json',
        'requirements-training.txt','index.html','tsconfig.json','vite.config.ts','docs/TRAINING.md','docs/DEPLOYMENT.md','models/README.md',
        'models/default-model.json','models/training-data-manifest.json','data/README.md','data/manifest.json',
        'models/archive/v1.37-candidate-era-verification.json','models/archive/v1.37-promotion-preserved.json',
        'scripts/build-wasm.sh','scripts/embed-default-model.py','scripts/check-web-release.mjs','scripts/prepare-web-release.mjs','scripts/prepare-web-book.py',
        'scripts/data-assets.py','scripts/verify-baseline.py','scripts/freeze-symmetry-default.py','scripts/verify-pattern-structure-history.py',
        'scripts/verify-engineering-history.py','scripts/candidate_era_artifacts.py','scripts/finalize-pattern-structure.py','scripts/pattern_structure.py',
        'scripts/train-pattern-structure.py','scripts/benchmark-pattern-structure.py','scripts/run-pattern-structure.py','scripts/pattern_model.py',
        'scripts/train-pattern-outcome.py','scripts/train-pattern.py','scripts/compare-nnue-candidate.py','scripts/compare-nnue-search.py','scripts/compare-resnet.py',
        'scripts/opening_policy.py','scripts/watch-match.sh','scripts/watch-matches.py','scripts/test-native.sh','scripts/verify-symmetry-default.py',
        'scripts/verify-symmetry-default.mjs','tools/export-web-book.cpp','tests/native.cpp','tests/default-engine.test.ts','tests/opening-book.test.ts',
        'tests/pattern_training_test.py','tests/pattern_structure_test.py']
    paths.update(ROOT/name for name in names)
    artifacts={ROOT/'src/game/wasm/othello.mjs',model,model.with_suffix('.json'),model.with_suffix('.pt')}
    if validated:artifacts.add(ROOT/'models/v1.37-default-validation.json')
    paths.difference_update(artifacts)
    release.update(sourceSha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},
        artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(artifacts)},
        nativeLibrarySha256=sha(ROOT/f'.build/native/libothello.{"dylib" if sys.platform=="darwin" else "so"}'),
        referenceNativeLibrary=dict(path=f'.build/native-v1.37-reference/libothello.{"dylib" if sys.platform=="darwin" else "so"}',
            sha256=sha(ROOT/f'.build/native-v1.37-reference/libothello.{"dylib" if sys.platform=="darwin" else "so"}')),
        checks=read(ROOT/'models/v1.37-default-validation.json') if validated else {},
        notes=['User-approved promotion of the exact measured symmetry weights; no retraining or new strength match.',
            '100ms/20 games: 9W/5D/6L (57.5%); 2s/20 games: 10W/7D/3L (67.5%). Budgets stay separate.',
            'Single training seed and small paired-color samples; no statistically significant strength claim.',
            '38 patterns, three pools and original conditional policy head retained; ten-pool and direct MLP prototypes excluded.',
            'All earlier baselines, model bytes and candidate-era reports remain frozen.',
            'Historical audit: python3 scripts/verify-pattern-structure-history.py.',
            'Mandatory verified Egaroucid book and 100ms/1s browser budgets retained. Linux throughput unmeasured.'])
    write(BASELINE,release)
    if validated:
        with tarfile.open(ARCHIVE,'w:gz') as t:
            for p in sorted(paths|artifacts):t.add(p,arcname=str(p.relative_to(ROOT)))
        write(MANIFEST,dict(status='frozen-default',version='v1.37',baseline=str(BASELINE.relative_to(ROOT)),baselineSha256=sha(BASELINE),
            archive=str(ARCHIVE.relative_to(ROOT)),sha256=sha(ARCHIVE),files={**release['sourceSha256'],**release['artifactSha256']}))
    print('v1.37 default '+('frozen' if validated else 'prepared for validation'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validated',action='store_true');record(parser.parse_args().validated)
