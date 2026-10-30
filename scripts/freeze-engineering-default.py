#!/usr/bin/env python3
"""Record the user-approved v1.36 default and freeze its delivery sources."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import sys

ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'models/v1.36-default.json'
SOURCE_MANIFEST='models/archive/v1.36-default-sources.json'
ARCHIVE='models/archive/v1.36-default-sources.tar.gz'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def record(validated=False):
    active=read(ROOT/'models/default-model.json')
    assert active['version']=='v1.36' and active['baselineManifest']=='models/v1.36-default.json'
    model=ROOT/active['defaultModel'];metadata=read(model.with_suffix('.json'))
    assert sha(model)==active['modelSha256']==metadata['sha256']
    assert model.read_bytes()[:8]==b'OTPAT2\0\0'
    experiments=[]
    for path,report in [('models/v1.35-pattern-speed-suite-experiment.json','models/v1.35-speed-suite-vs-v1.25-2s-20.json'),
                        ('models/v1.36-pattern-capacity-experiment.json','models/v1.36-capacity-vs-v1.25-2s-20.json')]:
        data=read(ROOT/report)
        assert data['complete'] and data['games']==20
        assert data['configs'][1]['modelSha256']==active['modelSha256']
        experiments.append(dict(path=path,sha256=sha(ROOT/path),report=report,reportSha256=sha(ROOT/report),
            millisecondsPerMove=data['millisecondsPerMove'],games=data['games'],candidateResults=data['results']['candidate']))
    paths=set()
    for directory in ['cpp','src']:
        paths.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.cpp','.hpp','.h','.inc','.md','.ts','.tsx','.mts','.css'))
    paths.update(ROOT/name for name in ['AGENTS.md','CMakeLists.txt','README.md','Dockerfile','compose.yaml','compose.prebuilt.yaml',
        'package.json','package-lock.json','requirements-training.txt','index.html','tsconfig.json','vite.config.ts',
        'docs/TRAINING.md','docs/DEPLOYMENT.md','models/README.md','models/default-model.json','models/training-data-manifest.json',
        'models/archive/v1.36-candidate-era-verification.json','data/README.md','data/manifest.json',
        'scripts/build-wasm.sh','scripts/embed-default-model.py','scripts/check-web-release.mjs','scripts/prepare-web-release.mjs',
        'scripts/prepare-web-book.py','scripts/data-assets.py','scripts/verify-baseline.py','scripts/freeze-engineering-default.py',
        'scripts/candidate_era_artifacts.py','scripts/finalize-pattern-outcome.py','scripts/finalize-pattern-outcome-2s.py',
        'scripts/train-pattern-outcome.py','scripts/train-pattern.py','scripts/pattern_model.py',
        'scripts/compare-nnue-candidate.py','scripts/compare-light-cnn-candidate.py','scripts/compare-big-cnn-candidate.py','scripts/compare-nnue-search.py','scripts/compare-resnet.py','scripts/opening_policy.py',
        'scripts/watch-matches.py','scripts/watch-match.sh','scripts/test-native.sh','tools/export-web-book.cpp',
        'tests/native.cpp','tests/default-engine.test.ts','tests/opening-book.test.ts','tests/pattern_training_test.py'])
    artifacts=[ROOT/'src/game/wasm/othello.mjs',model,model.with_suffix('.json')]
    paths.update(p for p in (ROOT/'benchmarks').rglob('*.cpp'))
    paths.update(ROOT/name for name in ['scripts/verify-engineering-history.py','cpp/pattern_kernels.hpp','scripts/run-pattern-capacity.py','scripts/run-pattern-speed-suite.py','scripts/finalize-pattern-capacity.py','scripts/finalize-pattern-speed-suite.py'])
    paths.difference_update(artifacts)
    release=dict(version='v1.36',status='frozen',date='2026-10-30',userApproved=True,
        approval='User explicitly requested: 工程提升，持平就行，作为基线，提交代码',previousBaseline='models/v1.25-default.json',
        defaultModel=active['defaultModel'],modelSha256=active['modelSha256'],modelFormat='OTPAT2',
        parameters=metadata['parameters'],bytes=model.stat().st_size,architecture=metadata['architecture'].replace('disc/WDL','outcome/WDL'),
        valueTarget='side-to-move recorded terminal outcome: win +1, draw 0, loss -1',
        policyTarget='recorded next legal move; masked cross entropy',
        search='Pattern learned-prior PUCT MCTS with finite first-play value',
        mcts=dict(exploration=1.5,maxNodes=4194304,unvisited='finite FPU: parent mean minus 0.15',
            rootChoice='maximum visits, then mean value, then seeded ties',subtreeReuse=True,
            terminalValue='sign(final side-to-move disc difference)',
            proofStopping='a proven losing child proves a win; otherwise all children must be solved',
            exactRootChoice='proven optimal children, seeded ties; auxiliary WDL head unused'),
        openingPolicy=dict(plies=12,temperature=32,maxLoss=64),
        browser=dict(normalMilliseconds=100,hardMilliseconds=1000,easy='heuristic',offlineEmbedded=True,
            persistentWorker=True,openingBook='mandatory verified Egaroucid book before every move; approved search on verified miss'),
        native=dict(default='embedded Pattern policy/outcome + MCTS',evaluator='othello-pattern32-outcome-v2',
            explicitAb='--search ab',weak='--network heuristic'),referenceExperiments=experiments,
        sourceManifest=SOURCE_MANIFEST,sourceSha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},
        artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts},
        nativeLibrarySha256=sha(ROOT/('.build/native/libothello.'+('dylib' if sys.platform=='darwin' else 'so'))),
        checks=dict(native=True,sanitizers=True,patternTests=4,webTests=19,frontendBuild=True,
            wasmRebuilt=True,embeddedModelParity=True,historicalReportsPreserved=True,candidateProtocol=True) if validated else {},
        engineering=dict(simd=True,fastCodes=True,block4=True,legalPolicy=True,integerRounding=False,patternCacheEntries=0),
        notes=['User explicitly approves engineering parity as the new baseline; weights unchanged.',
            '2M suite match: 5W/8D/7L; 4M capacity match: 7W/7D/6L, both 2s/20 games.',
            'Controlled sampling: 2.158x simulations/s and 1.868x evaluations/s versus v1.25.',
            '4M candidate: 0/603 capped decisions; peak RSS 2493.9 MiB on macOS ARM64.',
            'No statistically significant strength improvement claimed; Linux performance unmeasured.',
            'Previous baselines and all candidate-era records remain immutable.'])
    write(BASELINE,release)
    if validated:
        archive=ROOT/ARCHIVE
        assert not archive.exists(),'Refusing to replace a frozen default archive'
        all_paths=sorted(paths|set(artifacts))
        with tarfile.open(archive,'w:gz') as stream:
            for p in all_paths:stream.add(p,arcname=str(p.relative_to(ROOT)))
        write(ROOT/SOURCE_MANIFEST,dict(status='frozen-default',version='v1.36',baseline=str(BASELINE.relative_to(ROOT)),
            baselineSha256=sha(BASELINE),archive=ARCHIVE,sha256=sha(archive),
            files={**release['sourceSha256'],**release['artifactSha256']}))
        print('Frozen user-approved v1.36 default, native/Wasm hashes and separate measured reports')
    else:print('Prepared v1.36 release metadata; checks will be recorded after validation')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validated',action='store_true')
    args=parser.parse_args()
    assert not (ROOT/ARCHIVE).exists(),'Default is already frozen; use verify-baseline.py'
    record(args.validated)
