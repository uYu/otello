#!/usr/bin/env python3
"""Freeze the exact user-approved global-policy weights and v2 delivery sources."""
import argparse,hashlib,json,shutil,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
BASELINE=ROOT/'models/v2-default.json'
ARCHIVE=ROOT/'models/archive/v2-default-sources.tar.gz'
MANIFEST=ROOT/'models/archive/v2-default-sources.json'

def record(validated=False):
    assert not ARCHIVE.exists(),'v2 already frozen; create a new version for further changes'
    active=read(ROOT/'models/default-model.json');assert active['version']=='v2'
    model=ROOT/active['defaultModel'];assert model.read_bytes()[:8]==b'OTPAT9\0\0'
    assert sha(model)==active['modelSha256']==sha(ROOT/'models/pattern32-global-policy16-full-v1.49.otpat')
    suffix='dylib' if sys.platform=='darwin' else 'so'
    native=ROOT/f'.build/native/libothello.{suffix}'
    reference=ROOT/f'.build/native-v2-reference/libothello.{suffix}'
    reference.parent.mkdir(parents=True,exist_ok=True)
    if reference.exists():assert sha(reference)==sha(native)
    else:shutil.copy2(native,reference)
    metadata=read(model.with_suffix('.json'))
    metadata.update(nativeLibrary=str(reference.relative_to(ROOT)),nativeLibrarySha256=sha(reference),nativeValidation='models/v2-default-validation.json',frozenBaseline='models/v2-default.json',previousBaseline='models/v1.37-default.json')
    write(model.with_suffix('.json'),metadata)
    release=read(ROOT/'models/v1.37-default.json')
    release.update(version='v2',status='frozen',date='2026-10-30',userApproved=True,approval='感觉可以作为v2了',
        previousBaseline='models/v1.37-default.json',promotedCandidate='v1.49',defaultModel=active['defaultModel'],modelSha256=sha(model),
        modelFormat='OTPAT9',modelVariant=6,parameters=916987,bytes=model.stat().st_size,
        architecture='38 ternary line/corner patterns -> D4-tied 28,503 x 32 INT16 table -> three mean pools + empty/60 -> 97->32 trunk -> original conditional policy + once-per-board D4-tied 38->16->64 global correction',
        training=dict(report='models/v1.49-pattern-global-policy-full.json',reportSha256=sha(ROOT/'models/v1.49-pattern-global-policy-full.json'),allParametersTrainedFromScratch=True,selectedEpoch=15,selection='minimum validation legal-policy cross entropy',globalPolicyIndependentParameters=237),
        sourceManifest=str(MANIFEST.relative_to(ROOT)),referenceExperiments=[])
    release['native']['evaluator']='othello-pattern32-global-policy-outcome-v9'
    for label in ['v1.49-global-policy-2s-20','v1.49-v1.9-200ms-50']:
        protocol=ROOT/f'models/{label}-protocol.json';summary=ROOT/f'models/{label}-comparison.json';s=read(summary);report=ROOT/s['report'];r=read(report)
        assert r['complete'] and sha(report)==s['reportSha256']
        release['referenceExperiments'].append(dict(path=str(protocol.relative_to(ROOT)),sha256=sha(protocol),report=str(report.relative_to(ROOT)),reportSha256=sha(report),summary=str(summary.relative_to(ROOT)),summarySha256=sha(summary),baseline=s['baseline'],millisecondsPerMove=s['millisecondsPerMove'],games=s['games'],candidateResults=r['results']['candidate'],useBook=False))
    paths=set()
    for folder in ['cpp','src','scripts','tests','benchmarks']:
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix in ('.cpp','.hpp','.h','.inc','.md','.ts','.tsx','.mts','.css','.py','.mjs','.sh'))
    for name in ['AGENTS.md','CMakeLists.txt','README.md','Dockerfile','compose.yaml','compose.prebuilt.yaml','package.json','package-lock.json','requirements-training.txt','index.html','tsconfig.json','vite.config.ts','docs/TRAINING.md','docs/DEPLOYMENT.md','models/README.md','models/default-model.json','models/training-data-manifest.json','data/README.md','data/manifest.json','models/archive/v2-promotion-preserved.json','models/v1.49-global-policy-native-validation.json','tools/export-web-book.cpp']:
        paths.add(ROOT/name)
    artifacts={ROOT/'src/game/wasm/othello.mjs',model,model.with_suffix('.json'),model.with_suffix('.pt')}
    if validated:artifacts.add(ROOT/'models/v2-default-validation.json')
    paths.difference_update(artifacts)
    release.update(sourceSha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(artifacts)},nativeLibrarySha256=sha(native),referenceNativeLibrary=dict(path=str(reference.relative_to(ROOT)),sha256=sha(reference)),checks=read(ROOT/'models/v2-default-validation.json') if validated else {},notes=[
        'Exact complete v1.49 model promoted by explicit user approval; all 916,987 parameters were jointly trained from scratch. No new training or strength match.',
        '2s/20 vs frozen v1.37: 10W/6D/4L,65%,mean discs +7.5. Separate200ms/50 vs frozen v1.9: 48W/1D/1L,97%,mean discs +24.56.',
        'Different opponents and budgets; screening measurements are not Elo or significance claims. Candidate-era reports remain unchanged.',
        'Native and browser embed full OTPAT9 model with all64 global policy outputs. Mandatory verified Egaroucid book precedes browser search;100ms/1s budgets retained.',
        'All previous baselines remain frozen. Historical audit: python3 scripts/verify-v2-history.py. Future comparisons use v2.',
        'ARM64 measured; Linux throughput remains unmeasured. Earlier2s candidate process peak RSS1732.75MiB is a deployment capacity reference.'])
    write(BASELINE,release)
    if validated:
        with tarfile.open(ARCHIVE,'w:gz') as t:
            for p in sorted(paths|artifacts):t.add(p,arcname=str(p.relative_to(ROOT)))
        write(MANIFEST,dict(status='frozen-default',version='v2',baseline=str(BASELINE.relative_to(ROOT)),baselineSha256=sha(BASELINE),archive=str(ARCHIVE.relative_to(ROOT)),sha256=sha(ARCHIVE),files={**release['sourceSha256'],**release['artifactSha256']}))
    print('v2 default '+('frozen' if validated else 'prepared for validation'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--validated',action='store_true');record(p.parse_args().validated)
