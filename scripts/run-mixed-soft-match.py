#!/usr/bin/env python3
"""Compare mixed-soft symmetry weights with frozen v1.37 using identical native code."""
import argparse,hashlib,importlib.util,io,json,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.build/training/mixed-soft-match-v1.45'
SUFFIX='dylib' if sys.platform=='darwin' else 'so';LIB=ROOT/f'.build/native-v1.37-reference/libothello.{SUFFIX}'
MODEL=ROOT/'models/pattern32-symmetry-mixed-soft-v1.45.otpat';PROTOCOL=ROOT/'models/v1.45-soft-label-2s-20-protocol.json'
RUNTIME=WORK/'match-2s-20.json';REPORT=ROOT/'models/v1.45-soft-vs-v1.37-2s-20.json';SOURCE=ROOT/'models/archive/v1.45-soft-label-2s-20-sources.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,r:Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('wthor_match_helpers',ROOT/'scripts/run-pattern-structure.py');h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h);match=h.match


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');args=parser.parse_args();assert not REPORT.exists()
    WORK.mkdir(parents=True,exist_ok=True)
    active=read(ROOT/'models/default-model.json');baseline=read(ROOT/active['baselineManifest']);training=read(ROOT/'models/v1.45-mixed-label-ablation.json');variant=training['variants'][1]
    assert active['version']=='v1.37' and baseline['userApproved'] and baseline['status']=='frozen'
    assert variant['formatMagic']=='OTPAT4' and sha(MODEL)==variant['modelSha256'] and training['dataset']['sources']==['wthor','logistello']
    assert not training['dataset']['eloFiltering'] and not training['dataset']['eloWeighting']
    assert sha(LIB)==baseline['referenceNativeLibrary']['sha256']==training['librarySha256']
    original=read(ROOT/'models/v1.45-mixed-label-ablation-protocol.json')
    if not PROTOCOL.exists():
        assert not args.resume and not RUNTIME.exists()
        write(PROTOCOL,dict(version='v1.45-soft-label-2s-pilot',status='candidate',defaultPromoted=False,userRequest='soft和基线打一架噶；沿用2s/步20局预算',baselineManifest=active['baselineManifest'],baselineManifestSha256=sha(ROOT/active['baselineManifest']),candidate=str(MODEL.relative_to(ROOT)),candidateModelSha256=sha(MODEL),baselineModel=active['defaultModel'],baselineModelSha256=active['modelSha256'],games=20,millisecondsPerMove=2000,seed=20261029,start='initial',pairedColors=True,sharedPairSeeds=True,useBook=False,mcts=dict(maxNodes=4194304,patternCacheEntries=0,exploration=1.5,finiteFpuReduction=.15,subtreeReuse=True,terminal='outcome +1/0/-1'),openingPolicy=match.POLICY,librarySha256={str(LIB.relative_to(ROOT)):sha(LIB)},librarySourceManifest='models/archive/v1.37-default-sources.json',librarySourceManifestSha256=sha(ROOT/'models/archive/v1.37-default-sources.json'),trainingReportSha256=sha(ROOT/'models/v1.45-mixed-label-ablation.json'),preservedSha256=original['preservedSha256'],concurrentTraining=False,limits='Identical frozen native library for both independent engine processes; training finished before match. Small paired-color sample, no Elo/significance claim and no automatic promotion.'))
    protocol=read(PROTOCOL)
    for name,digest in {**protocol['preservedSha256'],**protocol['librarySha256']}.items():assert sha(ROOT/name)==digest,name
    frozen=read(ROOT/protocol['librarySourceManifest']);assert sha(ROOT/frozen['archive'])==frozen['sha256']
    with tarfile.open(ROOT/frozen['archive'],'r:gz') as tar:
        source={name:tar.extractfile(name).read() for name in frozen['files'] if name.startswith('cpp/') or name=='CMakeLists.txt'}
    for name,binary in source.items():assert hashlib.sha256(binary).hexdigest()==frozen['files'][name]
    for name in ['AGENTS.md','scripts/run-mixed-soft-match.py','scripts/finalize-mixed-soft-match.py','scripts/train-mixed-label-ablation.py','scripts/mixed_label_targets.py','scripts/wthor_label_ablation.py','scripts/pattern_model.py','scripts/pattern_structure.py','scripts/training_consensus.py','scripts/run-pattern-structure.py','scripts/compare-nnue-candidate.py','scripts/compare-nnue-search.py','scripts/compare-resnet.py','scripts/opening_policy.py','scripts/watch-match.sh','scripts/watch-matches.py','models/v1.45-soft-label-2s-20-protocol.json']:
        source[name]=(ROOT/name).read_bytes()
    archive=SOURCE.with_suffix('.tar.gz')
    hashes={name:hashlib.sha256(binary).hexdigest() for name,binary in source.items()}
    if SOURCE.exists():assert read(SOURCE)['files']==hashes and sha(archive)==read(SOURCE)['sha256']
    else:
        assert not archive.exists()
        with tarfile.open(archive,'w:gz') as tar:
            for name,binary in sorted(source.items()):
                info=tarfile.TarInfo(name);info.size=len(binary);tar.addfile(info,io.BytesIO(binary))
        write(SOURCE,dict(archive=str(archive.relative_to(ROOT)),sha256=sha(archive),files=hashes,compiledCppSource='Both engines use immutable v1.37 reference library; C++ bytes are taken from its frozen source archive. Python runtime scripts are captured from the active workspace.'))
    item=dict(name='混合语料 soft v1.45 vs 冻结 v1.37 · 2s/步 · 20局',focus='candidate',displayNames=dict(candidate='混合语料 soft',baseline='v1.37 对称共享'),displayOrder=['candidate','baseline'],games=20,runtimeReport=str(RUNTIME.relative_to(ROOT)),report=str(REPORT.relative_to(ROOT)))
    config=read(ROOT/'models/current-experiment.json');history=config.get('archivedExperiments',[])
    for old in reversed(config['experiments']):
        if old.get('report')!=item['report'] and not any(x.get('report')==old.get('report') for x in history):history.insert(0,old)
    config.update(version='v1.45 soft · 原百万混合语料对照',experiments=[item],archivedExperiments=history);write(ROOT/'models/current-experiment.json',config)
    match.worker=h.worker;match.Engine=h.Engine;original_write=match.rules.write_json
    def annotated(path,value):
        if Path(path).parent==WORK and 'configs' in value:
            value.update(protocolSha256=sha(PROTOCOL),sourceArchiveManifest=str(SOURCE.relative_to(ROOT)),maxNodesByEngine=dict(baseline=4194304,candidate=4194304),patternCacheEntries=0,librarySourceManifest=protocol['librarySourceManifest'],librarySourceManifestSha256=protocol['librarySourceManifestSha256'])
            if 'sourceSha256' in value:
                for name in value['sourceSha256']:
                    if name.startswith('cpp/') or name=='CMakeLists.txt':value['sourceSha256'][name]=hashes[name]
        return original_write(path,value)
    match.rules.write_json=annotated
    h.comparison.compare(argparse.Namespace(model=MODEL,baseline_manifest=ROOT/active['baselineManifest'],baseline_library=LIB,label='v1.45 mixed soft',library=LIB,output=RUNTIME,games=20,milliseconds=2000,seed=20261029,resume=args.resume))
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    write(REPORT,read(RUNTIME));print(json.dumps(dict(complete=True,report=str(REPORT.relative_to(ROOT)))),flush=True)


if __name__=='__main__':main()
