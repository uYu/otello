#!/usr/bin/env python3
"""Audit the full global policy experiment and the paired component exports."""
import hashlib,importlib.util,json,sys,tarfile
from pathlib import Path
import numpy as np
import torch
from pattern_model import encode,parity
from pattern_global_policy import GlobalPolicyNet
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'models/v1.49-pattern-global-policy-full.json'
AUDIT=ROOT/'models/v1.49-global-policy-full-validation.json'
SUMMARY=ROOT/'models/v1.49-global-policy-full-comparison.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def immutable(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists():assert path.read_text()==text,path
    else:path.write_text(text)


def main():
    r=read(REPORT);assert r['complete'] and not r['defaultPromoted'] and not r['freezeBaseline']
    assert r['trainableParameters']==916987 and r['encoderAndTrunkChanged']
    assert not r['pretrainedWeightsLoaded'] and r['sharedInitialParametersIdentical']
    best=min(r['history'],key=lambda x:x['policyCrossEntropy'])
    assert r['bestEpoch']==best['epoch'] and len(r['history'])==15
    protocol=ROOT/'models/v1.49-global-policy-full-protocol.json';assert sha(protocol)==r['protocolSha256']
    manifest=ROOT/r['sourceArchiveManifest'];m=read(manifest);assert sha(manifest)==r['sourceArchiveManifestSha256']
    assert sha(ROOT/m['archive'])==m['sha256']
    with tarfile.open(ROOT/m['archive'],'r:gz') as archive:
        for name,digest in m['files'].items():assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest,name
    for name,digest in r['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    for name,key in [('model','modelSha256'),('baseComponent','baseComponentSha256'),('headExport','headExportSha256')]:assert sha(ROOT/r[name])==r[key],name
    head=read(ROOT/r['headExport']);assert head['baseModel']==r['baseComponent'] and head['baseModelSha256']==r['baseComponentSha256']
    torch.set_num_threads(2);net=GlobalPolicyNet(1,freeze_baseline=False)
    net.load_state_dict(torch.load(ROOT/r['model'],weights_only=True));net.eval()
    data=np.load(ROOT/'.build/training/v1/positions.npz');suffix='dylib' if sys.platform=='darwin' else 'so'
    library=ROOT/f'.build/native-v1.37-reference/libothello.{suffix}'
    base_checks={}
    with torch.no_grad():
        for split in ['validation','test']:
            outputs=[[],[],[]]
            for start in range(0,len(data[split+'_board']),2048):
                board=data[split+'_board'][start:start+2048];turn=data[split+'_turn'][start:start+2048]
                codes=torch.from_numpy(encode(board,turn).astype(np.int64));phase=torch.from_numpy((64-np.count_nonzero(board,axis=1)).astype(np.float32)/60)
                value,wdl,projection,base=net.frozen_features(codes,phase)
                for rows,x in zip(outputs,[value,wdl,base]):rows.append(x.numpy())
            expected=tuple(np.concatenate(rows) for rows in outputs)
            check=parity(library,ROOT/r['baseComponent'],data[split+'_board'],data[split+'_turn'],expected)
            check['maxOutcomeError']=check.pop('maxDiscError');base_checks[split]=check
    verification=dict(complete=True,defaultPromoted=False,wholeNetworkTrained=True,trainableParameters=916987,selectedOn='validation legal-policy CE',
        baseComponentTorchNative=base_checks,globalHeadIntegerReference=r['integerHeadReference'],d4=r['symmetry'],
        fullModelNativeImplemented=False,componentPairVerified=True,nativeModelIsNotCompleteWithoutGlobalHead=True,
        reportSha256=sha(REPORT),sourceArchiveVerified=True,preservedSha256=r['preservedSha256'])
    immutable(AUDIT,verification)
    b=r['baselineMetrics'];c=r['metrics']
    summary=dict(version=r['version'],complete=True,status='candidate',defaultPromoted=False,training='All916987 parameters trained from random initialization; original common initialization/data order/augmentation; additional237 D4 policy coefficients',
        architecture='Ordered38 shared conditional pattern projections -> shared38->16->64 residual logits -> all8x8 move scores',
        selection=dict(metric='validation legal-policy cross entropy',bestEpoch=r['bestEpoch']),metrics=c,baselineMetrics=b,
        differences={split:{key:c[split][key]-b[split][key] for key in ['policyCrossEntropy','policyTop1','mse']} for split in ['validation','test']},
        nativePerformance='Not measured; no integrated native global policy kernel yet',matchesRun=False,
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in [REPORT,AUDIT,ROOT/r['model'],ROOT/r['baseComponent'],ROOT/r['headExport'],manifest,ROOT/m['archive']]},
        interpretation='Offline policy metrics and value metrics are reported separately. A single run and different validation selection from v1.37 do not establish strength or isolate architecture alone. The exported base component is not the complete policy model.')
    immutable(SUMMARY,summary)
    print(json.dumps(dict(complete=True,bestEpoch=r['bestEpoch'],differences=summary['differences'],baseComponentNativeChecks=base_checks),ensure_ascii=False))


if __name__=='__main__':main()
