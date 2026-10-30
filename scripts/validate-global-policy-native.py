#!/usr/bin/env python3
"""Validate the full native policy, legacy regression and search parity."""
import ctypes,importlib.util,json,struct,subprocess,tempfile
import numpy as np
import torch
from pattern_global_policy import GlobalPolicyNet
from pattern_model import parity
import global_policy_native_experiment as e
REPORT=e.ROOT/'models/v1.49-global-policy-native-validation.json'
spec=importlib.util.spec_from_file_location('global_native_prediction',e.ROOT/'scripts/train-pattern-outcome.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def main():
    assert not REPORT.exists(),'Validation reports are immutable'
    e.verify_preserved();training=e.read(e.TRAINING);e.export();torch.set_num_threads(2)
    net=GlobalPolicyNet(1,freeze_baseline=False);net.load_state_dict(torch.load(e.ROOT/training['model'],weights_only=True));net.eval()
    data=np.load(e.ROOT/'.build/training/v1/positions.npz');checks={}
    for split in ['validation','test']:
        e.progress('完整模型 Torch / C++ 三头导出核验',0,12000,split)
        predicted=h.predict(net,data[split+'_board'],data[split+'_turn'])
        check=parity(e.LIB,e.MODEL,data[split+'_board'],data[split+'_turn'],predicted);check['maxOutcomeError']=check.pop('maxDiscError');checks[split]=check
    pairs=[e.load(e.LIB,e.MODEL),e.load(e.PORTABLE,e.MODEL),e.load(e.LIB,e.BASE),e.load(e.REFERENCE,e.BASE)]
    try:
        for split in ['validation','test']:
            board,turn=data[split+'_board'],data[split+'_turn']
            assert np.array_equal(e.native(pairs[0],board,turn),e.native(pairs[1],board,turn))
            assert np.array_equal(e.native(pairs[2],board,turn),e.native(pairs[3],board,turn))
        for i in range(16):
            for a,b in [(0,1),(2,3)]:
                first,_=e.search(pairs[a],data['validation_board'][i],data['validation_turn'][i]);second,_=e.search(pairs[b],data['validation_board'][i],data['validation_turn'][i]);assert first==second,(i,a,b)
        with tempfile.TemporaryDirectory() as folder:
            folder=e.ROOT/'.build/training/pattern-global-policy-native-v1.49/test-models';folder.mkdir(exist_ok=True)
            binary=e.MODEL.read_bytes();zero=folder/'zero-correction.otpat';zero.write_bytes(binary[:-2304]+bytes(2304))
            control=[e.load(e.LIB,zero),e.load(e.REFERENCE,e.ROOT/training['baseComponent'])]
            try:
                for split in ['validation','test']:
                    assert np.array_equal(*[e.native(pair,data[split+'_board'],data[split+'_turn']) for pair in control])
                for i in range(8):assert e.search(control[0],data['validation_board'][i],data['validation_turn'][i])[0]==e.search(control[1],data['validation_board'][i],data['validation_turn'][i])[0]
            finally:
                for lib,handle in control:lib.ot_nnue_destroy(handle)
            lib=pairs[0][0];damaged=bytearray(binary);struct.pack_into('<I',damaged,40,15)
            for i,content in enumerate([binary[:-1],binary+b'X',damaged]):
                path=folder/f'bad-{i}.otpat';path.write_bytes(content);assert not lib.ot_nnue_load(str(path).encode())
            changed=bytearray(binary);struct.pack_into('<i',changed,len(changed)-256,struct.unpack_from('<i',changed,len(changed)-256)[0]+65536)
            path=folder/'changed.otpat';path.write_bytes(changed);other=e.load(e.LIB,path)
            try:
                board=np.zeros(64,np.int8);board[[28,35]]=1;board[[27,36]]=-1
                e.search(pairs[0],board,1,simulations=128)
                reused,_=e.search(pairs[0],board,1,simulations=128,clear=False)
                invalidated,_=e.search(other,board,1,simulations=128,clear=False)
                assert reused[3][3]>0 and invalidated[3][3]==0
            finally:other[0].ot_nnue_destroy(other[1])
        benchmark=json.loads(subprocess.check_output([str(e.LIB.parent/'otello_pattern_legal_benchmark'),str(e.MODEL),'1024'],text=True))
        e.verify_preserved()
        result=dict(version='v1.49-global-policy-native',complete=True,defaultPromoted=False,format='OTPAT9',model=str(e.MODEL.relative_to(e.ROOT)),modelSha256=e.sha(e.MODEL),bytes=e.MODEL.stat().st_size,
            trainingReportSha256=e.sha(e.TRAINING),torchNative=checks,scalarSimd=dict(positions=12000,all68OutputsExactlyEqual=True,fixedSearchRoots=16,simulations=2048),
            legacyRegression=dict(positions=12000,all68OutputsExactlyEqual=True,fixedSearchRoots=16,simulations=2048),
            zeroCorrection=dict(positions=12000,all68OutputsExactlyEqual=True,fixedSearchRoots=8),
            malformedModelsRejected=3,residualParameterCacheInvalidationPassed=True,nativeCTestBuilds=['SIMD','portable'],legalProjectionCheck=benchmark,
            libraries={str(p.relative_to(e.ROOT)):e.sha(p) for p in [e.LIB,e.PORTABLE,e.REFERENCE]},
            protocolSha256=e.sha(e.PROTOCOL),preservedSha256=e.read(e.PROTOCOL)['preservedSha256'],limits='Frozen selected weights, no retraining. Native macOS ARM64 correctness; no strength result in this record.')
        e.write(REPORT,result);e.progress('完整原生模型核验完成',1,1,'准备启动2s/20局',True)
        print(json.dumps(result,ensure_ascii=False))
    finally:
        for lib,handle in pairs:lib.ot_nnue_destroy(handle)


if __name__=='__main__':main()
