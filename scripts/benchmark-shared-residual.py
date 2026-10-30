#!/usr/bin/env python3
"""Speed gate: zero residual preserves the exact frozen baseline tree."""
import json,random,statistics,subprocess
import numpy as np
import torch
from pattern_shared_residual import SharedResidualNet,export
import shared_residual_experiment as e
REPORT=e.ROOT/'models/v1.48-shared-residual-speed-gate.json'


def main():
    assert not REPORT.exists(),'Reports are immutable'
    protocol=e.read(e.PROTOCOL);e.verify_preserved(protocol)
    state=torch.load(e.ROOT/'models/pattern32-symmetry-v1.37.pt',weights_only=True)
    models=[e.BASEMODEL]
    for width in [8,16]:
        torch.manual_seed(48148);net=SharedResidualNet(width);missing=net.load_state_dict(state,strict=False)
        assert set(missing.missing_keys)=={'residual_down.weight','residual_down.bias','residual_up.weight','residual_up.bias'} and not missing.unexpected_keys
        path=e.WORK/f'identity-r{width}.otpat';export(net,path);models.append(path)
    data=np.load(e.ROOT/'.build/training/v1/positions.npz');boards=np.concatenate([data['validation_board'],data['test_board']]);turns=np.concatenate([data['validation_turn'],data['test_turn']])
    handles=[e.load(e.LIB,p) for p in models]+[e.load(e.REFERENCE,e.BASEMODEL)]
    try:
        e.progress('速度预检：核验零残差与冻结基线完全一致')
        expected=e.native(handles[-1],boards,turns)
        for h in handles[:-1]:assert np.array_equal(e.native(h,boards,turns),expected)
        for i in range(16):
            results=[e.search(h,boards[i],turns[i])[0] for h in handles]
            assert all(x==results[0] for x in results)
        placed=np.count_nonzero(data['validation_board'],axis=1)-4
        indices=np.concatenate([np.flatnonzero((placed>=lo)&(placed<hi))[:128] for lo,hi in [(0,20),(20,44),(44,61)]])
        boardfile=e.WORK/'timing-boards.txt';boardfile.write_text(''.join(''.join('B' if x==1 else 'W' if x==-1 else '.' for x in data['validation_board'][i])+f' {int(data["validation_turn"][i])}\n' for i in indices))
        rng=random.Random(48148);timings=[];samples=[]
        bench=e.LIB.parent/'otello_shared_residual_benchmark'
        for round_ in range(9):
            for k in rng.sample(range(3),3):
                e.progress('速度预检：真实合法着路径',len(timings),27,str(models[k].name))
                row=json.loads(subprocess.check_output([str(bench),str(models[k]),str(boardfile),'65536','1'],text=True))
                timings.append(dict(model=k,round=round_,**row))
        roots=[]
        for lo,hi in [(8,16),(17,24),(25,32),(33,40)]:roots.extend(np.flatnonzero((placed>=lo)&(placed<=hi))[:2].tolist())
        for round_ in range(3):
            for i in roots:
                outputs=[]
                for k in rng.sample(range(3),3):
                    e.progress('速度预检：相同树、相同131072模拟',len(samples),72,str(models[k].name))
                    result,sample=e.search(handles[k],data['validation_board'][i],data['validation_turn'][i],simulations=131072)
                    outputs.append(result);samples.append(dict(model=k,root=i,round=round_,**sample))
                assert all(x==outputs[0] for x in outputs)
        summary=[];baseline_seconds=sum(s['wallSeconds'] for s in samples if s['model']==0)
        for k,width in enumerate([0,8,16]):
            phases=[statistics.median(row['groups'][g]['nanoseconds'] for row in timings if row['model']==k) for g in range(3)]
            seconds=sum(s['wallSeconds'] for s in samples if s['model']==k)
            summary.append(dict(residualWidth=width,model=str(models[k].relative_to(e.ROOT)),medianLegalNanosecondsByPhase=phases,fixedWorkSeconds=seconds,evaluationThroughputRatio=baseline_seconds/seconds))
        for x in summary:
            x['maximumPhaseLatencyRatio']=max(v/b for v,b in zip(x['medianLegalNanosecondsByPhase'],summary[0]['medianLegalNanosecondsByPhase']))
            x['passes']=x['residualWidth']>0 and x['evaluationThroughputRatio']>=protocol['speedGate']['minimumEvaluationThroughputRatio'] and x['maximumPhaseLatencyRatio']<=protocol['speedGate']['maximumLegalLatencyRatio']
        selected=next((w for w in [16,8] if next(x for x in summary if x['residualWidth']==w)['passes']),None)
        e.verify_preserved(protocol)
        report=dict(version='v1.48-speed-gate',complete=True,defaultPromoted=False,selectedResidualWidth=selected,summary=summary,timingSamples=timings,fixedWorkSamples=samples,
            parity=dict(positions=12000,all68OutputsExactlyEqual=True,frozenReferenceAndIdentityWidths=[8,16],fixedRoots=16,fixedSimulations=2048,speedSamplesHaveIdenticalSearchResults=True),
            protocolSha256=e.sha(e.PROTOCOL),librarySha256=e.sha(e.LIB),scriptSha256=e.sha(__file__),preservedSha256=protocol['preservedSha256'],limits='Native single-machine speed gate with identical weights/outputs/search trees. No training or strength inference. Nonexclusive CPU; raw interleaved samples and phase latencies retained.')
        e.write(REPORT,report);e.progress('速度预检完成',1,1,f'选定残差宽度 {selected}',selected is None)
        print(json.dumps(dict(selectedWidth=selected,summary=summary),ensure_ascii=False))
    finally:
        for lib,handle in handles:lib.ot_nnue_destroy(handle)


if __name__=='__main__':main()
