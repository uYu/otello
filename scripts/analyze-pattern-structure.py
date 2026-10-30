import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
import pattern_model as p
spec=importlib.util.spec_from_file_location('pattern_analysis_training',root/'scripts/train-pattern-outcome.py');training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
torch.set_num_threads(2)
model=root/'models/pattern32-outcome-v1.25.pt';net=p.PatternNet();net.load_state_dict(torch.load(model,map_location='cpu',weights_only=True));net.eval()
data=np.load(root/'.build/training/v1/positions.npz');labels=np.load(root/'.build/training/pattern-outcome-v1.25/policy.npz')
b,t,y=data['validation_board'],data['validation_turn'],np.sign(data['validation_target']);truth=labels['validation'];mask=p.legal_mask(b,t)
started=time.perf_counter();grid=np.arange(64).reshape(8,8);values=[];policies=[]
with torch.no_grad():
 for mirror in [False,True]:
  for rotation in range(4):
   perm=np.rot90(np.fliplr(grid) if mirror else grid,rotation).reshape(64)
   value,wdl,logits=training.predict(net,b[:,perm],t)
   aligned=logits[:,np.argsort(perm)]
   logits=np.where(mask,aligned,-1e9);probs=torch.softmax(torch.from_numpy(logits),1).numpy()
   values.append(value);policies.append(probs)
base=values[0];prob=policies[0];meanvalue=np.mean(values,axis=0);meanprob=np.mean(policies,axis=0)
def metrics(v,pr):return dict(outcomeMse=float(np.mean((v-y)**2)),nonDrawSignAccuracy=float(np.mean(np.sign(v[y!=0])==y[y!=0])),policyTop1=float(np.mean(pr.argmax(1)==truth)),policyCrossEntropy=float(-np.log(np.maximum(1e-12,pr[np.arange(len(b)),truth])).mean()))
symmetry=dict(valueSpanMean=float(np.ptp(values,axis=0).mean()),valueSpanMax=float(np.ptp(values,axis=0).max()),valueSpanAbove01=float(np.mean(np.ptp(values,axis=0)>.1)),meanTopMoveDisagreement=float(np.mean([np.mean(pr.argmax(1)!=prob.argmax(1)) for pr in policies[1:]])),baselineValidation=metrics(base,prob),eightSymmetryAverageValidation=metrics(meanvalue,meanprob),note='Averaging is a diagnostic at 8x inference cost, not a deployed candidate or a strength result')
with torch.no_grad():
 code=torch.from_numpy(p.encode(b,t).astype(np.int64));e=p.quant(net.embedding(code));pooled=torch.einsum('bpd,pg->bgd',e,net.pool)/net.counts[None,:,None]
 phase=torch.from_numpy((64-np.count_nonzero(b,axis=1)).astype(np.float32)/60)
 x=torch.cat((p.quant(pooled.flatten(1).clamp(-4,4)),p.quant(phase[:,None])),1)
 h=p.quant(F.linear(x,p.quant(net.trunk.weight),p.quant(net.trunk.bias,65536)).clamp(0,1)).numpy()
activation=dict(zeroFraction=float(np.mean(h==0)),saturatedFraction=float(np.mean(h==1)),alwaysZeroNeurons=int(np.all(h==0,axis=0).sum()),perNeuronActiveFraction=np.mean(h>0,axis=0).tolist())
placed=np.count_nonzero(b,axis=1)-4;stages=[]
for lo,hi in [(8,16),(17,32),(33,48),(49,56)]:
 use=(placed>=lo)&(placed<=hi)
 stages.append(dict(placed=[lo,hi],positions=int(use.sum()),outcomeMse=float(np.mean((base[use]-y[use])**2)),policyTop1=float(np.mean(prob[use].argmax(1)==truth[use])),policyCrossEntropy=float(-np.log(np.maximum(1e-12,prob[np.flatnonzero(use),truth[use]])).mean())))
# D4 maps a normalized inward corner to itself or its transpose; each line to itself or reversal.
canonical=np.arange(p.ROWS);types=[];offset=0
for name,n,count in [('line8',8,4)]+[(f'diagonal{n}',n,1) for n in range(4,9)]+[('corner3x3',9,1)]:
 size=3**n;ids=np.arange(size);digits=(ids[:,None]//(3**np.arange(n)))%3
 changed=digits.reshape(-1,3,3).transpose(0,2,1).reshape(-1,9) if n==9 else digits[:,::-1]
 other=changed@(3**np.arange(n));orbit=np.minimum(ids,other);unique=np.unique(orbit)
 types.append(dict(name=name,tables=count,rowsEach=size,canonicalRowsEach=len(unique)))
 for table in range(count):canonical[offset:offset+size]=offset+orbit;offset+=size
assert offset==p.ROWS
counts=np.zeros(p.ROWS,np.int64);train=data['train_board'];turn=data['train_turn']
for first in range(0,len(train),16384):
 codes=p.encode(train[first:first+16384],turn[first:first+16384]);counts+=np.bincount(codes.reshape(-1).astype(np.int64),minlength=p.ROWS)
canonical_ids,inverse=np.unique(canonical,return_inverse=True);orbitcounts=np.bincount(inverse,weights=counts,minlength=len(canonical_ids))
coverage=dict(rawRowsSeen=int(np.count_nonzero(counts)),rawRowsUnder10=int(np.sum(counts<10)),rawRowsUnder100=int(np.sum(counts<100)),canonicalRows=len(canonical_ids),canonicalRowsSeen=int(np.count_nonzero(orbitcounts)),rawRowsCoveredByFullD4Closure=int(np.count_nonzero(orbitcounts[inverse])),note='Counts use frozen raw training boards; orbit coverage accounts for symmetry closure, actual random augmentation multiplicities are not reconstructed')
embedding=p.ROWS*p.DIM;total=sum(v.numel() for v in net.parameters());head=p.HIDDEN*p.OUTPUTS+p.OUTPUTS
parameters=dict(total=total,embedding=embedding,embeddingFraction=embedding/total,trunk=p.INPUTS*p.HIDDEN+p.HIDDEN,head=head,modelBytes=(root/'models/pattern32-outcome-v1.25.otpat').stat().st_size,hidden64Total=embedding+p.INPUTS*64+64+64*p.OUTPUTS+p.OUTPUTS,hidden64Extra=64*(p.INPUTS+1+p.OUTPUTS)+p.OUTPUTS-(p.INPUTS*p.HIDDEN+p.HIDDEN+head),canonicalModelBytes=32+2*(len(canonical_ids)*32+p.HIDDEN*p.INPUTS+p.OUTPUTS*p.HIDDEN)+4*(p.HIDDEN+p.OUTPUTS))
spatial=np.column_stack((p.MEMBERS.T,np.eye(10)[p.BUCKET]));rank=int(np.linalg.matrix_rank(spatial,tol=1e-6));legal_ranks=[int(np.linalg.matrix_rank(spatial[row],tol=1e-6)) for row in mask]
policy=dict(spatialBasisRank=rank,legalRankDeficitPositions=int(sum(rank<int(row.sum()) for rank,row in zip(legal_ranks,mask))),validationPositions=len(mask),note='Fixed additive shape projection plus 10 position buckets; rank deficit counts are on legal moves only')
report=dict(scriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),baselineManifestSha256=hashlib.sha256((root/'models/v1.25-default.json').read_bytes()).hexdigest(),analysis='Pattern v1.25 architecture diagnostics',baseline='models/v1.25-default.json',modelSha256=hashlib.sha256((root/'models/pattern32-outcome-v1.25.otpat').read_bytes()).hexdigest(),validationOnly=True,testSetUsed=False,parameters=parameters,symmetry=symmetry,activation=activation,stages=stages,coverage=coverage,canonicalTypes=types,policy=policy,seconds=time.perf_counter()-started)
out=root/'models/pattern-v1.25-architecture-analysis.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
