#!/usr/bin/env python3
"""Preflight FP32 memory and training throughput for the large CNN on MPS."""
import gc,importlib.util,json,math,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('big_training',ROOT/'scripts/train-big-cnn.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    assert torch.backends.mps.is_available()
    rows=[]
    for batch,amp in ((128,False),(256,False),(256,True),(512,True)):
        net=m.BigCnn().to('mps').train();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
        board=torch.randint(-1,2,(batch,64),device='mps',dtype=torch.int8);turn=torch.ones(batch,device='mps',dtype=torch.int8)
        target=torch.randint(-1,2,(batch,),device='mps').float();moves=torch.randint(65,(batch,),device='mps',dtype=torch.long)
        scaler=torch.amp.GradScaler('mps',init_scale=1024,enabled=amp)
        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='mps',dtype=torch.float16,enabled=amp):
                v,p=net(board,turn);loss=4*(v.float()-target).square().mean()+torch.nn.functional.cross_entropy(p,moves)
            scaler.scale(loss).backward();scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=not amp)
            scaler.step(opt);scaler.update()
            assert torch.isfinite(loss)
            return float(loss.detach().cpu())
        for _ in range(2):step()
        torch.mps.synchronize();start=time.perf_counter()
        losses=[step() for _ in range(4)];torch.mps.synchronize();seconds=(time.perf_counter()-start)/4
        r=dict(batchSize=batch,amp=amp,scale=scaler.get_scale(),stepSeconds=seconds,positionsPerSecond=batch/seconds,
            estimatedTrainEpochSeconds=math.ceil(1000000/batch)*seconds,
            allocatedBytes=torch.mps.current_allocated_memory(),driverAllocatedBytes=torch.mps.driver_allocated_memory(),
            recommendedMaxBytes=torch.mps.recommended_max_memory(),parameters=sum(p.numel() for p in net.parameters()),losses=losses)
        rows.append(r);print(json.dumps(r),flush=True)
        del net,opt,scaler,board,turn,target,moves;gc.collect();torch.mps.empty_cache()
    output=ROOT/'.build/training/big-cnn-v1.24-checks';output.mkdir(exist_ok=True)
    chosen=max(rows,key=lambda r:r['positionsPerSecond'])
    m.h.write_json(output/'report.json',dict(complete=True,precision='FP32 and AMP-FP16',device='mps',rows=rows,recommendedBatchSize=chosen['batchSize'],recommendedAmp=chosen['amp'],
        caveat='Short synthetic warmup and4 measured steps; epoch estimate excludes validation/checkpoints and thermal variation'))
if __name__=='__main__':main()
