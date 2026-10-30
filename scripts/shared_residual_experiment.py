"""Shared constants, provenance and native helpers for candidate v1.48."""
import ctypes,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/pattern-shared-residual-v1.48'
SUFFIX='dylib' if sys.platform=='darwin' else 'so'
LIB=ROOT/f'.build/native-pattern-shared-residual-v1.48/libothello.{SUFFIX}'
PORTABLE=ROOT/f'.build/native-pattern-shared-residual-portable-v1.48/libothello.{SUFFIX}'
REFERENCE=ROOT/f'.build/native-v1.37-reference/libothello.{SUFFIX}'
BASEMODEL=ROOT/'models/pattern32-symmetry-v1.37.otpat'
PROTOCOL=ROOT/'models/v1.48-shared-residual-protocol.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
ptr=ctypes.POINTER(ctypes.c_int)


def preserved():
    return {str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'models/default-model.json',ROOT/'models/v1.37-default.json',BASEMODEL,REFERENCE,ROOT/'src/game/wasm/othello.mjs']}


def verify_preserved(protocol=None):
    for name,digest in (protocol or read(PROTOCOL))['preservedSha256'].items():assert sha(ROOT/name)==digest,name


def progress(stage,n=0,total=1,summary='',complete=False,validation=None):
    WORK.mkdir(parents=True,exist_ok=True)
    path=WORK/'progress.json';old=read(path) if path.exists() else {}
    result=dict(experiment='pattern-suite-sampling',stage=stage,completed=n,total=total,summary=summary,complete=complete,seconds=time.time()-read(PROTOCOL)['startedUnix'])
    for key in ['validation','validationEpoch','bestValidation']:
        if key in old:result[key]=old[key]
    if validation is not None:
        score=dict(validation,epoch=n);result.update(validation=score,validationEpoch=n)
        best=result.get('bestValidation')
        if best is None or score['mse']<best['mse']:result['bestValidation']=score
    write(path,result)


def monitor():
    config=read(ROOT/'models/current-experiment.json');items=config.get('experiments',[])
    archive=config.setdefault('archivedExperiments',[])
    for item in reversed(items):
        if item.get('focus')!='shared-residual' and not any(x.get('report')==item.get('report') for x in archive):archive.insert(0,item)
    config.update(version='v1.48 共享主干小残差 · 候选',experiments=[dict(name='共享残差：速度预检 → 训练 → 同时间对照',focus='shared-residual',games=0,runtimeReport=str((WORK/'progress.json').relative_to(ROOT)),report='models/v1.48-shared-residual-monitor.json')])
    write(ROOT/'models/current-experiment.json',config)


def load(path,model):
    lib=ctypes.CDLL(str(path));lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
    lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p];lib.ot_nnue_error.restype=ctypes.c_char_p
    lib.ot_pattern_predict.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
    lib.ot_nnue_mcts_play_budget.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*4+[ctypes.c_uint]*2+[ptr,ptr,ptr,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
    handle=lib.ot_nnue_load(str(model).encode());assert handle,lib.ot_nnue_error().decode()
    return lib,handle


def native(pair,boards,turns):
    import numpy as np
    lib,handle=pair;out=(ctypes.c_double*68)();rows=[]
    for board,turn in zip(boards,turns):
        cells=(ctypes.c_int*64)(*map(int,board));assert lib.ot_pattern_predict(handle,cells,int(turn),out)==1
        rows.append(list(out))
    return np.asarray(rows)


def search(pair,board,turn,ms=0,simulations=2048,clear=True):
    lib,handle=pair
    if clear:lib.ot_mcts_clear_cache()
    cells=(ctypes.c_int*64)(*map(int,board));out,sel,visits,stats=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)()
    tick=time.perf_counter_ns()
    assert lib.ot_nnue_mcts_play_budget(handle,cells,int(turn),ms,simulations,1832,12,32,64,1,0,4194304,out,sel,visits,stats,None)==1,lib.ot_nnue_error().decode()
    wall=(time.perf_counter_ns()-tick)/1e9
    sample=dict(simulations=int(stats[0]),evaluations=int(stats[1]),exact=bool(out[5]),memoryLimited=bool(stats[7]),elapsedMs=out[3],wallSeconds=wall)
    out[3]=0;stats[5]=0
    return [list(out),list(sel),list(visits),list(stats)],sample
