"""Native export and runtime helpers, separate from immutable v1.49 training."""
import ctypes,hashlib,json,struct,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/pattern-global-policy-native-v1.49'
SUFFIX='dylib' if sys.platform=='darwin' else 'so'
LIB=ROOT/f'.build/native-pattern-global-policy-v1.49/libothello.{SUFFIX}'
PORTABLE=ROOT/f'.build/native-pattern-global-policy-portable-v1.49/libothello.{SUFFIX}'
REFERENCE=ROOT/f'.build/native-v1.37-reference/libothello.{SUFFIX}'
BASE=ROOT/'models/pattern32-symmetry-v1.37.otpat'
MODEL=ROOT/'models/pattern32-global-policy16-full-v1.49.otpat'
PROTOCOL=ROOT/'models/v1.49-global-policy-native-protocol.json'
TRAINING=ROOT/'models/v1.49-pattern-global-policy-full.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
ptr=ctypes.POINTER(ctypes.c_int)


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def progress(stage,n=0,total=1,summary='',complete=False):
    WORK.mkdir(parents=True,exist_ok=True)
    write(WORK/'progress.json',dict(experiment='pattern-suite-sampling',stage=stage,completed=n,total=total,summary=summary,complete=complete,seconds=time.time()-read(PROTOCOL)['startedUnix']))


def verify_preserved():
    for name,digest in read(PROTOCOL)['preservedSha256'].items():assert sha(ROOT/name)==digest,name


def export():
    import numpy as np
    training=read(TRAINING);assert training['complete']
    component=ROOT/training['baseComponent'];headfile=ROOT/training['headExport'];head=read(headfile)
    assert sha(component)==training['baseComponentSha256']==head['baseModelSha256']
    assert sha(headfile)==training['headExportSha256']
    base=component.read_bytes();assert base[:8]==b'OTPAT4\0\0' and struct.unpack_from('<I',base,32)[0]==1
    binary=b'OTPAT9\0\0'+base[8:36]+struct.pack('<3I',38,16,64)+base[36:]
    for layer,shape in zip(head['layers'],[(16,38),(64,16)]):
        weights=np.asarray(layer['weight'],dtype=np.int64);bias=np.asarray(layer['bias'],dtype=np.int64)
        assert weights.shape==shape and bias.shape==(shape[0],)
        assert weights.min()>=-32768 and weights.max()<=32767 and np.abs(bias).max()<=10000000
        binary+=weights.astype('<i2').tobytes()+bias.astype('<i4').tobytes()
    if MODEL.exists():assert MODEL.read_bytes()==binary
    else:MODEL.write_bytes(binary)
    return binary


def load(library,model):
    lib=ctypes.CDLL(str(library));lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
    lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p];lib.ot_nnue_error.restype=ctypes.c_char_p
    lib.ot_pattern_predict.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
    lib.ot_nnue_mcts_play_budget.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*4+[ctypes.c_uint]*2+[ptr,ptr,ptr,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
    handle=lib.ot_nnue_load(str(model).encode());assert handle,lib.ot_nnue_error().decode()
    return lib,handle


def native(pair,boards,turns):
    import numpy as np
    lib,handle=pair;out=(ctypes.c_double*68)();rows=[]
    for board,turn in zip(boards,turns):
        cells=(ctypes.c_int*64)(*map(int,board));assert lib.ot_pattern_predict(handle,cells,int(turn),out)==1,lib.ot_nnue_error().decode()
        rows.append(list(out))
    return np.asarray(rows)


def search(pair,board,turn,ms=0,simulations=2048,clear=True):
    lib,handle=pair
    if clear:lib.ot_mcts_clear_cache()
    cells=(ctypes.c_int*64)(*map(int,board));out,selection,visits,stats=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)()
    tick=time.perf_counter_ns()
    assert lib.ot_nnue_mcts_play_budget(handle,cells,int(turn),ms,simulations,1832,12,32,64,1,0,4194304,out,selection,visits,stats,None)==1
    wall=(time.perf_counter_ns()-tick)/1e9
    meta=dict(simulations=int(stats[0]),evaluations=int(stats[1]),exact=bool(out[5]),memoryLimited=bool(stats[7]),elapsedMs=out[3],wallSeconds=wall)
    out[3]=0;stats[5]=0
    return [list(out),list(selection),list(visits),list(stats)],meta
