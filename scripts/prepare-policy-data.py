#!/usr/bin/env python3
"""Join recorded next moves onto the frozen v1 positions, without changing splits."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    if args.output.exists():
        raise ValueError('Choose a fresh policy cache output')
    data = dict(np.load(args.cache))
    meta = json.loads(args.cache.with_suffix('.json').read_text())
    if sha(args.cache) != meta['cacheSha256'] or sha(args.games) != meta['inputSha256']:
        raise ValueError('Frozen cache or games checksum mismatch')
    lookup, labels, masks = {}, {}, {}
    for split in ('train', 'validation', 'test'):
        boards, turns = data[split+'_board'], data[split+'_turn']
        lookup[split] = {b.tobytes()+bytes([int(t)&255]):i for i,(b,t) in enumerate(zip(boards,turns))}
        labels[split] = np.full(len(boards), -1, dtype=np.int16)
        masks[split] = np.zeros((len(boards),65), dtype=np.bool_)
    lib = ctypes.CDLL(str(args.library.resolve()))
    ptr = ctypes.POINTER(ctypes.c_int)
    lib.ot_play.argtypes = [ptr, ctypes.c_int, ctypes.c_int, ptr]
    lib.ot_legal.argtypes = [ptr, ctypes.c_int, ptr]
    cell_type, out_type = ctypes.c_int*64, ctypes.c_int*136
    with args.games.open() as stream:
        for number,line in enumerate(stream,1):
            game = json.loads(line)
            if not game['complete']:
                continue
            split = game['split']
            if not lookup[split]:
                continue
            cells=cell_type();cells[28]=cells[35]=1;cells[27]=cells[36]=-1
            turn,ply=1,0
            for move in game['moves']:
                if 8<=ply<=56 and ply%3==0 and move>=0:
                    key=np.array(cells,dtype=np.int8).tobytes()+bytes([turn&255])
                    index=lookup[split].get(key)
                    if index is not None and np.float32(game['blackResult']/64*turn)==data[split+'_target'][index]:
                        legal=out_type();n=lib.ot_legal(cells,turn,legal)
                        if move not in legal[:n]:
                            raise ValueError('Illegal policy target')
                        labels[split][index]=move;masks[split][index,legal[:n]]=True
                        del lookup[split][key]
                output=out_type()
                if lib.ot_play(cells,turn,move,output)!=1:
                    raise ValueError('Illegal source game: '+game['id'])
                cells=cell_type(*output[:64]);turn=output[64];ply+=move>=0
            if number%5000==0:
                print(json.dumps(dict(games=number,remaining={s:len(v) for s,v in lookup.items()})),flush=True)
    remaining={s:len(v) for s,v in lookup.items()}
    if any(remaining.values()):
        raise ValueError('Missing next-move labels: '+str(remaining))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output, **{s+'_'+k:v for s in labels for k,v in [('move',labels[s]),('legal',masks[s])]})
    report=dict(cacheSha256=sha(args.output),positionsSha256=sha(args.cache),gamesSha256=sha(args.games),
        positions={s:len(v) for s,v in labels.items()},label='first same-split, same-board/turn and same-outcome recorded next move',
        split='unchanged frozen opening-group splits and symmetry-deduplicated positions',
        passPolicy='existing cache excludes pass targets; forced pass is deterministic at inference',
        scriptSha256=sha(Path(__file__)))
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('.build/training/v1/positions.npz'))
    p.add_argument('--games',type=Path,default=Path('.build/training/games.jsonl'))
    p.add_argument('--output',type=Path,default=Path('.build/training/policy/labels.npz'))
    p.add_argument('--library',type=Path,default=Path('.build/native/libothello.'+('dylib' if sys.platform=='darwin' else 'so')))
    prepare(p.parse_args())
