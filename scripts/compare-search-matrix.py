#!/usr/bin/env python3
"""Round robin: v1.3 NNUE / ResNet32x2, each with alpha-beta / uniform PUCT."""
import argparse
from collections import Counter, defaultdict
import ctypes
import hashlib
import importlib.util
import itertools
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import opening_policy
spec = importlib.util.spec_from_file_location('matrix_rules', ROOT / 'scripts/compare-resnet.py')
rules = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rules
spec.loader.exec_module(rules)
CONFIGS = [dict(id='nnue_ab', name='v1.3 + AB', search='ab', model='models/nnue256-v1.2.otnnue'),
           dict(id='cnn_ab', name='CNN32x2 + AB', search='ab', model='models/resnet32x2-v1.5.otres'),
           dict(id='nnue_mcts', name='v1.3 + MCTS', search='mcts', model='models/nnue256-v1.2.otnnue'),
           dict(id='cnn_mcts', name='CNN32x2 + MCTS', search='mcts', model='models/resnet32x2-v1.5.otres')]
BY_ID = {c['id']: c for c in CONFIGS}
POLICY = dict(plies=12, temperature=32, max_loss=64)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(connection, library, config):
    handle = None
    try:
        lib = ctypes.CDLL(library)
        ptr = ctypes.POINTER(ctypes.c_int)
        lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
        lib.ot_nnue_load.restype = ctypes.c_void_p
        lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]
        lib.ot_nnue_error.restype = ctypes.c_char_p
        lib.ot_nnue_play.argtypes = [ctypes.c_void_p, ptr] + [ctypes.c_int]*4 + [ctypes.c_uint] + [ctypes.c_int]*3 + [ptr, ptr]
        lib.ot_nnue_mcts_play.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint] + [ctypes.c_int]*3 + [ptr, ptr, ptr, ctypes.POINTER(ctypes.c_double)]
        handle = lib.ot_nnue_load(str((ROOT / config['model']).resolve()).encode())
        if not handle:
            raise RuntimeError(lib.ot_nnue_error().decode())
        connection.send(dict(ready=True))
        while True:
            command = connection.recv()
            if command['kind'] == 'quit':
                break
            if command['kind'] == 'clear':
                lib.ot_search_clear_cache(); lib.ot_mcts_clear_cache()
                connection.send(dict(cleared=True)); continue
            board = (ctypes.c_int*64)(*command['board'])
            output, policy, visits, stats = (ctypes.c_int*136)(), (ctypes.c_int*6)(), (ctypes.c_int*64)(), (ctypes.c_double*72)()
            seed = command['seed']; tick = time.perf_counter_ns()
            if config['search'] == 'mcts':
                success = lib.ot_nnue_mcts_play(handle, board, command['turn'], command['ms'], 0, seed,
                                               POLICY['plies'], POLICY['temperature'], POLICY['max_loss'], output, policy, visits, stats)
            else:
                success = lib.ot_nnue_play(handle, board, command['turn'], command['ms'], 64, 2, seed,
                                          POLICY['plies'], POLICY['temperature'], POLICY['max_loss'], output, policy)
            wall = (time.perf_counter_ns() - tick)/1e6
            if success != 1:
                raise RuntimeError(lib.ot_nnue_error().decode())
            candidates = [dict(move=output[7+2*i], score=output[8+2*i]) for i in range(output[6])]
            if config['search'] == 'mcts':
                for i, c in enumerate(candidates):
                    c.update(visits=visits[i], value=stats[8+i])
            connection.send(dict(move=output[0], score=output[1], depth=output[2], elapsedMs=output[3],
                                 nodes=output[4], exact=bool(output[5]), ttHits=output[135], wallMs=wall,
                                 candidates=candidates, search=config['search'],
                                 mcts=dict(simulations=int(stats[0]), evaluations=int(stats[1]), meanDepth=stats[2]/1000,
                                           reusedNodes=int(stats[3]), treeNodes=int(stats[4]), memoryLimited=bool(stats[7])) if config['search']=='mcts' else None,
                                 selection=dict(seed=seed, eligible=policy[0], temperatureMilli=policy[1], lossLimit=policy[2],
                                                bestScore=policy[3], scoreLoss=policy[4], sampled=bool(policy[5]))))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        connection.send(dict(error=str(error)))
    finally:
        if handle:
            lib.ot_nnue_destroy(handle)
        connection.close()


class Engine(rules.Engine):
    def __init__(self, library, config):
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker, args=(child,str(library.resolve()),config))
        self.process.start(); child.close(); self.receive()


def tie(board, turn, candidate, seed):
    return opening_policy.splitmix(opening_policy.board_hash(board,turn) ^ (seed << 32) ^ (candidate['move']+1))[1]


def audit_decision(board, turn, decision, policy=POLICY):
    candidates = decision['candidates']; move = decision['move']; selection = decision['selection']
    assert move in rules.legal(board, turn)
    if not candidates: # timed AB may have only its legal fallback
        assert decision['depth'] == 0 and not selection['sampled']
        return
    assert len(candidates)==len({c['move'] for c in candidates})
    assert {c['move'] for c in candidates}==set(rules.legal(board,turn))
    placed = sum(bool(c) for c in board)-4
    if decision['search']=='mcts':
        assert all(c['visits']>=0 and -1 <= c['value'] <= 1 for c in candidates)
        if decision['exact']:
            base=max(candidates,key=lambda c:(c['value'],tie(board,turn,c,selection['seed'])))
        else:
            base=max(candidates,key=lambda c:(c['visits'],c['value'],tie(board,turn,c,selection['seed'])))
        if placed>=policy['plies'] or decision['exact'] or decision['depth']<1:
            assert move==base['move'] and decision['score']==base['score']
            assert selection==dict(seed=selection['seed'],eligible=1,temperatureMilli=0,lossLimit=0,bestScore=base['score'],scoreLoss=0,sampled=False)
            return
        candidates=[base]+[c for c in candidates if c['move']!=base['move']]
    if placed < policy['plies'] and decision['depth']>0 and not decision['exact']:
        chosen, score, meta=opening_policy.choose(board,turn,candidates,selection['seed'],**policy,exact=decision['exact'])
        assert chosen==move and score==decision['score'] and meta==selection
    else:
        assert not selection['sampled'] and selection['temperatureMilli']==0 and selection['scoreLoss']==0
        assert any(c['move']==move and c['score']==decision['score'] for c in candidates)


def replay(record):
    board, turn=rules.initial(),1
    steps=iter(record['decisions'])
    for move in record['moves']:
        if move>=0:
            d=next(steps);assert d['turn']==turn and d['move']==move
            assert d['engine']==(record['black'] if turn==1 else record['white'])
            assert d['selection']['seed']==record['seed']
            audit_decision(board,turn,d,record['openingPolicy'])
        board,turn=rules.play(board,turn,move)
    assert next(steps,None) is None and not rules.legal(board,turn) and not rules.legal(board,-turn)
    assert record['finalBoard']==board and record['blackDiscDifference']==sum(board)
    assert record['winner']==(record['black'] if sum(board)>0 else record['white'] if sum(board)<0 else None)
    return len(record['moves'])


def aggregate(records):
    totals={c['id']:Counter(games=0,win=0,draw=0,loss=0,discDifference=0) for c in CONFIGS}
    matrix={c['id']:{o['id']:dict(win=0,draw=0,loss=0) for o in CONFIGS if o!=c} for c in CONFIGS}
    stats=defaultdict(Counter)
    for game in records:
        for who,other,sign in [(game['black'],game['white'],1),(game['white'],game['black'],-1)]:
            outcome='draw' if game['winner'] is None else 'win' if game['winner']==who else 'loss'
            totals[who].update(games=1,discDifference=sign*game['blackDiscDifference']);totals[who][outcome]+=1;matrix[who][other][outcome]+=1
        for d in game['decisions']:
            s=stats[d['engine']];s.update(moves=1,nodes=d['nodes'],wallMs=d['wallMs'],depth=d['depth'],exactMoves=d['exact'],
                                        nonExactMoves=not d['exact'],nonExactDepth=d['depth'] if not d['exact'] else 0)
            if d['mcts']:
                m=d['mcts'];s.update(simulations=m['simulations'],evaluations=m['evaluations'],
                                    simulationDepth=m['meanDepth']*m['simulations'],reusedSearches=m['reusedNodes']>0,memoryLimited=m['memoryLimited'])
    output={}
    for c in CONFIGS:
        name=c['id'];v=totals[name];n=v['games'];s=stats[name];moves=s['moves']
        output[name]=dict(v,scoreRate=(v['win']+v['draw']/2)/n if n else 0,meanDiscDifference=v['discDifference']/n if n else 0,
                          moves=moves,meanWallMs=s['wallMs']/moves if moves else 0,
                          meanSearchUnits=s['nodes']/moves if moves else 0,searchUnit='nodes' if c['search']=='ab' else 'simulations',
                          meanNonExactDepth=s['nonExactDepth']/s['nonExactMoves'] if s['nonExactMoves'] else 0,
                          depthKind='completed' if c['search']=='ab' else 'maximumVisited',
                          meanSimulationDepth=s['simulationDepth']/s['simulations'] if s['simulations'] else None,
                          memoryLimitedSearches=s['memoryLimited'],reusedMctsSearches=s['reusedSearches'],exactMoves=s['exactMoves'])
    return output,matrix


def compare(args):
    if args.output.exists():
        raise ValueError('Choose a fresh output; reports are never overwritten')
    configs=[dict(c,modelSha256=sha(ROOT/c['model'])) for c in CONFIGS]
    provenance=dict(version='v1.6',experiment='search-matrix',millisecondsPerMove=args.milliseconds,
                    gamesPerPair=args.games_per_pair,plannedGames=6*args.games_per_pair,baseSeed=args.seed,
                    start='initial',opening=[],useBook=False,openingPolicy=POLICY,configs=configs,
                    mcts=dict(algorithm='uniform-prior PUCT',exploration=1.5,maxNodes=2097152,rootChoice='maximum visits, then mean value, then seeded tie',
                              value='side-to-move expected final disc difference / 64; terminal actual disc difference / 64',
                              rollout=False,policyHead=False,subtreeReuse=True),
                    librarySha256=sha(args.library),runnerSha256=sha(Path(__file__)),
                    rulesSha256=sha(ROOT/'scripts/compare-resnet.py'),policyAuditSha256=sha(ROOT/'scripts/opening_policy.py'),
                    sourceSha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'cpp').rglob('*')) if p.suffix in ('.cpp','.hpp','.h')},
                    platform=platform.platform(),cachePolicy='Clear both caches at each game; retain TT / reached MCTS subtree between own moves',
                    limits='Small paired-color screen from standard initial board. Repeated paths possible; no Elo or reliable win-rate claim. AB completed depth and MCTS visited depth/search units differ.')
    started=time.perf_counter();records=[];current=None;engines={}
    def save(complete=False,full=False):
        totals,matrix=aggregate(records)
        value=dict(**provenance,complete=complete,games=len(records),seconds=time.perf_counter()-started,
                   results=totals,matrix=matrix,records=records,replayedGames=len(records),replayedActions=sum(len(g['moves']) for g in records),
                   uniqueTranscripts=len({tuple(g['moves']) for g in records}))
        if full: rules.write_json(args.output,value)
        value.pop('records');value['currentGame']=current;rules.write_json(args.output.with_suffix('.live.json'),value)
    try:
        for c in CONFIGS:engines[c['id']]=Engine(args.library,c)
        save(full=True)
        for pair,(left,right) in enumerate(itertools.combinations(CONFIGS,2)):
            for round_index in range(args.games_per_pair):
                black,white=(left,right) if round_index%2==0 else (right,left)
                for c in (black,white):engines[c['id']].request(kind='clear')
                seed=(args.seed^((round_index//2+1)*0x9e3779b9))&0x7fffffff
                board,turn=rules.initial(),1; moves=[];decisions=[]
                current=dict(number=len(records)+1,pair=pair,black=black['id'],white=white['id'],ply=0)
                while True:
                    available=rules.legal(board,turn)
                    if not available:
                        if not rules.legal(board,-turn):break
                        move=-1
                    else:
                        name=black['id'] if turn==1 else white['id']
                        r=engines[name].request(kind='search',board=board,turn=turn,seed=seed,ms=args.milliseconds)
                        decision=dict(ply=len(moves),turn=turn,engine=name,**r);audit_decision(board,turn,decision)
                        decisions.append(decision);move=r['move']
                    board,turn=rules.play(board,turn,move);moves.append(move)
                    assert len(moves)<=120
                    current.update(ply=len(moves),blackCount=board.count(1),whiteCount=board.count(-1));save()
                difference=sum(board)
                record=dict(number=len(records)+1,pair=pair,round=round_index,black=black['id'],white=white['id'],seed=seed,
                            opening=[],openingPolicy=POLICY,moves=moves,decisions=decisions,finalBoard=board,
                            blackDiscDifference=difference,winner=black['id'] if difference>0 else white['id'] if difference<0 else None)
                replay(record);records.append(record);current=None;save(full=True)
                print(json.dumps(dict(completed=len(records),black=record['black'],white=record['white'],difference=difference,winner=record['winner'])),flush=True)
        assert sha(args.library)==provenance['librarySha256']
        assert all(sha(ROOT/c['model'])==c['modelSha256'] for c in configs)
        save(complete=True,full=True)
    finally:
        for e in engines.values():e.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    suffix='dylib' if sys.platform=='darwin' else 'so'
    p.add_argument('--library',type=Path,default=ROOT/f'.build/native-v1.6/libothello.{suffix}')
    p.add_argument('--output',type=Path,default=ROOT/'models/v1.6-search-matrix.json')
    p.add_argument('--milliseconds',type=int,default=1000)
    p.add_argument('--games-per-pair',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261027)
    a=p.parse_args()
    if a.games_per_pair<2 or a.games_per_pair%2 or not 1<=a.milliseconds<=60000:p.error('Even games per pair >=2 and milliseconds 1..60000 required')
    compare(a)
