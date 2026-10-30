#!/usr/bin/env python3
"""Compare candidate weights with the approved baseline using identical MCTS."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('nnue_match', ROOT / 'scripts/compare-nnue-search.py')
match = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = match
spec.loader.exec_module(match)
BASE_AUDIT=match.audit_decision


def audit_decision(board,turn,decision,policy=match.POLICY):
    # Outcome search can prove a win before the other children are solved.
    if decision.get('valueTarget')=='outcome' and decision['exact'] and decision['score']==100000:
        candidates=decision['candidates'];selection=decision['selection']
        assert {c['move'] for c in candidates}==set(match.rules.legal(board,turn))
        assert len(candidates)==len({c['move'] for c in candidates})
        assert all(c['visits']>=0 and -1<=c['value']<=1 for c in candidates)
        proven=[c for c in candidates if c['score']==100000]
        assert proven and all(c['value']==1 for c in proven)
        chosen=max(proven,key=lambda c:match.tie(board,turn,c,selection['seed']))
        assert decision['move']==chosen['move']
        assert selection==dict(seed=selection['seed'],eligible=1,temperatureMilli=0,lossLimit=0,bestScore=100000,scoreLoss=0,sampled=False)
        return
    BASE_AUDIT(board,turn,decision,policy)


match.audit_decision=audit_decision


def compare(args):
    resume=getattr(args,'resume',False)
    if args.output.exists() and not resume:
        raise ValueError('Choose a fresh output; reports are never overwritten')
    if resume and not args.output.exists():
        raise ValueError('Resume report does not exist')
    baseline = json.loads(args.baseline_manifest.read_text())
    magic=args.model.read_bytes()[:8]
    baseline_magic=(ROOT/baseline['defaultModel']).read_bytes()[:8]
    learned=magic in (b'OTNNUE3\0',b'OTNNUE4\0',b'OTNNUE5\0',b'OTCNN2\0\0',b'OTPAT1\0\0',b'OTPAT2\0\0',b'OTPAT3\0\0')
    baseline_learned=baseline_magic in (b'OTNNUE3\0',b'OTNNUE4\0',b'OTNNUE5\0',b'OTCNN2\0\0',b'OTPAT1\0\0',b'OTPAT2\0\0',b'OTPAT3\0\0')
    configs = [dict(id='baseline', name=baseline['version'], search='mcts', model=baseline['defaultModel'],prior='learned' if baseline_learned else 'uniform',valueTarget='outcome' if baseline_magic in (b'OTNNUE4\0',b'OTNNUE5\0',b'OTCNN2\0\0',b'OTPAT2\0\0',b'OTPAT3\0\0') else 'discDifference'),
               dict(id='candidate', name=args.label, search='mcts', model=str(args.model.resolve()),prior='learned' if learned else 'uniform',valueTarget='outcome' if magic in (b'OTNNUE4\0',b'OTNNUE5\0',b'OTCNN2\0\0',b'OTPAT2\0\0',b'OTPAT3\0\0') else 'discDifference')]
    assert match.sha(ROOT / configs[0]['model']) == baseline['modelSha256']
    match.CONFIGS = configs
    provenance = dict(version=args.label, experiment='candidate-vs-approved-baseline',
        baselineVersion=baseline['version'], baselineManifest=str(args.baseline_manifest),
        baselineManifestSha256=match.sha(args.baseline_manifest),
        configs=[dict(c, modelSha256=match.sha(ROOT / c['model'])) for c in configs],
        millisecondsPerMove=args.milliseconds, plannedGames=args.games, baseSeed=args.seed,
        openingPolicy=match.POLICY, start='initial', useBook=False,
        search='PUCT MCTS; per-config priors; exploration 1.5; reached-subtree reuse; maximum visits root choice',
        librarySha256=match.sha(args.library), runnerSha256=match.sha(__file__),
        baselineLibrarySha256=match.sha(args.baseline_library or args.library),
        sharedRunnerSha256=match.sha(ROOT / 'scripts/compare-nnue-search.py'),
        rulesSha256=match.sha(ROOT / 'scripts/compare-resnet.py'),
        policyAuditSha256=match.sha(ROOT / 'scripts/opening_policy.py'),
        sourceSha256={str(p.relative_to(ROOT)):match.sha(p) for p in sorted((ROOT/'cpp').rglob('*'))
                      if p.suffix in ('.cpp', '.hpp', '.h', '.inc')},
        candidatePolicySearch=('Pattern legal priors + parent mean minus 0.15 FPU' if magic.startswith(b'OTPAT') else 'Learned-prior order for unvisited children' if learned else 'Uniform priors'),
        platform=platform.platform(),
        limits='Paired colors and shared seeds from the initial board; repeated paths possible. No Elo inference.')
    records, engines = [], {}
    elapsed_offset=0
    if resume:
        previous=json.loads(args.output.read_text())
        if previous['complete']:
            raise ValueError('Completed reports cannot be resumed')
        for key in ('version','configs','millisecondsPerMove','plannedGames','baseSeed','openingPolicy',
                    'librarySha256','baselineLibrarySha256','baselineManifestSha256','sourceSha256',
                    'sharedRunnerSha256','rulesSha256','policyAuditSha256'):
            if previous[key]!=provenance[key]:
                raise ValueError('Resume provenance mismatch: '+key)
        records=previous['records']
        if len(records)>=args.games or previous['games']!=len(records):
            raise ValueError('Invalid unfinished report size')
        for i,record in enumerate(records):
            assert record['number']==i+1
            assert (record['black'],record['white'])==(('baseline','candidate') if i%2==0 else ('candidate','baseline'))
            assert record['seed']==(args.seed^((i//2+1)*0x9e3779b9))&0x7fffffff
            match.replay(record)
        elapsed_offset=previous['seconds']
        provenance['runnerSegments']=previous.get('runnerSegments',[
            dict(firstGame=1,lastGame=len(records),sha256=previous['runnerSha256'])])
        provenance['runnerSegments'].append(dict(firstGame=len(records)+1,sha256=provenance['runnerSha256']))
        provenance['resumeHistory']=previous.get('resumeHistory',[])+[
            dict(completedGames=len(records),previousReportSha256=match.sha(args.output),
                 previousRunnerSha256=previous['runnerSha256'],restartedPartialGame=len(records)+1)]
        provenance['elapsedPolicy']='Sum of recorded segment runtimes, excluding interruption downtime and discarded partial game'
    started = time.perf_counter()
    current=None
    summary=None

    def save(complete=False,full=True):
        nonlocal summary
        if full or summary is None:
            summary=match.aggregate(records)
        results,matrix=summary
        report = dict(**provenance, complete=complete, games=len(records),
                      seconds=elapsed_offset+time.perf_counter()-started, results=results, matrix=matrix,
                      replayedActions=sum(len(g['moves']) for g in records), records=records,currentGame=current)
        if full:
            match.rules.write_json(args.output, report)
        live = dict(report)
        live.pop('records')
        match.rules.write_json(args.output.with_suffix('.live.json'), live)

    try:
        for c in configs:
            library=(args.baseline_library or args.library) if c['id']=='baseline' else args.library
            engines[c['id']] = match.Engine(library, c)
        save()
        for index in range(len(records),args.games):
            black, white = configs if index % 2 == 0 else configs[::-1]
            for engine in engines.values():
                engine.request(kind='clear')
            seed = (args.seed ^ ((index//2+1)*0x9e3779b9)) & 0x7fffffff
            board, turn = match.rules.initial(), 1
            moves, decisions = [], []
            current=dict(number=index+1,black=black['name'],white=white['name'],
                         ply=0,blackCount=2,whiteCount=2,turn=turn,lastMove=None,board=board)
            save(full=False)
            while True:
                available = match.rules.legal(board, turn)
                if not available:
                    if not match.rules.legal(board, -turn):
                        break
                    move = -1
                else:
                    who = black['id'] if turn == 1 else white['id']
                    response = engines[who].request(kind='search', board=board, turn=turn,
                                                   seed=seed, ms=args.milliseconds)
                    config=black if turn==1 else white
                    decision = dict(ply=len(moves), turn=turn, engine=who,valueTarget=config['valueTarget'], **response)
                    match.audit_decision(board, turn, decision)
                    decisions.append(decision)
                    move = response['move']
                board, turn = match.rules.play(board, turn, move)
                moves.append(move)
                assert len(moves) <= 120
                current.update(ply=len(moves),blackCount=board.count(1),whiteCount=board.count(-1),
                               turn=turn,lastMove=move,board=board)
                save(full=False)
            difference = sum(board)
            record = dict(number=index+1, black=black['id'], white=white['id'], seed=seed,
                openingPolicy=match.POLICY, opening=[], moves=moves, decisions=decisions,
                finalBoard=board, blackDiscDifference=difference,
                winner=black['id'] if difference > 0 else white['id'] if difference < 0 else None)
            match.replay(record)
            records.append(record)
            current=None
            save()
            print(json.dumps(dict(completed=len(records), winner=record['winner'], difference=difference)), flush=True)
        assert match.sha(args.library) == provenance['librarySha256']
        assert match.sha(args.baseline_library or args.library) == provenance['baselineLibrarySha256']
        assert all(match.sha(ROOT/c['model']) == c['modelSha256'] for c in provenance['configs'])
        if provenance.get('runnerSegments'):
            provenance['runnerSegments'][-1]['lastGame']=len(records)
        save(complete=True)
    finally:
        for engine in engines.values():
            engine.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    active=json.loads((ROOT/'models/default-model.json').read_text())
    parser.add_argument('--baseline-manifest', type=Path, default=ROOT / active['baselineManifest'])
    parser.add_argument('--baseline-library', type=Path, help='Frozen approved-baseline reference library, otherwise same as candidate')
    parser.add_argument('--label', default='v1.8-candidate')
    parser.add_argument('--library', type=Path, default=ROOT / ('.build/native/libothello.' + ('dylib' if sys.platform=='darwin' else 'so')))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume',action='store_true',help='Resume an unfinished report after verifying models, budgets and completed games')
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--milliseconds', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261028)
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or not 1 <= args.milliseconds <= 60000:
        parser.error('Even games >=2 and milliseconds 1..60000 required')
    compare(args)
