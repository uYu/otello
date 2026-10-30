#!/usr/bin/env python3
"""Build independent empirical soft targets and causal, internal Elo metadata."""
import argparse,collections,ctypes,datetime,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from training_consensus import EloPool,rate_records,canonical,actor_weight,aggregate,boards_from_keys
from pattern_model import legal_mask
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,r:Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=ROOT/'.build/training/consensus-elo-v1.40')
    parser.add_argument('--report',type=Path,default=ROOT/'models/v1.40-training-consensus-elo.json')
    args=parser.parse_args();work=args.work.resolve();report=args.report.resolve()
    assert not report.exists() and not (work/'consensus.npz').exists(),'Choose fresh output paths; completed data candidates are immutable'
    protocol_path=ROOT/'models/v1.40-training-consensus-elo-protocol.json';protocol=json.loads(protocol_path.read_text());started=time.perf_counter()
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    work.mkdir(parents=True,exist_ok=True)
    def progress(stage,n=0,total=1,summary='',complete=False):
        write(work/'progress.json',dict(experiment='pattern-suite-sampling',stage=stage,completed=n,total=total,summary=summary,complete=complete,seconds=time.perf_counter()-started))
    config=json.loads((ROOT/'models/current-experiment.json').read_text());config['experiments'].append(dict(name='v1.40 数据候选：Elo 重放 + 多走法/多结果',focus='consensus-elo',games=0,runtimeReport=str((work/'progress.json').relative_to(ROOT)),report='models/v1.40-training-consensus-elo-monitor.json'));write(ROOT/'models/current-experiment.json',config)
    audit=json.loads((ROOT/'models/training-data-audit-2026-10-30.json').read_text());sidecar=ROOT/audit['playerProvenance']['sidecar'];assert sha(sidecar)==audit['playerProvenance']['sidecarSha256']
    metadata={r['gameId']:r for r in map(json.loads,sidecar.open())}
    names_path=ROOT/'.build/training/data-audit-20261030/metadata/WTHOR.JOU';raw=names_path.read_bytes();size=(len(raw)-16)//20
    names=[raw[16+i*20:36+i*20].split(b'\0')[0].decode('latin-1') for i in range(size)]
    games=ROOT/'.build/training/games.jsonl';records=[]
    for line in games.open():
        game=json.loads(line)
        if game['source']=='wthor':
            m=metadata[game['id']];records.append(dict(gameId=game['id'],year=m['year'],rawRecordIndex=m['rawRecordIndex'],blackPlayerIndex=m['blackPlayerIndex'],whitePlayerIndex=m['whitePlayerIndex'],split=game['split'],complete=game['complete'],blackResult=game['blackResult']))
    progress('按年份、原始记录顺序计算两份 Elo',0,len(records))
    ledger,full,train,snapshots=rate_records(records,size,protocol['elo']['initial'],protocol['elo']['k'],protocol['elo']['scale'])
    by_id={r['gameId']:r for r in ledger};ledger_path=work/'game-elo.jsonl'
    with ledger_path.open('w') as f:
        for r in ledger:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    players=[dict(playerIndex=i,name=names[i],fullCorpusElo=float(full.rating[i]),fullCorpusRatedGames=int(full.games[i]),trainOnlyElo=float(train.rating[i]),trainOnlyRatedGames=int(train.games[i]),provisional=int(full.games[i])<20) for i in range(1,size)]
    write(work/'players.json',dict(namespace='WTHOR.JOU index; internal estimates, not WOF Elo',unknownIndex=0,players=players))
    # Quantify order sensitivity for display, never use it to choose training weights.
    reverse=EloPool(size,protocol['elo']['initial'],protocol['elo']['k'],protocol['elo']['scale'])
    for r in sorted(records,key=lambda r:(r['year'],-r['rawRecordIndex'],r['gameId'])):
        if r['complete']:reverse.update(r['blackPlayerIndex'],r['whitePlayerIndex'],(r['blackResult']>0)+.5*(r['blackResult']==0))
    active=full.games>=20;order_rms=float(np.sqrt(np.mean((full.rating[active]-reverse.rating[active])**2)))
    parent=np.arange(size)
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for r in ledger:
        if r['rated']:parent[find(r['blackPlayerIndex'])]=find(r['whitePlayerIndex'])
    components=collections.Counter(find(i) for i in range(1,size) if full.games[i])
    data=dict(np.load(ROOT/'.build/training/v1/positions.npz'))
    def keys_for(split):
        return np.concatenate([canonical(data[split+'_board'][start:start+16384],data[split+'_turn'][start:start+16384])[0] for start in range(0,len(data[split+'_board']),16384)])
    holdout=np.concatenate([keys_for('test'),keys_for('validation')]);matched=keys_for('train')
    lib=ctypes.CDLL(str(ROOT/protocol['nativeLibrary']));ptr=ctypes.POINTER(ctypes.c_int);lib.ot_play.argtypes=[ptr,ctypes.c_int,ctypes.c_int,ptr];lib.ot_play.restype=ctypes.c_int
    chunks=[];buffer=[];targets=[];moves=[];quality=[];sources=[];indices=[];elo=[];prior=[];gindex=[]
    outcomes=[];bits=[];weights=[];srcs=[];groups=[];elo_rows=[];prior_rows=[];game_rows=[]
    stages=collections.Counter();known=0;neutral=0;games_replayed=0;actions=0;raw_positions=0;removed=0
    def flush():
        nonlocal removed
        if not buffer:return
        key,action=canonical(np.stack(buffer),np.ones(len(buffer),np.int8),np.array(moves));keep=~np.isin(key,holdout);removed+=int((~keep).sum())
        chunks.append(key[keep]);bits.append(action[keep]);outcomes.append(np.array(targets,np.int8)[keep]);weights.append(np.array(quality,np.float64)[keep]);srcs.append(np.array(sources,np.uint8)[keep]);game_rows.append(np.array(gindex,np.uint32)[keep]);elo_rows.append(np.array(elo,np.float32)[keep]);prior_rows.append(np.array(prior,np.uint32)[keep])
        buffer.clear();targets.clear();moves.clear();quality.clear();sources.clear();gindex.clear();elo.clear();prior.clear()
    progress('回放训练棋谱并保留每次出现的证据',0,141336)
    index_path=work/'game-index.jsonl'
    with index_path.open('w') as output:
        for index,line in enumerate(games.open()):
            game=json.loads(line);output.write(json.dumps(dict(index=index,gameId=game['id'],source=game['source'],sourceFile=game['sourceFile'],split=game['split']))+'\n')
            if game['split']!='train' or not game['complete']:continue
            row=by_id.get(game['id']);cells=(ctypes.c_int*64)();cells[28]=cells[35]=1;cells[27]=cells[36]=-1;turn=1;ply=0;out=(ctypes.c_int*67)()
            for move in game['moves']:
                if 8<=ply<=56 and ply%3==0 and move>=0:
                    if row:
                        color='black' if turn==1 else 'white';rating=row[color+'TrainYearStartElo'];n_prior=row[color+'TrainYearStartGames']
                    else:rating=None;n_prior=0
                    buffer.append(np.ctypeslib.as_array(cells).astype(np.int8)*turn);targets.append(int(np.sign(game['blackResult']*turn)));moves.append(move)
                    quality.append(actor_weight(rating,n_prior,protocol['elo']['initial'],protocol['elo']['scale'],protocol['quality']['warmupGames']))
                    sources.append(0 if game['source']=='wthor' else 1);gindex.append(index);elo.append(np.nan if rating is None else rating);prior.append(n_prior)
                    known+=rating is not None;neutral+=rating is None or n_prior==0;raw_positions+=1
                    if len(buffer)==32768:flush()
                assert lib.ot_play(cells,turn,move,out)==1,game['id'];ctypes.memmove(cells,out,64*ctypes.sizeof(ctypes.c_int));turn=out[64];ply+=move>=0;actions+=1
            assert out[65]==1;games_replayed+=1
            if games_replayed%10000==0:progress('回放训练棋谱',games_replayed,141336)
    flush();keys=np.concatenate(chunks);action_bits=np.concatenate(bits);y=np.concatenate(outcomes);q=np.concatenate(weights);source=np.concatenate(srcs)
    del buffer,chunks,bits,outcomes,weights,srcs,metadata,records,data
    progress('聚合走法分布、结果分布与 Elo 质量权重')
    result=aggregate(keys,action_bits,y,q,source)
    unique=result['keys'];mapping=np.searchsorted(unique,matched);assert np.array_equal(unique[mapping],matched)
    result['matched_train_group']=mapping.astype(np.uint32)
    result['value_mean']=(result['wdl_count'][:,0].astype(np.float64)-result['wdl_count'][:,1])/result['occurrence_count']
    assert len(unique)==audit['rawTrainingLabelMultiplicity']['eligibleTrain']['uniquePositions'] and games_replayed==141336 and actions==8593796
    assert not np.isin(unique,holdout).any()
    group=np.repeat(np.arange(len(unique)),np.diff(result['policy_offsets']).astype(np.int64));multi=int((np.diff(result['policy_offsets'])>1).sum())
    # Verify every sparse action against legal moves in the canonical orientation.
    invalid=0
    for start in range(0,len(unique),16384):
        end=min(start+16384,len(unique));board=boards_from_keys(unique[start:end]);mask=legal_mask(board,np.ones(len(board),np.int8));lo,hi=int(result['policy_offsets'][start]),int(result['policy_offsets'][end])
        invalid+=int((~mask[group[lo:hi]-start,result['policy_action'][lo:hi]]).sum())
    assert invalid==0
    dataset=work/'consensus.npz';np.savez_compressed(dataset,**result)
    occurrences=work/'occurrences.npz';np.savez_compressed(occurrences,keys=keys,action_bits=action_bits,outcome=y,quality=q,source=source,game_index=np.concatenate(game_rows),actor_train_year_start_elo=np.concatenate(elo_rows),actor_train_year_start_games=np.concatenate(prior_rows))
    rated=[r for r in ledger if r['rated']];actual=np.array([(r['blackResult']>0)+.5*(r['blackResult']==0) for r in rated]);predicted=np.array([r['blackExpectedScore'] for r in rated])
    top=sorted([p for p in players if p['fullCorpusRatedGames']>=100],key=lambda p:p['fullCorpusElo'],reverse=True)[:10]
    stats=dict(complete=True,version='v1.40-training-consensus-elo',status='data-candidate',modelTrained=False,defaultPromoted=False,seconds=time.perf_counter()-started,
        elo=dict(initial=protocol['elo']['initial'],k=protocol['elo']['k'],scale=protocol['elo']['scale'],knownNamesInitialized=size-1,observedKnownPlayers=int((full.games>0).sum()),ratedCompleteGames=len(rated),unratedOrIncompleteGames=len(ledger)-len(rated),trainOnlyRatedGames=int(train.games.sum()//2),fullCorpusRatingQuantiles=dict(zip(['minimum','p25','median','p75','maximum'],map(float,np.quantile(full.rating[full.games>0],[0,.25,.5,.75,1])))),provisionalObservedPlayers=int(((full.games>0)&(full.games<20)).sum()),components=sorted(components.values(),reverse=True),withinYearReverseOrderRmsFor20PlusGames=order_rms,preGameOutcomeSquaredError=float(np.mean((predicted-actual)**2)),neutralPredictionSquaredError=float(np.mean((.5-actual)**2)),top100Games=top),
        data=dict(replayedTrainGames=games_replayed,replayedActions=actions,candidateOccurrencesBeforeHoldoutExclusion=raw_positions,removedHoldoutOccurrences=removed,retainedOccurrences=len(keys),uniquePositions=len(unique),matchedFrozenTrainPositions=len(mapping),sparseActionEntries=len(group),multiActionPositionsIncludingStabilizerSymmetry=multi,multipleOutcomePositions=int(((result['wdl_count']>0).sum(1)>1).sum()),bothWinAndLossPositions=int(((result['wdl_count'][:,0]>0)&(result['wdl_count'][:,1]>0)).sum()),crossHoldoutOverlap=0,illegalSparseActions=invalid,qualityQuantiles=list(map(float,np.quantile(q,[0,.1,.25,.5,.75,.9,1]))),neutralQualityOccurrences=int((q==.5).sum()),positionsWithPriorKnownElo=known,neutralBeforeHoldoutExclusion=neutral),
        schema=dict(wdlOrder=['win','loss','draw'],sources=['wthor','logistello'],policy='CSR: offsets/action/mass; mass normalized by occurrence_count gives empirical policy; quality_mass normalized by quality_total gives optional internal-Elo policy',value='value_mean=P(win)-P(loss); WDL probabilities=wdl_count/occurrence_count; do not apply sign() to soft value targets',orientation='D4 canonical relative board, own=+1, opponent=-1, turn=+1. Stabilizer-equivalent actions share observation mass equally.',views='All unique positions and matched_train_group for the exact original million rows. Frozen validation/test caches remain separate and unchanged.'),
        protocolSha256=sha(protocol_path),scriptSha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'scripts/training_consensus.py',ROOT/'scripts/pattern_model.py']},nativeLibrarySha256=sha(ROOT/protocol['nativeLibrary']),preservedSha256=protocol['preservedSha256'],artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in [dataset,occurrences,ledger_path,work/'players.json',index_path]},
        limits='Internal logistic Elo with fixed K, not official WOF/FIDE ratings. Only year/file order is available; within-year real chronology unknown. Full-corpus ratings are descriptive only. Training weights use prior-year train-only ratings and support counts; no held-out outcome or own-game outcome enters. Unknown ID 0 is not one person and is never updated. Logistello has no player ID and gets neutral policy weight. Soft targets are empirical continuation statistics, not minimax values. No training, match, or low-Elo deletion performed.')
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    write(report,stats);progress('数据候选构建完成',1,1,f'{len(unique):,} 个局面；保留走法/结果分布；内部 Elo 已完成',True);write(ROOT/'models/v1.40-training-consensus-elo-monitor.json',json.loads((work/'progress.json').read_text()))
    print(json.dumps(dict(complete=True,seconds=stats['seconds'],elo={k:v for k,v in stats['elo'].items() if k!='top100Games'},data=stats['data']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
