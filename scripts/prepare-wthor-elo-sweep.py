#!/usr/bin/env python3
"""Sweep WTHOR-only training filters and export one equally counted data candidate."""
import collections,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from training_consensus import aggregate,boards_from_keys,canonical
from pattern_model import legal_mask
from wthor_elo_filter import both_below,choose_candidate
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'.build/training/consensus-elo-v1.40';WORK=ROOT/'.build/training/wthor-elo-v1.41'
REPORT=ROOT/'models/v1.41-wthor-elo-filter.json';PROTOCOL=ROOT/'models/v1.41-wthor-elo-filter-protocol.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,r:Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')


def fingerprint(array):return dict(shape=list(array.shape),dtype=array.dtype.str,sha256=hashlib.sha256(array.tobytes()).hexdigest())


def main():
    assert not REPORT.exists() and not (WORK/'consensus.npz').exists(),'Completed candidates are immutable'
    protocol=json.loads(PROTOCOL.read_text());base=json.loads((ROOT/'models/v1.40-training-consensus-elo.json').read_text())
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in base['artifactSha256'].items():assert sha(ROOT/name)==digest,name
    WORK.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    def progress(stage,n=0,total=1,summary='',complete=False):
        write(WORK/'progress.json',dict(experiment='pattern-suite-sampling',stage=stage,completed=n,total=total,summary=summary,complete=complete,seconds=time.perf_counter()-started))
    config=json.loads((ROOT/'models/current-experiment.json').read_text());config['version']='WTHOR-only Elo 阈值比较 · 不训练';config['experiments'].append(dict(name='WTHOR：Elo 阈值与不加权聚合',focus='wthor-elo-filter',games=0,runtimeReport=str((WORK/'progress.json').relative_to(ROOT)),report='models/v1.41-wthor-elo-filter-monitor.json'));write(ROOT/'models/current-experiment.json',config)
    progress('核验冻结验证/测试与 WTHOR 来源')
    frozen=dict(np.load(ROOT/'.build/training/v1/positions.npz'));labels=dict(np.load(ROOT/'.build/training/v1.8/policy.npz'))
    holdouts={split:{name:fingerprint(frozen[split+'_'+name]) for name in ['board','turn','target']} for split in ['validation','test']}
    for split in holdouts:holdouts[split]['policy']=fingerprint(labels[split])
    holdout_keys=np.concatenate([canonical(frozen[s+'_board'],frozen[s+'_turn'])[0] for s in ['validation','test']])
    del frozen,labels
    ledger={r['gameId']:r for r in map(json.loads,(BASE/'game-elo.jsonl').open())}
    index=list(map(json.loads,(BASE/'game-index.jsonl').open()));n=len(index)
    eligible=np.zeros(n,bool);wthor=np.zeros(n,bool);logistello=np.zeros(n,bool);a=np.full(n,np.nan);b=np.full(n,np.nan);support=np.zeros((n,2),np.uint32);theory=np.zeros(n,np.int8);actual=np.zeros(n,np.int8);comparable=np.zeros(n,bool)
    for i,line in enumerate((ROOT/'.build/training/games.jsonl').open()):
        game=json.loads(line);assert index[i]['index']==i and index[i]['gameId']==game['id']
        eligible[i]=game['complete'] and game['split']=='train';wthor[i]=game['source']=='wthor';logistello[i]=game['source']=='logistello'
        if wthor[i]:
            r=ledger[game['id']];a[i]=np.nan if r['blackTrainYearStartElo'] is None else r['blackTrainYearStartElo'];b[i]=np.nan if r['whiteTrainYearStartElo'] is None else r['whiteTrainYearStartElo'];support[i]=[r['blackTrainYearStartGames'],r['whiteTrainYearStartGames']]
            actual[i]=np.sign(game['blackResult'])
            if game['theoreticalBlackResult'] is not None:theory[i]=np.sign(game['theoreticalBlackResult']);comparable[i]=True
    eligible_wthor=eligible&wthor;eligible_log=eligible&logistello;assert eligible_wthor.sum()==80561 and eligible_log.sum()==60775
    with np.load(BASE/'occurrences.npz') as archive:occ={k:archive[k] for k in ['keys','action_bits','outcome','source','game_index']}
    g=occ['game_index'];assert eligible[g].all();np.testing.assert_array_equal(occ['source']==0,wthor[g])
    source_mask=wthor[g];wg=g[source_mask];keys=occ['keys'][source_mask];bits=occ['action_bits'][source_mask];outcome=occ['outcome'][source_mask]
    unique,inverse=np.unique(keys,return_inverse=True);placed=np.unpackbits((unique['own']|unique['opponent']).astype('<u8').view(np.uint8).reshape(-1,8),axis=1).sum(1).astype(int)-4
    lowest=np.array([(int(x)&-int(x)).bit_length()-1 for x in bits],np.uint8);orbit_code=inverse.astype(np.uint64)*64+lowest
    rows=[];unknown=~np.isfinite(a)|~np.isfinite(b)
    for threshold in [None]+protocol['thresholds']:
        keep_game=eligible_wthor.copy() if threshold is None else eligible_wthor&~both_below(a,b,threshold)
        keep=keep_game[wg];counts=np.bincount(inverse[keep],minlength=len(unique));present=counts>0
        wdl=np.zeros((len(unique),3),np.uint32);category=np.where(outcome[keep]>0,0,np.where(outcome[keep]<0,1,2));np.add.at(wdl,(inverse[keep],category),1)
        pairs=np.unique(orbit_code[keep]);choices=np.bincount((pairs//64).astype(np.int64),minlength=len(unique))
        compare=keep_game&comparable;stage=[]
        for lo,hi in [(9,18),(21,36),(39,54)]:stage.append(dict(placements=[lo,hi],uniquePositions=int((present&(placed>=lo)&(placed<=hi)).sum())))
        row=dict(threshold=threshold,trainGames=int(keep_game.sum()),removedWthorGames=int((eligible_wthor&~keep_game).sum()),gameRetentionVsWthor=float(keep_game.sum()/eligible_wthor.sum()),retainedOccurrences=int(keep.sum()),uniquePositions=int(present.sum()),uniqueRetentionVsWthor=float(present.mean()),multiActionOrbitPositions=int((choices>1).sum()),multipleOutcomePositions=int(((wdl>0).sum(1)>1).sum()),bothWinAndLossPositions=int(((wdl[:,0]>0)&(wdl[:,1]>0)).sum()),unknownRatingGamesRetained=int((keep_game&unknown).sum()),bothPlayers20PriorGames=int((keep_game&(support.min(1)>=20)).sum()),theoryComparableGames=int(compare.sum()),recordedVs24EmptyWdlDisagreementGames=int((compare&(actual!=theory)).sum()),recordedVs24EmptyWdlDisagreementRate=float((actual[compare]!=theory[compare]).mean()),stages=stage)
        rows.append(row);progress('比较 Elo 阈值',len(rows),len(protocol['thresholds'])+1,f'{threshold}: {row["trainGames"]:,} 盘 / {row["uniquePositions"]:,} 局面');print(json.dumps(row,ensure_ascii=False),flush=True)
    chosen=choose_candidate(rows,protocol['selection']['minGameRetention'],protocol['selection']['minUniqueRetention']);threshold=chosen['threshold']
    keep_game=eligible_wthor&~both_below(a,b,threshold);keep=keep_game[g];assert (occ['source'][keep]==0).all()
    progress('生成所选候选：按次数聚合，不使用 Elo 权重')
    filtered={k:v[keep] for k,v in occ.items()};result=aggregate(filtered['keys'],filtered['action_bits'],filtered['outcome'],np.ones(int(keep.sum())),filtered['source'])
    np.testing.assert_array_equal(result.pop('policy_quality_mass'),result['policy_mass']);np.testing.assert_array_equal(result.pop('policy_quality_total'),result['occurrence_count'])
    selected_keys=result['keys'];count=result['occurrence_count'];sg=np.repeat(np.arange(len(count)),np.diff(result['policy_offsets']).astype(np.int64))
    result['wdl_probability']=result['wdl_count']/count[:,None];result['value_mean']=result['wdl_probability'][:,0]-result['wdl_probability'][:,1];result['policy_probability']=result['policy_mass']/count[sg];result['policy_loss_weight']=np.ones(len(count),np.float32)
    with np.load(BASE/'consensus.npz') as archive:matched_keys=archive['keys'][archive['matched_train_group']]
    location=np.searchsorted(selected_keys,matched_keys);valid=location<len(selected_keys);valid[valid]&=selected_keys[location[valid]]==matched_keys[valid]
    result['matched_train_group']=np.where(valid,location,-1).astype(np.int32);result['matched_train_row']=np.flatnonzero(valid).astype(np.uint32);result['matched_retained_group']=location[valid].astype(np.uint32)
    assert len(selected_keys)==chosen['uniquePositions'] and len(filtered['keys'])==chosen['retainedOccurrences']
    assert not np.isin(selected_keys,holdout_keys).any() and not result['source_count'][:,1].any()
    invalid=0
    for start in range(0,len(count),16384):
        end=min(start+16384,len(count));mask=legal_mask(boards_from_keys(selected_keys[start:end]),np.ones(end-start,np.int8));lo,hi=map(int,result['policy_offsets'][[start,end]]);invalid+=int((~mask[sg[lo:hi]-start,result['policy_action'][lo:hi]]).sum())
    assert invalid==0
    np.testing.assert_array_equal(result['wdl_count'].sum(1),count);np.testing.assert_array_equal(np.bincount(sg,weights=result['policy_mass'],minlength=len(count)),count)
    np.testing.assert_allclose(np.bincount(sg,weights=result['policy_probability'],minlength=len(count)),1,atol=1e-12,rtol=1e-12)
    np.savez_compressed(WORK/'consensus.npz',**result);np.savez_compressed(WORK/'occurrences.npz',**filtered)
    decisions=WORK/'game-filter.jsonl'
    with decisions.open('w') as f:
        for i in np.flatnonzero(eligible):f.write(json.dumps(dict(index=int(i),gameId=index[i]['gameId'],source=index[i]['source'],keep=bool(keep_game[i]),reason='keep' if keep_game[i] else 'exclude_logistello' if logistello[i] else 'both_year_start_train_elo_below_threshold',blackElo=None if np.isnan(a[i]) else float(a[i]),whiteElo=None if np.isnan(b[i]) else float(b[i])))+'\n')
    outputs=[WORK/'consensus.npz',WORK/'occurrences.npz',decisions]
    report=dict(complete=True,version='v1.41-wthor-elo-filter',status='data-candidate-no-training',modelTrained=False,defaultPromoted=False,eloWeighting=False,originalTrainGames=int(eligible.sum()),excludedLogistelloGames=int(eligible_log.sum()),sourceOnlyWthor=rows[0],sweep=rows[1:],selected=chosen,matchedFrozenMillionRetained=int(valid.sum()),matchedFrozenMillionMissing=int((~valid).sum()),crossHoldoutOverlap=0,illegalSparseActions=invalid,frozenHoldouts=holdouts,protocolSha256=sha(PROTOCOL),preservedSha256=protocol['preservedSha256'],artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in outputs},scriptSha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'scripts/wthor_elo_filter.py',ROOT/'scripts/training_consensus.py']},seconds=time.perf_counter()-started,
        schema='Train-only relative D4 canonical boards; unweighted empirical policy/WLD counts and probabilities. policy_loss_weight is exactly 1. matched_train_group has -1 for removed original-million rows; matched_train_row/matched_retained_group provide the safe paired subset. Frozen validation/test remain in original cache.',limits='Selection uses training-data retention and a training-only ending consistency proxy, not validation/test model scores or match results. Recorded outcomes and exact results at one 24-empty position can differ due to later play; disagreement is not a per-move blunder rate. Unknown Elo retained separately. AND filter allows weak-player moves against strong opponents. WTHOR contains human and program participants; not a certified professional-only corpus. No Elo recomputation after filtering; no training.')
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    write(REPORT,report);progress('WTHOR 阈值比较与等权聚合完成',1,1,f'候选 {threshold}: {chosen["trainGames"]:,} 盘 / {len(selected_keys):,} 局面；未训练',True);write(ROOT/'models/v1.41-wthor-elo-filter-monitor.json',json.loads((WORK/'progress.json').read_text()))
    print(json.dumps(dict(complete=True,selected=chosen,matchedMillionRetained=int(valid.sum()),seconds=report['seconds']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
