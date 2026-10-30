#!/usr/bin/env python3
"""Read-only audit of frozen positions and recovery of WTHOR player provenance."""
import collections,ctypes,datetime,hashlib,json,struct,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from pattern_model import legal_mask
WORK=ROOT/'.build/training/data-audit-20261030';META=WORK/'metadata'
REPORT=ROOT/'models/training-data-audit-2026-10-30.json'
KEY=np.dtype([('own','<u8'),('opponent','<u8')])
grid=np.arange(64).reshape(8,8)
PERMS=np.stack([np.rot90(np.fliplr(grid) if mirror else grid,r).reshape(64) for mirror in [False,True] for r in range(4)])
INVERSE=np.argsort(PERMS,axis=1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,r:Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')


def canonical(board,turn,moves=None):
    relative=(board*turn[:,None]).astype(np.int8)
    best=np.empty(len(board),KEY);best['own']=np.iinfo(np.uint64).max;best['opponent']=np.iinfo(np.uint64).max
    action=np.full(len(board),64,np.uint8)
    for permutation,inverse in zip(PERMS,INVERSE):
        transformed=relative[:,permutation]
        own=np.packbits(transformed==1,axis=1,bitorder='little').copy().view('<u8').reshape(-1)
        opp=np.packbits(transformed==-1,axis=1,bitorder='little').copy().view('<u8').reshape(-1)
        better=(own<best['own'])|((own==best['own'])&(opp<best['opponent']))
        equal=(own==best['own'])&(opp==best['opponent'])
        if moves is not None:
            aligned=inverse[moves];action=np.where(better,aligned,np.where(equal,np.minimum(action,aligned),action)).astype(np.uint8)
        best['own'][better]=own[better];best['opponent'][better]=opp[better]
    return best,action


def lookup(name,size):
    path=META/name
    if not path.exists():return []
    b=path.read_bytes();n=struct.unpack_from('<H',b,8)[0];assert len(b)==16+n*size
    return [b[16+i*size:16+(i+1)*size].split(b'\0')[0].decode('latin-1') for i in range(n)]


def main():
    assert not REPORT.exists(),'Audit report is immutable; choose a new report for a new run'
    WORK.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    cache=ROOT/'.build/training/v1/positions.npz';policy=ROOT/'.build/training/v1.8/policy.npz';games=ROOT/'.build/training/games.jsonl'
    metadata=json.loads(cache.with_suffix('.json').read_text());provenance=json.loads(policy.with_suffix('.json').read_text())
    assert sha(cache)==metadata['cacheSha256']==provenance['cacheSha256'] and sha(policy)==provenance['policySha256'] and sha(games)==metadata['inputSha256']
    preserved={str(p.relative_to(ROOT)):sha(p) for p in [cache,policy,games,ROOT/'models/default-model.json',ROOT/'models/v1.37-default.json',ROOT/'models/pattern32-symmetry-v1.37.otpat',ROOT/'src/game/wasm/othello.mjs']}
    def progress(stage,n=0,total=1):
        write(WORK/'progress.json',dict(experiment='pattern-suite-sampling',stage=stage,completed=n,total=total,summary='只读审计；不改冻结标签',complete=False,seconds=time.perf_counter()-started))
    config=json.loads((ROOT/'models/current-experiment.json').read_text());config['experiments'].append(dict(name='训练数据：完整性、棋手溯源与多标签审计',focus='data-audit',games=0,runtimeReport=str((WORK/'progress.json').relative_to(ROOT)),report='models/training-data-audit-2026-10-30-monitor.json'));write(ROOT/'models/current-experiment.json',config)
    data=dict(np.load(cache));labels=dict(np.load(policy));split_results={};all_keys={}
    for split in ['test','validation','train']:
        progress('冻结局面完整性与对称去重：'+split)
        boards,turns,targets=[data[split+'_'+x] for x in ['board','turn','target']];keys=[];counts=[];invalid=0
        assert np.isfinite(targets).all() and np.isin(boards,[-1,0,1]).all() and np.isin(turns,[-1,1]).all()
        for start in range(0,len(boards),16384):
            b,t=boards[start:start+16384],turns[start:start+16384];key,_=canonical(b,t);keys.append(key)
            mask=legal_mask(b,t);actions=labels[split][start:start+len(b)].astype(int)
            valid=(actions>=0)&(actions<64);invalid+=int((~valid).sum())+int((~mask[np.arange(len(b))[valid],actions[valid]]).sum());counts.append(mask.sum(1))
        keys=np.concatenate(keys);all_keys[split]=keys;legal=np.concatenate(counts);placed=np.count_nonzero(boards,axis=1)-4
        unique=np.unique(keys);stage=[]
        for lo,hi in [(9,18),(21,36),(39,54)]:
            mask=(placed>=lo)&(placed<=hi);y=np.sign(targets[mask]);stage.append(dict(placements=[lo,hi],positions=int(mask.sum()),meanLegalMoves=float(legal[mask].mean()),win=int((y>0).sum()),draw=int((y==0).sum()),loss=int((y<0).sum())))
        y=np.sign(targets)
        split_results[split]=dict(positions=len(boards),uniqueD4RelativePositions=len(unique),duplicates=len(boards)-len(unique),invalidPolicyLabels=invalid,onlyOneLegalMove=int((legal==1).sum()),noLegalMove=int((legal==0).sum()),placements={str(k):int(v) for k,v in zip(*np.unique(placed,return_counts=True))},actorOutcomes=dict(win=int((y>0).sum()),draw=int((y==0).sum()),loss=int((y<0).sum())),stages=stage)
    overlaps={a+' / '+b:int(len(np.intersect1d(all_keys[a],all_keys[b]))) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    players=lookup('WTHOR.JOU',20);tournaments=lookup('WTHOR.TRN',26);raw={};appearances=collections.Counter();raw_games=0;depths=collections.Counter();raw_hashes={}
    manifest=json.loads((ROOT/'.build/training/raw/manifest.json').read_text());expected={v['file']:v['sha256'] for v in manifest['files']}
    for p in sorted((ROOT/'.build/training/raw').glob('*.wtb')):
        b=p.read_bytes();assert sha(p)==expected[p.name];raw_hashes[p.name]=sha(p);n=struct.unpack_from('<I',b,4)[0];assert len(b)==16+n*68
        for index,offset in enumerate(range(16,len(b),68)):
            tournament,black,white=struct.unpack_from('<HHH',b,offset);appearances[black]+=1;appearances[white]+=1
            moves=bytes((v//10-1)*8+v%10-1 for v in b[offset+8:offset+68] if v)
            raw.setdefault((p.name,moves),dict(rawRecordIndex=index,tournamentIndex=tournament,blackPlayerIndex=black,whitePlayerIndex=white,year=int(p.stem[-4:])))
        raw_games+=n;depths[b[14]]+=n
    if players:assert max(appearances)<len(players)
    suffix='dylib' if sys.platform=='darwin' else 'so';lib=ctypes.CDLL(str(ROOT/f'.build/native-v1.37-reference/libothello.{suffix}'))
    ptr=ctypes.POINTER(ctypes.c_int);lib.ot_play.argtypes=[ptr,ctypes.c_int,ctypes.c_int,ptr];lib.ot_play.restype=ctypes.c_int
    chunks=[];outcomes=[];movesets=[];buffer=[];ys=[];ms=[];source_split=collections.Counter();matched=0;missing=0;actions=0;complete_train_games=0
    def flush():
        if buffer:
            key,move=canonical(np.stack(buffer),np.ones(len(buffer),np.int8),np.array(ms));chunks.append(key);outcomes.append(np.array(ys,np.int8));movesets.append(move);buffer.clear();ys.clear();ms.clear()
    progress('恢复棋手溯源；回放训练棋谱并检查多结果/多走法')
    sidecar=WORK/'game-player-metadata.jsonl'
    with games.open() as stream,sidecar.open('w') as output:
        for number,line in enumerate(stream,1):
            game=json.loads(line);source_split[(game['source'],game['split'],bool(game['complete']))]+=1
            if game['source']=='wthor':
                entry=raw.get((game['sourceFile'],bytes(m for m in game['moves'] if m>=0)))
                if entry is None:missing+=1
                else:
                    matched+=1;record=dict(gameId=game['id'],source='wthor',sourceFile=game['sourceFile'],sourceSha256=raw_hashes[game['sourceFile']],split=game['split'],**entry,playerIndexNamespace='WTHOR.JOU file index, not WOF ID',blackName=players[entry['blackPlayerIndex']] if players else None,whiteName=players[entry['whitePlayerIndex']] if players else None,historicalBlackRating=None,historicalWhiteRating=None)
                    output.write(json.dumps(record,ensure_ascii=False)+'\n')
            if not game['complete'] or game['split']!='train':continue
            complete_train_games+=1;cells=(ctypes.c_int*64)();cells[28]=cells[35]=1;cells[27]=cells[36]=-1;turn=1;ply=0;out=(ctypes.c_int*67)()
            for move in game['moves']:
                if 8<=ply<=56 and ply%3==0 and move>=0:
                    buffer.append(np.ctypeslib.as_array(cells).astype(np.int8)*turn);ys.append(int(np.sign(game['blackResult']*turn)));ms.append(move)
                    if len(buffer)==32768:flush()
                assert lib.ot_play(cells,turn,move,out)==1,game['id'];ctypes.memmove(cells,out,64*ctypes.sizeof(ctypes.c_int));turn=out[64];ply+=move>=0;actions+=1
            assert out[65]==1,game['id']
            if complete_train_games%10000==0:progress('回放训练棋谱与多标签统计',complete_train_games,141336)
    flush();key=np.concatenate(chunks);outcome=np.concatenate(outcomes);move=np.concatenate(movesets);unique,inverse,count=np.unique(key,return_inverse=True,return_counts=True)
    holdout=np.concatenate([all_keys['test'],all_keys['validation']]);owned=np.isin(unique,holdout);eligible=~owned
    wins=np.bincount(inverse,weights=outcome>0,minlength=len(unique));draws=np.bincount(inverse,weights=outcome==0,minlength=len(unique));losses=np.bincount(inverse,weights=outcome<0,minlength=len(unique))
    action_pairs=np.unique(inverse.astype(np.uint64)*64+move);distinct=np.bincount((action_pairs//64).astype(np.int64),minlength=len(unique))
    index=np.searchsorted(unique,all_keys['train']);assert (unique[index]==all_keys['train']).all() and not owned[index].any()
    def multiplicity(indices):
        c=count[indices];w,d,l=wins[indices],draws[indices],losses[indices]
        return dict(uniquePositions=len(c),repeatedPositions=int((c>1).sum()),multiActionOrbitPositions=int((distinct[indices]>1).sum()),multipleRecordedOutcomePositions=int((((w>0).astype(int)+(d>0)+(l>0))>1).sum()),bothWinAndLossPositions=int(((w>0)&(l>0)).sum()),maximumOccurrences=int(c.max()))
    joint=dict(replayedCompleteTrainGames=complete_train_games,replayedActions=actions,sampledTrainOccurrences=len(key),rawUniqueTrainPositions=len(unique),rawTrainPositionsExcludedByFrozenHoldoutOwnership=int(owned.sum()),eligibleTrain=multiplicity(np.flatnonzero(eligible)),currentMillionTrain=multiplicity(index),limits='Outcome disagreement reflects differing continuations, not automatically corrupt records. Action diversity uses D4 stabilizer orbits. Aggregate train occurrences only; held-out canonical positions excluded. No quality weighting or relabeling performed.')
    names=collections.Counter(players);duplicates=[name for name,n in names.items() if n>1]
    player_report=dict(rawWthorGames=raw_games,rawFiles=len(raw_hashes),distinctPlayerIndexValues=len(appearances),knownDistinctPlayerIndexValues=len(appearances)-int(0 in appearances),unknownIndex=0,unknownIndexName=players[0] if players else None,unknownAppearances=appearances[0],lookupEntries=len(players),lookupDuplicateDisplayNames=len(duplicates),joinedConvertedWthorGames=matched,unmatchedConvertedWthorGames=missing,sidecar=str(sidecar.relative_to(ROOT)),sidecarSha256=sha(sidecar),historicalRatingsAttached=0,theoreticalDepthHistogram=dict(depths),identity='WTHOR indices belong to the names-file snapshot; they are not WOF rating identifiers. WTHOR.JOU contains names, not Elo. Encoding rendered as reversible Latin-1, not used for automatic name matching.',ratingLookupStatus=json.loads((META/'manifest.json').read_text()))
    report=dict(complete=True,createdAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),scriptSha256=sha(__file__),preservedSha256=preserved,splitIntegrity=split_results,crossSplitD4Overlaps=overlaps,rawTrainingLabelMultiplicity=joint,playerProvenance=player_report,sourceSplitGames=[dict(source=k[0],split=k[1],complete=k[2],games=v) for k,v in sorted(source_split.items())],baselineDataPreprocessing=metadata['filters'],policyRule=provenance['rule'],seconds=time.perf_counter()-started,rawSourceSha256=raw_hashes,limits='Read-only diagnosis of recorded labels, not a strength or player-skill evaluation. No low-Elo threshold has been applied; historical Elo and identity joins remain unverified. Original caches and model defaults remain frozen.')
    for p,digest in preserved.items():assert sha(ROOT/p)==digest,p
    write(REPORT,report);progress('审计完成',1,1);r=json.loads((WORK/'progress.json').read_text());r['complete']=True;r['summary']='完整性、棋手溯源及训练标签多样性审计完成；原始数据未改';write(WORK/'progress.json',r);write(ROOT/'models/training-data-audit-2026-10-30-monitor.json',r)
    print(json.dumps(dict(complete=True,seconds=report['seconds'],splitIntegrity=split_results,overlaps=overlaps,playerProvenance={k:player_report[k] for k in ['distinctPlayerIndexValues','unknownAppearances','joinedConvertedWthorGames','unmatchedConvertedWthorGames']},multiplicity=joint),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
