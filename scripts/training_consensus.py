"""Internal Elo and D4-aligned empirical targets; no frozen-cache mutation."""
from dataclasses import dataclass, field
import math
import numpy as np

KEY=np.dtype([('own','<u8'),('opponent','<u8')])
GRID=np.arange(64).reshape(8,8)
PERMS=np.stack([np.rot90(np.fliplr(GRID) if mirror else GRID,r).reshape(64) for mirror in [False,True] for r in range(4)])
INVERSE=np.argsort(PERMS,axis=1)


def expected_score(a,b,scale=400.):
    difference=(b-a)/scale
    if difference>=0:
        t=10.**(-difference);return t/(1+t)
    return 1/(1+10.**difference)


@dataclass
class EloPool:
    size:int
    initial:float=1500.
    k:float=32.
    scale:float=400.
    def __post_init__(self):
        self.rating=np.full(self.size,self.initial,np.float64)
        self.games=np.zeros(self.size,np.uint32)
    def valid(self,a,b):return 0<a<self.size and 0<b<self.size and a!=b
    def update(self,a,b,score):
        if score not in (0.,.5,1.):raise ValueError('Expected win/draw/loss score')
        if not self.valid(a,b):return None
        expected=expected_score(self.rating[a],self.rating[b],self.scale)
        delta=self.k*(score-expected)
        self.rating[a]+=delta;self.rating[b]-=delta
        self.games[a]+=1;self.games[b]+=1
        return expected,delta


def actor_weight(rating,prior_games,initial=1500.,scale=400.,warmup=20):
    """Bounded soft quality prior. Unknown/new players get neutral weight .5."""
    if rating is None:return .5
    confidence=min(float(prior_games)/warmup,1.)
    return .5+confidence*(expected_score(rating,initial,scale)-.5)


def rate_records(records,size,initial=1500.,k=32.,scale=400.):
    """Only train results update the training pool; expose pre-year and pre-game values."""
    full=EloPool(size,initial,k,scale);train=EloPool(size,initial,k,scale)
    current=None;snapshots={};ledger=[]
    for row in sorted(records,key=lambda r:(r['year'],r['rawRecordIndex'],r['gameId'])):
        year=row['year']
        if year!=current:
            current=year;snapshots[year]=(train.rating.copy(),train.games.copy())
        before,n_before=snapshots[year];a,b=row['blackPlayerIndex'],row['whitePlayerIndex']
        data={**row}
        for color,player in [('black',a),('white',b)]:
            known=0<player<size
            data[color+'FullPreGameElo']=float(full.rating[player]) if known else None
            data[color+'TrainPreGameElo']=float(train.rating[player]) if known else None
            data[color+'TrainYearStartElo']=float(before[player]) if known else None
            data[color+'TrainYearStartGames']=int(n_before[player]) if known else 0
        data['rated']=bool(row['complete'] and full.valid(a,b))
        if data['rated']:
            score=(row['blackResult']>0)+.5*(row['blackResult']==0)
            expected,delta=full.update(a,b,score);data.update(blackExpectedScore=expected,blackRatingChange=delta)
            if row['split']=='train':train.update(a,b,score)
        ledger.append(data)
    return ledger,full,train,snapshots


def canonical(board,turn,moves=None):
    """Return canonical relative bitboards and *all* equivalent aligned action bits."""
    relative=(board*turn[:,None]).astype(np.int8)
    best=np.empty(len(board),KEY);best['own']=np.iinfo(np.uint64).max;best['opponent']=np.iinfo(np.uint64).max
    action=np.zeros(len(board),np.uint64)
    for permutation,inverse in zip(PERMS,INVERSE):
        transformed=relative[:,permutation]
        own=np.packbits(transformed==1,axis=1,bitorder='little').copy().view('<u8').reshape(-1)
        opp=np.packbits(transformed==-1,axis=1,bitorder='little').copy().view('<u8').reshape(-1)
        better=(own<best['own'])|((own==best['own'])&(opp<best['opponent']))
        equal=(own==best['own'])&(opp==best['opponent'])
        if moves is not None:
            aligned=np.left_shift(np.uint64(1),inverse[moves].astype(np.uint64))
            action=np.where(better,aligned,np.where(equal,action|aligned,action))
        best['own'][better]=own[better];best['opponent'][better]=opp[better]
    return best,action


def boards_from_keys(keys):
    def bits(name):return np.unpackbits(keys[name].astype('<u8').view(np.uint8).reshape(-1,8),axis=1,bitorder='little').astype(np.int8)
    return bits('own')-bits('opponent')


def aggregate(keys,action_bits,outcomes,quality,source):
    unique,inverse,count=np.unique(keys,return_inverse=True,return_counts=True)
    n=len(unique);wdl=np.zeros((n,3),np.uint32)
    category=np.where(outcomes>0,0,np.where(outcomes<0,1,2))
    np.add.at(wdl,(inverse,category),1)
    source_count=np.zeros((n,2),np.uint32);np.add.at(source_count,(inverse,source),1)
    # Sum stabilizer-equivalent actions as an invariant probability distribution.
    orbit=np.array([int(x).bit_count() for x in action_bits],np.uint8)
    assert (orbit>0).all()
    pairs=[];masses=[];weighted=[]
    for action in range(64):
        present=((action_bits>>np.uint64(action))&np.uint64(1)).astype(bool)
        if present.any():
            pairs.append(inverse[present].astype(np.uint64)*64+action)
            masses.append(1./orbit[present]);weighted.append(quality[present]/orbit[present])
    pairs=np.concatenate(pairs);mass=np.concatenate(masses);qmass=np.concatenate(weighted)
    pair,back=np.unique(pairs,return_inverse=True)
    mass=np.bincount(back,weights=mass);qmass=np.bincount(back,weights=qmass)
    group=(pair//64).astype(np.uint32);actions=(pair%64).astype(np.uint8)
    offsets=np.concatenate([[0],np.cumsum(np.bincount(group,minlength=n))]).astype(np.uint64)
    total_quality=np.bincount(inverse,weights=quality,minlength=n)
    assert np.array_equal(wdl.sum(1),count) and np.array_equal(source_count.sum(1),count)
    assert np.array_equal(np.bincount(group,weights=mass,minlength=n),count)
    np.testing.assert_allclose(np.bincount(group,weights=qmass,minlength=n),total_quality,rtol=1e-12,atol=1e-10)
    return dict(keys=unique,occurrence_count=count.astype(np.uint32),wdl_count=wdl,source_count=source_count,
                policy_offsets=offsets,policy_action=actions,policy_mass=mass,policy_quality_mass=qmass,
                policy_quality_total=total_quality)
