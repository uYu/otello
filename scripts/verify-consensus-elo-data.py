#!/usr/bin/env python3
"""Verify v1.40 and materialize exact per-occurrence player provenance."""
import collections,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from training_consensus import actor_weight
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
WORK=ROOT/'.build/training/consensus-elo-v1.40'


def main():
    report=json.loads((ROOT/'models/v1.40-training-consensus-elo.json').read_text());protocol=json.loads((ROOT/'models/v1.40-training-consensus-elo-protocol.json').read_text())
    for name,digest in {**report['artifactSha256'],**report['preservedSha256'],**report['scriptSha256']}.items():assert sha(ROOT/name)==digest,name
    ledger={};gold=np.full(4448,1500.,np.float64);seen=np.zeros(4448,np.uint32);year=None;before=None;support=None;rated=0
    for line in (WORK/'game-elo.jsonl').open():
        row=json.loads(line);ledger[row['gameId']]=row
        if row['year']!=year:year=row['year'];before=gold.copy();support=seen.copy()
        a,b=row['blackPlayerIndex'],row['whitePlayerIndex']
        for color,player in [('black',a),('white',b)]:
            if player:
                assert abs(row[color+'TrainYearStartElo']-before[player])<1e-8
                assert row[color+'TrainYearStartGames']==support[player]
        if row['complete'] and row['split']=='train' and a and b and a!=b:
            expected=1/(1+10**((gold[b]-gold[a])/400));actual=1 if row['blackResult']>0 else .5 if row['blackResult']==0 else 0
            change=32*(actual-expected);gold[a]+=change;gold[b]-=change;seen[a]+=1;seen[b]+=1;rated+=1
    players=json.loads((WORK/'players.json').read_text())['players']
    for row in players:assert abs(row['trainOnlyElo']-gold[row['playerIndex']])<1e-8 and row['trainOnlyRatedGames']==seen[row['playerIndex']]
    assert abs(gold.sum()-1500*len(gold))<1e-6 and rated==report['elo']['trainOnlyRatedGames']
    # Recover the acting color at each placement from the actual transcript, including passes.
    n=0
    for _ in (WORK/'game-index.jsonl').open():n+=1
    colors=np.zeros((n,61),np.int8);black=np.zeros(n,np.uint16);white=np.zeros(n,np.uint16)
    elo_black=np.full(n,np.nan);elo_white=np.full(n,np.nan);games_black=np.zeros(n,np.uint32);games_white=np.zeros(n,np.uint32)
    for i,line in enumerate((ROOT/'.build/training/games.jsonl').open()):
        game=json.loads(line);turn=1;placed=0
        for move in game['moves']:
            if move>=0:colors[i,placed]=turn;placed+=1
            turn=-turn
        if game['id'] in ledger:
            row=ledger[game['id']];black[i]=row['blackPlayerIndex'];white[i]=row['whitePlayerIndex']
            elo_black[i]=np.nan if row['blackTrainYearStartElo'] is None else row['blackTrainYearStartElo']
            elo_white[i]=np.nan if row['whiteTrainYearStartElo'] is None else row['whiteTrainYearStartElo']
            games_black[i]=row['blackTrainYearStartGames'];games_white[i]=row['whiteTrainYearStartGames']
    occurrence=dict(np.load(WORK/'occurrences.npz'));g=occurrence['game_index'];keys=occurrence['keys'];occupied=(keys['own']|keys['opponent']).astype('<u8');placed=np.unpackbits(occupied.view(np.uint8).reshape(-1,8),axis=1).sum(1).astype(np.int16)-4
    color=colors[g,placed];assert np.isin(color,[-1,1]).all()
    actor=np.where(color==1,black[g],white[g]);opponent=np.where(color==1,white[g],black[g])
    elo=np.where(color==1,elo_black[g],elo_white[g]);support=np.where(color==1,games_black[g],games_white[g])
    expected=np.array([actor_weight(None if np.isnan(r) else r,int(s)) for r,s in zip(elo,support)])
    np.testing.assert_allclose(expected,occurrence['quality'],rtol=0,atol=1e-14)
    np.testing.assert_allclose(elo,occurrence['actor_train_year_start_elo'],rtol=1e-7,atol=1e-3,equal_nan=True)
    np.testing.assert_array_equal(support,occurrence['actor_train_year_start_games'])
    provenance=WORK/'occurrence-players.npz';assert not provenance.exists(),'Refusing to overwrite verified provenance'
    np.savez_compressed(provenance,game_index=g,placements=placed.astype(np.uint8),actor_color=color,actor_player_index=actor,opponent_player_index=opponent,actor_train_year_start_elo=elo,actor_train_year_start_games=support,source=occurrence['source'])
    data=dict(np.load(WORK/'consensus.npz'));unique=data['keys'];group=np.searchsorted(unique,keys);assert np.array_equal(unique[group],keys)
    # Independently reconstruct multiplicity on action orbits (not stabilizer-expanded squares).
    minimum=np.array([(int(x)&-int(x)).bit_length()-1 for x in occurrence['action_bits']],np.uint8)
    orbit_pairs=np.unique(group.astype(np.uint64)*64+minimum);orbit_count=np.bincount((orbit_pairs//64).astype(np.int64),minlength=len(unique))
    audit=json.loads((ROOT/'models/training-data-audit-2026-10-30.json').read_text())
    assert int((orbit_count>1).sum())==audit['rawTrainingLabelMultiplicity']['eligibleTrain']['multiActionOrbitPositions']==24857
    np.testing.assert_array_equal(np.bincount(group,minlength=len(unique)),data['occurrence_count'])
    for k,truth in enumerate([1,-1,0]):np.testing.assert_array_equal(np.bincount(group,weights=(occurrence['outcome']==truth),minlength=len(unique)),data['wdl_count'][:,k])
    offsets=data['policy_offsets'];sg=np.repeat(np.arange(len(unique)),np.diff(offsets).astype(np.int64));mass=data['policy_mass'];weighted=data['policy_quality_mass']
    np.testing.assert_array_equal(np.bincount(sg,weights=mass,minlength=len(unique)),data['occurrence_count'])
    np.testing.assert_allclose(np.bincount(sg,weights=weighted,minlength=len(unique)),data['policy_quality_total'],rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(np.bincount(group,weights=occurrence['quality'],minlength=len(unique)),data['policy_quality_total'],rtol=1e-12,atol=1e-10)
    result=dict(status='passed',independentTrainOnlyEloReplayGames=rated,allPreYearRatingsAndSupportVerified=True,zeroSumVerified=True,occurrencePolicyWeightsVerified=len(group),actorProvenanceWithPassesVerified=True,eligibleMultiActionOrbits=24857,rawWdlAndPolicyMassesVerified=True,
        matchedMillionMultiActionOrbits=int((orbit_count[data['matched_train_group']]>1).sum()),provenanceArtifact=str(provenance.relative_to(ROOT)),provenanceArtifactSha256=sha(provenance),scriptSha256=sha(__file__),reportSha256=sha(ROOT/'models/v1.40-training-consensus-elo.json'),knownActorOccurrences=int(((actor>0)&(occurrence['source']==0)).sum()),unknownOrNoIdOccurrences=int((actor==0).sum()))
    out=ROOT/'models/v1.40-training-consensus-elo-validation.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
