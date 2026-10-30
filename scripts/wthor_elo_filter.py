"""Whole-game threshold decisions using frozen train-only, year-start ratings."""
import numpy as np


def both_below(black,white,threshold):
    a=np.asarray(black,float);b=np.asarray(white,float)
    return np.isfinite(a)&np.isfinite(b)&(a<threshold)&(b<threshold)


def choose_candidate(rows,min_game_retention=.6,min_unique_retention=.6):
    eligible=[r for r in rows if r['threshold'] is not None and r['gameRetentionVsWthor']>=min_game_retention and r['uniqueRetentionVsWthor']>=min_unique_retention and r['theoryComparableGames']>0]
    if not eligible:raise ValueError('No candidate meets the declared retention floors')
    return min(eligible,key=lambda r:(r['recordedVs24EmptyWdlDisagreementRate'],-r['uniquePositions']))
