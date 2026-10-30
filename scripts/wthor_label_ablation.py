"""Select the first WTHOR observation without changing the canonical position set."""
import hashlib
import numpy as np


def fingerprint(array):
    h=hashlib.sha256();h.update(str(array.shape).encode());h.update(array.dtype.str.encode());h.update(array.tobytes())
    return h.hexdigest()


def first_labels(occurrences,expected_keys):
    selected=np.flatnonzero(occurrences['source']==0)
    keys,first=np.unique(occurrences['keys'][selected],return_index=True)
    if not np.array_equal(keys,expected_keys):raise ValueError('Hard/soft position sets differ')
    indices=selected[first];outcome=occurrences['outcome'][indices]
    action=np.array([(int(x)&-int(x)).bit_length()-1 for x in occurrences['action_bits'][indices]],np.int64)
    if not np.isin(outcome,[-1,0,1]).all() or not ((action>=0)&(action<64)).all():raise ValueError('Invalid first observation')
    category=np.where(outcome>0,0,np.where(outcome<0,1,2))
    return dict(first_occurrence=indices.astype(np.uint32),first_game_index=occurrences['game_index'][indices],value=outcome.astype(np.float32),wdl=np.eye(3,dtype=np.float32)[category],action=action)


def policy_targets(data,hard,mode):
    n=len(data['keys']);result=np.zeros((n,64),np.float32)
    if mode=='hard':result[np.arange(n),hard['action']]=1
    elif mode=='soft':
        group=np.repeat(np.arange(n),np.diff(data['policy_offsets']).astype(np.int64));result[group,data['policy_action']]=data['policy_probability']
    else:raise ValueError(mode)
    return result


def parameter_fingerprint(net):
    h=hashlib.sha256()
    for name,p in net.named_parameters():
        a=p.detach().cpu().numpy();h.update(name.encode());h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
    return h.hexdigest()
