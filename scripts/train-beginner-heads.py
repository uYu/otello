#!/usr/bin/env python3
"""Compare direct spatial policy and 32-unit value with the tiny beginner model."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.build/training/positions.npz'
LABELS = ROOT / '.build/training/beginner/labels.npz'
OUTPUT = ROOT / 'models/beginner-board-heads-v1.json'
REPORT = ROOT / 'models/beginner-board-heads-v1-training.json'
SEED, SAMPLES, EPOCHS = 20261101, 12000, 6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists() or REPORT.exists():
        raise ValueError('Choose a new version; preserve existing candidates')
    cache_meta = json.loads(CACHE.with_suffix('.json').read_text())
    label_meta = json.loads(LABELS.with_suffix('.json').read_text())
    if sha(CACHE) != cache_meta['cacheSha256'] or sha(CACHE) != label_meta['positionsSha256'] or sha(LABELS) != label_meta['cacheSha256']:
        raise ValueError('Cache/label checksums differ')
    tiny = json.loads((ROOT / 'models/beginner8-v1-training.json').read_text())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    data, labels = np.load(CACHE), np.load(LABELS)
    indices = rng.choice(len(data['train_board']), SAMPLES, replace=False)
    indices_sha = hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()
    assert indices_sha == tiny['selectedIndicesSha256'] and sha(CACHE) == tiny['cacheSha256']
    grid = np.arange(64).reshape(8, 8)
    permutations = np.stack([np.rot90(np.fliplr(grid) if mirror else grid, rotation).reshape(64)
                             for mirror in (False, True) for rotation in range(4)])
    inverses = np.argsort(permutations, axis=1)
    policy = nn.Linear(192, 64)
    value = nn.Sequential(nn.Linear(192, 32), nn.Tanh(), nn.Linear(32, 1), nn.Tanh())
    parameters = list(policy.parameters()) + list(value.parameters())
    initial_sha = hashlib.sha256(torch.cat([p.detach().flatten() for p in parameters]).numpy().tobytes()).hexdigest()
    optimizer = torch.optim.Adam(parameters, lr=0.003)

    def features(boards):
        return np.stack((boards == 1, boards == -1, boards == 0), axis=2).astype(np.float32).reshape(-1, 192)

    def predict(boards):
        policies, values = [], []
        with torch.no_grad():
            for mapping, inverse in zip(permutations, inverses):
                x = torch.from_numpy(features(boards[:, mapping]))
                policies.append(policy(x).numpy()[:, inverse])
                values.append(value(x).numpy().ravel())
        return np.mean(policies, axis=0), np.mean(values, axis=0)

    def metrics(split):
        boards = data[split+'_board'] * data[split+'_turn'][:, None]
        logits, values = predict(boards)
        legal = labels[split+'_legal'][:, :64]
        moves = labels[split+'_move']
        logits = np.where(legal, logits, -np.inf)
        logits -= logits.max(axis=1, keepdims=True)
        ce = np.log(np.exp(logits).sum(axis=1)) - logits[np.arange(len(moves)), moves]
        target = np.sign(data[split+'_target'])
        return dict(positions=len(moves), policyCrossEntropy=float(ce.mean()),
                    policyTop1=float(np.mean(logits.argmax(axis=1) == moves)),
                    uniformCrossEntropy=float(np.log(legal.sum(axis=1)).mean()),
                    uniformTop1=float((1/legal.sum(axis=1)).mean()),
                    outcomeMse=float(np.mean((values-target)**2)),
                    zeroPredictorMse=float(np.mean(target**2)),
                    nonDrawSignAccuracy=float(np.mean(np.sign(values[target != 0]) == target[target != 0])))

    boards = data['train_board'][indices] * data['train_turn'][indices, None]
    targets = np.sign(data['train_target'][indices]).astype(np.float32)
    moves = labels['train_move'][indices]
    legal = labels['train_legal'][indices, :64]
    history = [dict(epoch=0, validation=metrics('validation'))]
    started = time.perf_counter()
    for epoch in range(1, EPOCHS+1):
        for batch in rng.permutation(SAMPLES).reshape(-1, 250):
            symmetry = rng.integers(8)
            mapping, inverse = permutations[symmetry], inverses[symmetry]
            x = torch.from_numpy(features(boards[batch][:, mapping]))
            mask = torch.from_numpy(legal[batch][:, mapping])
            logits = policy(x).masked_fill(~mask, -torch.inf)
            policy_loss = nn.functional.cross_entropy(logits, torch.from_numpy(inverse[moves[batch]].astype(np.int64)))
            value_loss = torch.mean((value(x).ravel() - torch.from_numpy(targets[batch]))**2)
            optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            optimizer.step()
        row = dict(epoch=epoch, validation=metrics('validation'))
        history.append(row)
        print(json.dumps(row), flush=True)
    state = value.state_dict()
    model = dict(format='otello-beginner-board-heads-v1', version='beginner-board-heads-v1',
                 input='8x8x3 interleaved own/opponent/empty relative to side to move',
                 policyArchitecture='192 -> 64 linear logits', valueArchitecture='192 -> 32 tanh -> 1 tanh',
                 symmetry='D4 mean; policy coordinates restored before averaging',
                 policyWeights=policy.weight.detach().tolist(), policyBias=policy.bias.detach().tolist(),
                 hiddenWeights=state['0.weight'].tolist(), hiddenBias=state['0.bias'].tolist(),
                 outputWeights=state['2.weight'][0].tolist(), outputBias=float(state['2.bias'][0]))
    OUTPUT.write_text(json.dumps(model, separators=(',', ':')) + '\n')
    fixture_boards = data['test_board'][:12]
    fixture_turns = data['test_turn'][:12]
    logits, values = predict(fixture_boards * fixture_turns[:, None])
    fixtures = [dict(board=b.astype(int).tolist(), turn=int(t), logits=p.tolist(), value=float(v))
                for b, t, p, v in zip(fixture_boards, fixture_turns, logits, values)]
    report = dict(version=model['version'], status='comparison candidate; no default promotion',
                  initialization='all parameters from random initialization; no pretrained weights or frozen parameters',
                  initialWeightsSha256=initial_sha, modelSha256=sha(OUTPUT), sourceSha256=sha(Path(__file__)),
                  parameters=dict(policy=sum(p.numel() for p in policy.parameters()), value=sum(p.numel() for p in value.parameters())),
                  independentHeads=True, loss='masked recorded-move CE + final WDL MSE; no shared trainable layers',
                  seed=SEED, trainPositions=SAMPLES, epochs=EPOCHS, batchSize=250, learningRate=0.003,
                  optimizer='Adam', checkpoint='fixed final epoch; no test selection', selectedIndicesSha256=indices_sha,
                  cacheSha256=sha(CACHE), labels=label_meta, dataNotices='data/README.md',
                  history=history, test=metrics('test'), fixtures=fixtures, seconds=time.perf_counter()-started,
                  environment=dict(torch=torch.__version__, numpy=np.__version__, threads=2, device='cpu'),
                  inference='Compare direct masked policy argmax vs one-ply value argmax separately; no book or MCTS',
                  limits='Same 12000 sampled boards and six epochs as tiny value model; parameter count and objectives differ; not equal FLOPs.')
    REPORT.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(model=str(OUTPUT), test=report['test'], parameters=report['parameters'])), flush=True)


if __name__ == '__main__':
    main()
