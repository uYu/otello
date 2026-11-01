#!/usr/bin/env python3
"""Fit direct-board novice imitation from random weights, retaining training budgets."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'.build/training/beginner-low-elo-v1'
DATA_REPORT = ROOT/'models/beginner-low-elo-v1-data.json'
CHECKPOINTS = (4, 16, 64, 117)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    meta = json.loads(DATA_REPORT.read_text())
    cache = WORK/'positions.npz'
    if sha(cache) != meta['cacheSha256']:
        raise ValueError('Low-Elo cache checksum differs')
    for step in CHECKPOINTS:
        if (ROOT/f'models/beginner-low-elo-policy-s{step}-v1.json').exists():
            raise ValueError('Preserve existing training candidates')
    data = np.load(cache)
    torch.set_num_threads(2); torch.manual_seed(20261101)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(20261101)
    grid = np.arange(64).reshape(8, 8)
    perms = np.stack([np.rot90(np.fliplr(grid) if mirror else grid, rotation).reshape(64)
                      for mirror in (False, True) for rotation in range(4)])
    inverse = np.argsort(perms, axis=1)
    net = nn.Linear(192, 64)
    initial_sha = hashlib.sha256(torch.cat([p.detach().flatten() for p in net.parameters()]).numpy().tobytes()).hexdigest()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)

    def features(board):
        return np.stack((board == 1, board == -1, board == 0), axis=2).astype(np.float32).reshape(-1,192)

    def predict(board):
        with torch.no_grad():
            return np.mean([net(torch.from_numpy(features(board[:, p]))).numpy()[:, inv]
                            for p,inv in zip(perms,inverse)], axis=0)

    def metric(split):
        logits = predict(data[split+'_board'])
        mask = data[split+'_legal']; moves = data[split+'_move']
        logits = np.where(mask, logits, -np.inf); logits -= logits.max(1,keepdims=True)
        ce = np.log(np.exp(logits).sum(1))-logits[np.arange(len(moves)), moves]
        return dict(positions=len(moves), crossEntropy=float(ce.mean()), top1=float((logits.argmax(1)==moves).mean()),
                    uniformCrossEntropy=float(np.log(mask.sum(1)).mean()), uniformTop1=float((1/mask.sum(1)).mean()))

    initial_metric = metric('validation')
    order = rng.permutation(len(data['train_board']))
    seen = 0; started = time.perf_counter()
    for step, offset in enumerate(range(0,len(order),250),1):
        batch = order[offset:offset+250]; symmetry = rng.integers(8)
        p,inv = perms[symmetry],inverse[symmetry]
        x = torch.from_numpy(features(data['train_board'][batch][:,p]))
        mask = torch.from_numpy(data['train_legal'][batch][:,p])
        moves = torch.from_numpy(inv[data['train_move'][batch]].astype(np.int64))
        loss = nn.functional.cross_entropy(net(x).masked_fill(~mask,-torch.inf),moves)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); seen += len(batch)
        if step not in CHECKPOINTS:
            continue
        name = f'beginner-low-elo-policy-s{step}-v1'
        output = ROOT/f'models/{name}.json'
        model = dict(format='otello-beginner-policy-v1', version=name, parameters=12352,
                     architecture='8x8x3 interleaved own/opponent/empty -> 8x8 linear logits',
                     policyWeights=net.weight.detach().tolist(), policyBias=net.bias.detach().tolist(),
                     symmetry='D4 mean with output coordinates restored; legal masked argmax')
        output.write_text(json.dumps(model,separators=(',',':'))+'\n')
        fixtures = []
        boards = data['test_board'][:12]; predictions = predict(boards)
        for board,logits in zip(boards,predictions):
            fixtures.append(dict(board=board.astype(int).tolist(),turn=1,logits=logits.tolist()))
        report = dict(version=name, initialization='random; all 12352 parameters trainable; no pretrained weights or frozen parameters',
                      initialWeightsSha256=initial_sha, modelSha256=sha(output), parameters=12352, seed=20261101,
                      dataReport=str(DATA_REPORT.relative_to(ROOT)), dataReportSha256=sha(DATA_REPORT),
                      cacheSha256=sha(cache), actorEloThreshold=1350, minPriorTrainGames=20,
                      steps=step, seenPositions=seen, totalTrainPool=len(order), batchSize=250, learningRate=0.003,
                      optimizer='Adam', loss='masked CE on low-rated actors recorded moves only; no value/engine teacher',
                      initialValidation=initial_metric, validation=metric('validation'), test=metric('test'),
                      fixtures=fixtures, sourceSha256=sha(Path(__file__)), seconds=time.perf_counter()-started,
                      environment=dict(torch=torch.__version__, numpy=np.__version__, device='cpu', threads=2),
                      checkpoint='fixed training budgets, to be screened for low playing strength; no heldout accuracy optimization',
                      inference='Pure deterministic policy argmax over legal moves; no book, MCTS, value, heuristics or random mixing',
                      dataNotices='data/README.md', limits='Low internal Elo alone does not establish beginner difficulty; match screening is required.')
        output.with_name(name+'-training.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(model=name,seenPositions=seen,validation=report['validation'])),flush=True)


if __name__ == '__main__':
    main()
