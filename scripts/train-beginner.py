#!/usr/bin/env python3
"""Train a deliberately small, independent one-ply player; never load v2 weights."""
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.build/training/positions.npz'
OUTPUT = ROOT / 'models/beginner8-v1.json'
REPORT = ROOT / 'models/beginner8-v1-training.json'
SEED = 20261101
SAMPLES = 12000
EPOCHS = 6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists() or REPORT.exists():
        raise ValueError('Preserve the trained model/report; use a new version for another run')
    metadata = json.loads(CACHE.with_suffix('.json').read_text())
    if sha(CACHE) != metadata['cacheSha256']:
        raise ValueError('Training cache checksum mismatch')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    data = np.load(CACHE)
    indices = rng.choice(len(data['train_board']), SAMPLES, replace=False)
    boards = data['train_board'][indices].astype(np.float32) * data['train_turn'][indices, None]
    targets = np.sign(data['train_target'][indices]).astype(np.float32)
    grid = np.arange(64).reshape(8, 8)
    permutations = np.stack([np.rot90(np.fliplr(grid) if mirror else grid, rotation).reshape(64)
                             for mirror in (False, True) for rotation in range(4)])
    # All 529 parameters are randomly initialized and jointly optimized on CPU.
    net = nn.Sequential(nn.Linear(64, 8), nn.Tanh(), nn.Linear(8, 1), nn.Tanh())
    initial_weights = torch.cat([p.detach().flatten() for p in net.parameters()]).numpy().tobytes()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)

    def predict(x):
        with torch.no_grad():
            return np.mean([net(torch.from_numpy(x[:, permutation].copy())).numpy().ravel()
                            for permutation in permutations], axis=0)

    def metric(split):
        x = data[split + '_board'].astype(np.float32) * data[split + '_turn'][:, None]
        y = np.sign(data[split + '_target'])
        prediction = predict(x)
        return dict(positions=len(y), outcomeMse=float(np.mean((prediction-y)**2)),
                    zeroPredictorMse=float(np.mean(y**2)),
                    nonDrawSignAccuracy=float(np.mean(np.sign(prediction[y != 0]) == y[y != 0])))

    history = [dict(epoch=0, validation=metric('validation'))]
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for batch in rng.permutation(SAMPLES).reshape(-1, 250):
            symmetry = rng.integers(8)
            x = torch.from_numpy(boards[batch][:, permutations[symmetry]].copy())
            y = torch.from_numpy(targets[batch])
            loss = torch.mean((net(x).ravel() - y)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        row = dict(epoch=epoch, trainMse=float(np.mean(losses)), validation=metric('validation'))
        history.append(row)
        print(json.dumps(row), flush=True)
    state = net.state_dict()
    model = dict(format='otello-beginner-v1', version='beginner8-v1', hidden=8, parameters=529,
                 input='64 signed squares relative to the evaluated side; empty=0, own=1, opponent=-1',
                 target='side-to-move final W/D/L (+1/0/-1)', activation='tanh',
                 symmetry='mean of all eight D4 predictions',
                 hiddenWeights=state['0.weight'].tolist(), hiddenBias=state['0.bias'].tolist(),
                 outputWeights=state['2.weight'][0].tolist(), outputBias=float(state['2.bias'][0]))
    OUTPUT.write_text(json.dumps(model, separators=(',', ':')) + '\n')
    # Cross-runtime fixtures exercise all exported parameters, not just the chosen move.
    fixtures = []
    for i in range(12):
        board = data['test_board'][i].astype(np.float32)
        turn = int(data['test_turn'][i])
        fixtures.append(dict(board=board.astype(int).tolist(), turn=turn,
                             value=float(predict((board * turn)[None, :])[0])))
    report = dict(version=model['version'], scope='beginner only; approved v2 baseline unchanged',
                  initialization='random; no pretrained weights; all parameters trainable',
                  initialWeightsSha256=hashlib.sha256(initial_weights).hexdigest(),
                  modelSha256=sha(OUTPUT), parameters=sum(p.numel() for p in net.parameters()),
                  seed=SEED, trainPositions=SAMPLES, epochs=EPOCHS, batchSize=250,
                  learningRate=0.003, optimizer='Adam', checkpoint='fixed final epoch; no test selection',
                  cache=str(CACHE.relative_to(ROOT)), cacheSha256=sha(CACHE), cacheMetadata=metadata,
                  selectedIndicesSha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                  sourceSha256=sha(Path(__file__)), dataNotices='data/README.md',
                  history=history, test=metric('test'), fixtures=fixtures,
                  seconds=time.perf_counter()-started,
                  environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, device='cpu', threads=2),
                  inference='Score each legal successor once with D4-averaged network; deterministic argmax; no search, noise, book, heuristics or terminal override',
                  limits='Limited-capacity training is not evidence of beginner suitability; requires playtesting and separate strength screening.')
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(model=str(OUTPUT), test=report['test'], seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
