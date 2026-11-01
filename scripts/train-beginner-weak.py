#!/usr/bin/env python3
"""Fresh reduced-capacity/data candidates after the 58% beginner was rejected."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.build/training/positions.npz'
VARIANTS = [(8, 12000, 1), (4, 4096, 1), (4, 4096, 3), (2, 4096, 1),
            (2, 4096, 3), (4, 12000, 3), (8, 2048, 1)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    meta = json.loads(CACHE.with_suffix('.json').read_text())
    if sha(CACHE) != meta['cacheSha256']:
        raise ValueError('Cache checksum differs')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = np.load(CACHE)
    grid = np.arange(64).reshape(8, 8)
    permutations = np.stack([np.rot90(np.fliplr(grid) if mirror else grid, rotation).reshape(64)
                             for mirror in (False, True) for rotation in range(4)])
    for hidden, samples, epochs in VARIANTS:
        name = f'beginner-h{hidden}-n{samples}-e{epochs}-v1'
        output, report_path = ROOT/f'models/{name}.json', ROOT/f'models/{name}-training.json'
        if output.exists() or report_path.exists():
            raise ValueError('Preserve existing candidates: '+name)
        seed = 20261101
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(data['train_board']), samples, replace=False)
        x = data['train_board'][indices].astype(np.float32)*data['train_turn'][indices, None]
        y = np.sign(data['train_target'][indices]).astype(np.float32)
        net = nn.Sequential(nn.Linear(64, hidden), nn.Tanh(), nn.Linear(hidden, 1), nn.Tanh())
        initial_sha = hashlib.sha256(torch.cat([p.detach().flatten() for p in net.parameters()]).numpy().tobytes()).hexdigest()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.003)

        def predict(boards):
            with torch.no_grad():
                return np.mean([net(torch.from_numpy(boards[:, mapping].copy())).numpy().ravel()
                                for mapping in permutations], axis=0)

        def metric(split):
            b = data[split+'_board'].astype(np.float32)*data[split+'_turn'][:, None]
            target = np.sign(data[split+'_target'])
            prediction = predict(b)
            return dict(positions=len(b), outcomeMse=float(np.mean((prediction-target)**2)),
                        zeroPredictorMse=float(np.mean(target**2)),
                        nonDrawSignAccuracy=float(np.mean(np.sign(prediction[target != 0]) == target[target != 0])))

        started = time.perf_counter()
        history = [dict(epoch=0, validation=metric('validation'))]
        for epoch in range(1, epochs+1):
            order = rng.permutation(samples)
            for offset in range(0, samples, 250):
                batch = order[offset:offset+250]
                mapping = permutations[rng.integers(8)]
                loss = torch.mean((net(torch.from_numpy(x[batch][:, mapping].copy())).ravel()-torch.from_numpy(y[batch]))**2)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            history.append(dict(epoch=epoch, validation=metric('validation')))
        state = net.state_dict()
        model = dict(format='otello-beginner-v1', version=name, hidden=hidden,
                     parameters=hidden*66+1, activation='tanh', input='64 signed squares relative to side to move',
                     target='side-to-move final WDL', symmetry='D4 mean',
                     hiddenWeights=state['0.weight'].tolist(), hiddenBias=state['0.bias'].tolist(),
                     outputWeights=state['2.weight'][0].tolist(), outputBias=float(state['2.bias'][0]))
        output.write_text(json.dumps(model, separators=(',', ':'))+'\n')
        boards = data['test_board'][:12]; turns = data['test_turn'][:12]
        predictions = predict(boards.astype(np.float32)*turns[:, None])
        report = dict(version=name, parameters=model['parameters'], initialization='random; all parameters trained; no pretrained weights',
                      seed=seed, trainPositions=samples, epochs=epochs, batchSize=250, learningRate=0.003,
                      optimizer='Adam', checkpoint='fixed final epoch', initialWeightsSha256=initial_sha,
                      selectedIndicesSha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                      modelSha256=sha(output), sourceSha256=sha(Path(__file__)), cacheSha256=sha(CACHE), cacheMetadata=meta,
                      history=history, test=metric('test'), seconds=time.perf_counter()-started,
                      environment=dict(torch=torch.__version__, numpy=np.__version__, device='cpu', threads=2),
                      fixtures=[dict(board=b.astype(int).tolist(), turn=int(t), value=float(v)) for b,t,v in zip(boards,turns,predictions)],
                      inference='One-ply D4-mean model argmax, no book/noise/heuristics/search/terminal override',
                      dataNotices='data/README.md', status='candidate; rejected stronger tiny model stays separately preserved')
        report_path.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(model=name, test=report['test'])), flush=True)


if __name__ == '__main__':
    main()
