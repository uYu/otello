#!/usr/bin/env python3
"""Small reproducible Othello NNUE value baseline; needs numpy and torch."""
import argparse
from collections import Counter, defaultdict
import ctypes
import hashlib
import json
from pathlib import Path
import random
import struct
import time
import numpy as np
import torch
from torch import nn

SPLITS = ('train', 'validation', 'test')


class ValueNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.transform = nn.Linear(128, hidden)
        self.head = nn.Linear(hidden, 1, bias=False)

    def forward(self, features, turn):
        swapped = torch.cat((features[:, 64:], features[:, :64]), dim=1)
        black = self.transform(features).clamp(0, 1)
        white = self.transform(swapped).clamp(0, 1)
        return torch.tanh(self.head(black - white).squeeze(1) * turn * 0.5)


def symmetries():
    grid = np.arange(64).reshape(8, 8)
    return np.stack([np.rot90(np.fliplr(grid) if mirror else grid, rotation).reshape(64)
                     for mirror in (False, True) for rotation in range(4)])


def prepare(args):
    """Sample whole games first; augment only after opening-group train/test split."""
    rng = random.Random(args.seed)
    strata, counts = defaultdict(list), Counter()
    holdout = dict(np.load(args.holdout_cache)) if args.holdout_cache else None
    holdout_meta = json.loads(args.holdout_cache.with_suffix('.json').read_text()) if holdout else None
    # Balance source representation without taking only the first Logistello branch.
    quota = {'train': max(1, args.max_games * 4 // 10),
             'validation': max(1, args.max_games // 20), 'test': max(1, args.max_games // 20)}
    digest = hashlib.sha256()
    with args.games.open('rb') as stream:
        for line in stream:
            digest.update(line)
            game = json.loads(line)
            if not game['complete']:
                continue
            if holdout is not None and game['split'] != 'train':
                continue
            key = (game['source'], game['split'])
            counts[key] += 1
            reservoir = strata[key]
            if (args.all_training_games and key[1] == 'train') or len(reservoir) < quota[key[1]]:
                reservoir.append(game)
            else:
                index = rng.randrange(counts[key])
                if index < len(reservoir):
                    reservoir[index] = game
    lib = ctypes.CDLL(str(args.library.resolve()))
    cells_type, out_type = ctypes.c_int * 64, ctypes.c_int * 67
    lib.ot_play.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                            ctypes.POINTER(ctypes.c_int)]
    lib.ot_play.restype = ctypes.c_int
    permutations = symmetries()
    seen, examples, selected_games = set(), {}, Counter()
    if holdout is not None:
        if holdout_meta['inputSha256'] != digest.hexdigest() or holdout_meta['seed'] != args.seed:
            raise ValueError('Frozen holdout uses a different data snapshot or seed')
        for split in ('test', 'validation'):
            rows = []
            for board, turn, target in zip(*(holdout[split + '_' + k] for k in ('board', 'turn', 'target'))):
                canonical = min((board * turn).astype(np.int8)[p].tobytes() for p in permutations)
                if canonical in seen:
                    raise ValueError('Frozen holdouts contain a symmetric duplicate')
                seen.add(canonical)
                rows.append((board, turn, target))
            examples[split] = rows
        selected_games.update({k: v for k, v in holdout_meta['games'].items() if not k.endswith(':train')})
    # Held-out ownership wins if two different opening groups transpose later.
    for split in ('test', 'validation', 'train'):
        if holdout is not None and split != 'train':
            continue
        reservoir, visited = [], 0
        cap = max(1, int(args.max_positions * (0.8 if split == 'train' else 0.1)))
        if split == 'train' and args.train_positions:
            cap = args.train_positions
        games = sorted([g for key, group in strata.items() if key[1] == split for g in group],
                       key=lambda g: g['id'])
        for game_index, game in enumerate(games):
            cells = cells_type()
            cells[28] = cells[35] = 1
            cells[27] = cells[36] = -1
            turn, ply = 1, 0
            selected_games[game['source'] + ':' + split] += 1
            for move in game['moves']:
                if 8 <= ply <= 56 and ply % 3 == 0 and move >= 0:
                    board = np.array(cells, dtype=np.int8)
                    relative = board * turn
                    canonical = min(relative[p].tobytes() for p in permutations)
                    if canonical not in seen:
                        seen.add(canonical)
                        sample = (board, turn, game['blackResult'] / 64 * turn)
                        visited += 1
                        if len(reservoir) < cap:
                            reservoir.append(sample)
                        else:
                            index = rng.randrange(visited)
                            if index < cap:
                                reservoir[index] = sample
                out = out_type()
                if lib.ot_play(cells, turn, move, out) != 1:
                    raise ValueError('Illegal prepared transcript: ' + game['id'])
                cells = cells_type(*out[:64])
                turn = out[64]
                ply += move >= 0
            if (game_index + 1) % 2000 == 0:
                print(json.dumps(dict(preparing=split, games=game_index + 1, totalGames=len(games),
                                      retainedPositions=len(reservoir), uniqueCandidates=visited)), flush=True)
        if not reservoir:
            raise ValueError('No positions in ' + split)
        examples[split] = reservoir
    data = {}
    for split, rows in examples.items():
        data[split + '_board'] = np.stack([r[0] for r in rows])
        data[split + '_turn'] = np.array([r[1] for r in rows], dtype=np.float32)
        data[split + '_target'] = np.array([r[2] for r in rows], dtype=np.float32)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache, **data)
    report = dict(seed=args.seed, inputSha256=digest.hexdigest(), games=dict(selected_games),
                  positions={k: len(v) for k, v in examples.items()},
                  filters='complete games; placements 8..56, stride 3; 8-symmetry/color-relative position dedup; test/validation ownership before train',
                  split='native converter canonical first 8 moves; no move-wise split',
                  target='recorded final black disc difference / 64, from side-to-move view',
                  maxGames=args.max_games, maxPositions=args.max_positions,
                  allTrainingGames=args.all_training_games, trainPositions=args.train_positions,
                  frozenHoldout=(dict(path=str(args.holdout_cache),
                                     sha256=hashlib.sha256(args.holdout_cache.read_bytes()).hexdigest())
                                 if args.holdout_cache else None))
    report['cacheSha256'] = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    args.cache.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'prepared': report}), flush=True)
    return data


def features(board):
    return torch.cat(((board == 1).float(), (board == -1).float()), dim=1)


def quantized(net):
    def convert(tensor, dtype):
        scaled = np.rint(tensor.detach().cpu().numpy() * 256)
        limits = np.iinfo(dtype)
        if not np.isfinite(scaled).all() or scaled.min() < limits.min or scaled.max() > limits.max:
            raise ValueError('Weights exceed quantized format')
        return scaled.astype(dtype)
    return (convert(net.transform.weight.T, '<i2'), convert(net.transform.bias, '<i4'),
            convert(net.head.weight.reshape(-1), '<i2'))


def quantized_predict(weights, board, turn):
    w, bias, head = [x.astype(np.int64) for x in weights]
    x = np.concatenate((board == 1, board == -1), axis=1).astype(np.int64)
    swapped = np.concatenate((x[:, 64:], x[:, :64]), axis=1)
    b = np.clip(x @ w + bias, 0, 256)
    other = np.clip(swapped @ w + bias, 0, 256)
    return np.tanh(((b - other) @ head) * turn / (2 * 256 * 256))


def metric(prediction, target):
    error = prediction - target
    return dict(mse=float(np.mean(error * error)), maeDiscs=float(np.mean(np.abs(error)) * 64))


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.prepare or not args.cache.exists():
        data = prepare(args)
    else:
        meta = json.loads(args.cache.with_suffix('.json').read_text())
        if meta['seed'] != args.seed:
            raise ValueError('Cache seed differs; use --prepare')
        if args.train_positions and meta.get('trainPositions') != args.train_positions:
            raise ValueError('Cache training size differs; use --prepare')
        if meta.get('cacheSha256') and meta['cacheSha256'] != hashlib.sha256(args.cache.read_bytes()).hexdigest():
            raise ValueError('Cache checksum mismatch')
        data = dict(np.load(args.cache))
    net = ValueNet(args.hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=0.001)
    boards = torch.from_numpy(data['train_board'])
    turns = torch.from_numpy(data['train_turn'])
    target = torch.from_numpy(data['train_target'])
    val_x = features(torch.from_numpy(data['validation_board']))
    val_turn = torch.from_numpy(data['validation_turn'])
    val_target = data['validation_target']
    permutations = torch.from_numpy(symmetries())
    best, best_state, history = float('inf'), None, []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        net.train()
        for indices in torch.randperm(len(boards)).split(args.batch_size):
            batch = boards[indices]
            # Random D4 augmentation applies only to training positions.
            selection = permutations[torch.randint(8, (len(indices),))]
            batch = torch.gather(batch, 1, selection)
            pred = net(features(batch), turns[indices])
            loss = torch.mean((pred - target[indices]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(val_x, val_turn).numpy()
        score = metric(pred, val_target)
        history.append(dict(epoch=epoch + 1, **score))
        if score['mse'] < best:
            best = score['mse']
            best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
        print(json.dumps(history[-1]), flush=True)
    net.load_state_dict(best_state)
    net.eval()
    weights = quantized(net)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    binary = struct.pack('<8sIIII', b'OTNNUE1\0', 128, args.hidden, 256, 256)
    binary += b''.join(x.tobytes() for x in weights)
    args.output.write_bytes(binary)
    torch.save(net.state_dict(), args.output.with_suffix('.pt'))
    report = dict(architecture=f'128->{args.hidden}->1, shared black/white transform and head; clipped ReLU; tanh',
                  parameters=sum(x.numel() for x in net.parameters()), bytes=len(binary),
                  sha256=hashlib.sha256(binary).hexdigest(), seed=args.seed,
                  bestEpoch=min(history, key=lambda row: row['mse'])['epoch'],
                  torchVersion=torch.__version__, numpyVersion=np.__version__,
                  trainingScriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dataset=json.loads(args.cache.with_suffix('.json').read_text()),
                  epochs=args.epochs, batchSize=args.batch_size, learningRate=args.learning_rate,
                  seconds=time.perf_counter() - started, history=history, metrics={})
    for split in ('validation', 'test'):
        board, turn, truth = [data[split + '_' + key] for key in ('board', 'turn', 'target')]
        with torch.no_grad():
            fp = net(features(torch.from_numpy(board)), torch.from_numpy(turn)).numpy()
        qp = quantized_predict(weights, board, turn)
        report['metrics'][split] = dict(float=metric(fp, truth), quantized=metric(qp, truth),
                                        zeroBaseline=metric(np.zeros_like(truth), truth),
                                        maxQuantizationError=float(np.max(np.abs(fp - qp))))
    # Verify exported bytes through the actual C++ library and caller-owned C buffers.
    rows = data['test_board'][:args.parity_positions]
    turn = data['test_turn'][:args.parity_positions]
    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
    lib.ot_nnue_load.restype = ctypes.c_void_p
    lib.ot_nnue_evaluate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_int)]
    lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]
    lib.ot_nnue_error.restype = ctypes.c_char_p
    handle = lib.ot_nnue_load(str(args.output.resolve()).encode())
    if not handle:
        raise ValueError(lib.ot_nnue_error().decode())
    actual = []
    try:
        for board, actor in zip(rows, turn):
            cells = (ctypes.c_int * 64)(*map(int, board))
            score = ctypes.c_int()
            if lib.ot_nnue_evaluate(handle, cells, int(actor), ctypes.byref(score)) != 1:
                raise ValueError(lib.ot_nnue_error().decode())
            actual.append(score.value)
    finally:
        lib.ot_nnue_destroy(handle)
    expected = quantized_predict(weights, rows, turn) * 1000
    if len(actual) != len(expected) or np.max(np.abs(actual - expected)) > 0.501:
        raise ValueError('Exported network differs from C++ inference')
    report['cppParityPositions'] = len(actual)
    report['cppMaxRoundingError'] = float(np.max(np.abs(actual - expected)))
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('history', 'dataset')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=Path, default=Path('.build/training/games.jsonl'))
    parser.add_argument('--library', type=Path, default=Path('.build/native/libothello.dylib'))
    parser.add_argument('--cache', type=Path, default=Path('.build/training/positions.npz'))
    parser.add_argument('--output', type=Path, default=Path('.build/training/nnue-256.otnnue'))
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--holdout-cache', type=Path, help='Freeze validation/test positions from an existing cache')
    parser.add_argument('--all-training-games', action='store_true')
    parser.add_argument('--train-positions', type=int, default=0, help='Override training reservoir cap')
    parser.add_argument('--parity-positions', type=int, default=64)
    parser.add_argument('--hidden', type=int, choices=(32, 64, 128, 256), default=256)
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--max-games', type=int, default=6000)
    parser.add_argument('--max-positions', type=int, default=60000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    for value in (args.max_games, args.max_positions, args.epochs, args.batch_size, args.threads):
        if value < 1:
            parser.error('Counts must be positive')
    if args.train_positions < 0 or args.parity_positions < 1:
        parser.error('Invalid training/parity size')
    if args.holdout_cache and args.holdout_cache.resolve() == args.cache.resolve():
        parser.error('Frozen holdout and output cache must differ')
    train(args)
