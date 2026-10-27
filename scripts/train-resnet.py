#!/usr/bin/env python3
"""Train a 32-channel residual value network on the frozen v1 position cache."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import time
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class ConvBn(nn.Sequential):
    def __init__(self, inputs, outputs, kernel=3):
        super().__init__(nn.Conv2d(inputs, outputs, kernel, padding=kernel // 2, bias=False),
                         nn.BatchNorm2d(outputs))


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = ConvBn(32, 32)
        self.second = ConvBn(32, 32)

    def forward(self, x):
        return torch.relu(x + self.second(torch.relu(self.first(x))))


class ResValueNet(nn.Module):
    def __init__(self, blocks=2):
        super().__init__()
        if blocks not in (2, 4):
            raise ValueError('Supported residual block counts: 2, 4')
        self.blocks = blocks
        self.stem = ConvBn(2, 32)
        self.tower = nn.Sequential(*(ResidualBlock() for _ in range(blocks)))
        self.value = ConvBn(32, 1, 1)
        self.fc = nn.Linear(64, 64)
        self.head = nn.Linear(64, 1)

    def forward(self, board, turn):
        relative = board * turn[:, None]
        x = torch.stack((relative == 1, relative == -1), dim=1).float().reshape(-1, 2, 8, 8)
        x = self.tower(torch.relu(self.stem(x)))
        x = torch.relu(self.value(x))
        x = torch.relu(self.fc(x.flatten(1)))
        return torch.tanh(self.head(x).squeeze(1))


def export(net, path):
    net.eval()
    layers = [net.stem]
    for block in net.tower:
        layers.extend((block.first, block.second))
    layers.append(net.value)
    arrays = []
    for layer in layers:
        fused = torch.nn.utils.fusion.fuse_conv_bn_eval(layer[0], layer[1])
        arrays.extend((fused.weight.detach().cpu().numpy().astype('<f4'),
                       fused.bias.detach().cpu().numpy().astype('<f4')))
    arrays.extend(p.detach().cpu().numpy().astype('<f4') for layer in (net.fc, net.head) for p in layer.parameters())
    assert sum(a.size for a in arrays) == 4866 + net.blocks * 18496
    if not all(np.isfinite(a).all() and np.max(np.abs(a)) <= 1000 for a in arrays):
        raise ValueError('Residual weights outside supported finite range')
    binary = struct.pack('<8sIIIII', b'OTRES1\0\0', 8, 2, 32, net.blocks, 64) + b''.join(a.tobytes() for a in arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(binary)
    return binary


def metric(prediction, truth):
    error = prediction - truth
    return dict(mse=float(np.mean(error ** 2)), maeDiscs=float(np.mean(np.abs(error)) * 64))


def predict(net, board, turn, batch_size=1024):
    predictions = []
    net.eval()
    device = next(net.parameters()).device
    with torch.no_grad():
        for offset in range(0, len(board), batch_size):
            predictions.append(net(torch.from_numpy(board[offset:offset + batch_size]).to(device),
                                   torch.from_numpy(turn[offset:offset + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(predictions)


def parity(library, model, board, turn, expected):
    lib = ctypes.CDLL(str(library.resolve()))
    lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
    lib.ot_nnue_load.restype = ctypes.c_void_p
    lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]
    lib.ot_nnue_evaluate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_int)]
    lib.ot_nnue_error.restype = ctypes.c_char_p
    handle = lib.ot_nnue_load(str(model.resolve()).encode())
    if not handle:
        raise ValueError(lib.ot_nnue_error().decode())
    actual = []
    try:
        for cells, actor in zip(board, turn):
            output = ctypes.c_int()
            if lib.ot_nnue_evaluate(handle, (ctypes.c_int * 64)(*map(int, cells)), int(actor), ctypes.byref(output)) != 1:
                raise ValueError(lib.ot_nnue_error().decode())
            actual.append(output.value)
    finally:
        lib.ot_nnue_destroy(handle)
    error = float(np.max(np.abs(np.array(actual) - expected * 1000)))
    if error > 0.51 or np.max(np.abs(np.array(actual) - np.rint(expected * 1000))) > 1:
        raise ValueError(f'Residual export differs from C++ inference: {error}')
    return error


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    metadata = json.loads(args.cache.with_suffix('.json').read_text())
    original = json.loads((ROOT / 'models/nnue256-v1.json').read_text())
    digest = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if digest != metadata['cacheSha256'] or digest != original['dataset']['cacheSha256']:
        raise ValueError('Residual experiment requires the exact frozen v1 cache')
    if args.seed != metadata['seed']:
        raise ValueError('Experiment seed must match the frozen training protocol')
    data = dict(np.load(args.cache))
    device = torch.device('mps' if args.device == 'auto' and torch.backends.mps.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    net = ResValueNet(args.blocks).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=0.001)
    boards, turns, truth = [torch.from_numpy(data['train_' + key]) for key in ('board', 'turn', 'target')]
    grid = np.arange(64).reshape(8, 8)
    permutations = torch.from_numpy(np.stack([np.rot90(np.fliplr(grid) if mirror else grid, rotation).reshape(64)
                                             for mirror in (False, True) for rotation in range(4)]))
    best, best_state, history = float('inf'), None, []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        net.train()
        for indices in torch.randperm(len(boards)).split(args.batch_size):
            batch = boards[indices]
            selection = permutations[torch.randint(8, (len(indices),))]
            batch = torch.gather(batch, 1, selection)
            prediction = net(batch.to(device), turns[indices].to(device))
            loss = torch.mean((prediction - truth[indices].to(device)) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        prediction = predict(net, data['validation_board'], data['validation_turn'])
        scores = metric(prediction, data['validation_target'])
        history.append(dict(epoch=epoch + 1, seconds=time.perf_counter() - started, **scores))
        if scores['mse'] < best:
            best = scores['mse']
            best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
        print(json.dumps(history[-1]), flush=True)
    net.load_state_dict(best_state)
    net.eval()
    binary = export(net, args.output)
    torch.save(net.state_dict(), args.output.with_suffix('.pt'))
    report = dict(version='v1.5-resnet-experiment', architecture=f'2x8x8 own/opponent -> Conv3x3 32/BN/ReLU -> {args.blocks} residual blocks (Conv3x3 32/BN/ReLU/Conv3x3 32/BN/add/ReLU) -> Conv1x1 1/BN/ReLU -> Flatten64 -> Linear64/ReLU -> Linear1/tanh',
                  modelFormat=4, formatMagic='OTRES1', precision='FP32; batch normalization folded into convolutions for inference',
                  channels=32, residualBlocks=args.blocks, headWidth=64, parameters=sum(p.numel() for p in net.parameters()),
                  inferenceParameters=(len(binary)-28)//4,
                  bytes=len(binary), sha256=hashlib.sha256(binary).hexdigest(), seed=args.seed,
                  bestEpoch=min(history, key=lambda row: row['mse'])['epoch'], epochs=args.epochs,
                  batchSize=args.batch_size, learningRate=args.learning_rate, weightDecay=0.001, threads=args.threads,
                  torchVersion=torch.__version__, numpyVersion=np.__version__, device=str(device), dataset=metadata,
                  trainingScriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  seconds=time.perf_counter() - started, history=history, metrics={})
    for split in ('validation', 'test'):
        prediction = predict(net, data[split + '_board'], data[split + '_turn'])
        report['metrics'][split] = dict(float=metric(prediction, data[split + '_target']),
                                        zeroBaseline=metric(np.zeros_like(prediction), data[split + '_target']))
        if split == 'test':
            count = min(args.parity_positions, len(prediction))
            report['cppMaxRoundingError'] = parity(args.library, args.output, data['test_board'][:count],
                                                  data['test_turn'][:count], prediction[:count])
            report['cppParityPositions'] = count
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('history', 'dataset')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / '.build/training/v1/positions.npz')
    parser.add_argument('--output', type=Path, default=ROOT / '.build/training/v1.5/resnet32x2-v1.5.otres')
    parser.add_argument('--library', type=Path, default=ROOT / ('.build/native-v1.5/libothello.' + ('dylib' if sys.platform == 'darwin' else 'so')))
    parser.add_argument('--blocks', type=int, choices=(2, 4), default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--parity-positions', type=int, default=1024)
    options = parser.parse_args()
    if min(options.epochs, options.batch_size, options.threads, options.parity_positions) < 1 or options.learning_rate <= 0:
        parser.error('Training counts and learning rate must be positive')
    train(options)
