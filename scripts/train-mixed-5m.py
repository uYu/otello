#!/usr/bin/env python3
"""Stream all eligible 5m-game positions plus the existing training cache.

Keep v1.9's transform/format; fit both heads with soft empirical targets.
Only frozen validation chooses checkpoints; test is evaluated after training.
"""
import argparse
import csv
import ctypes
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('deep', ROOT/'scripts/train-deep.py')
deep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deep)
outcome, policy = deep.outcome, deep.policy


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


class MixedNet(nn.Module):
    def __init__(self, binary):
        super().__init__()
        base = deep.DeepNet(binary)
        self.transform, self.head = base.transform, base.head
        self.hidden, self.policy = base.policy_hidden, base.policy_out
        self.transform.requires_grad_(False)
        self.hidden.requires_grad_(True)
        self.policy.requires_grad_(True)

    def forward(self, board, turn):
        with torch.no_grad():
            x = policy.value.features(board).to(self.transform.weight.dtype)
            a = self.transform(x).clamp(0, 1)
            b = self.transform(torch.cat((x[:, 64:], x[:, :64]), 1)).clamp(0, 1)
            actor = (turn > 0)[:, None]
            joined = torch.cat((torch.where(actor, a, b), torch.where(actor, b, a)), 1)
        return torch.tanh(self.head(joined).squeeze(1)), self.policy(self.hidden(joined).clamp(0, 1))


def export(net, parent):
    value = deep.packed(net.head.weight, 256, '<i2')+deep.packed(net.head.bias, 65536, '<i4')
    return (parent[:outcome.VALUE_OFFSET]+value+parent[outcome.VALUE_END:67620]
            + b''.join(w.tobytes() for w in policy.quantize(net)))


def unpack_bits(bits):
    return ((bits[:, None] >> np.arange(64, dtype=np.uint64)) & 1)


def shard_batch(data, indices):
    """CSR counts become distributions only for the current minibatch."""
    board = unpack_bits(data['own'][indices]).astype(np.int8)-unpack_bits(data['opponent'][indices]).astype(np.int8)
    count = data['count'][indices].astype(np.float32)
    counts = data['outcome_counts'][indices].astype(np.float32)
    target = (counts[:, 2]-counts[:, 0])/count
    legal = np.zeros((len(indices), 65), dtype=bool)
    legal[:, :64] = unpack_bits(data['legal'][indices])
    start, end = data['policy_offsets'][indices], data['policy_offsets'][indices+1]
    sizes = (end-start).astype(np.int64)
    owners = np.repeat(np.arange(len(indices)), sizes)
    offsets = np.repeat(start, sizes).astype(np.int64)
    offsets += np.arange(len(owners))-np.repeat(np.cumsum(sizes)-sizes, sizes)
    soft = np.zeros((len(indices), 65), dtype=np.float32)
    soft[owners, data['policy_move'][offsets]] = data['policy_count'][offsets]/count[owners]
    return board, np.ones(len(indices), np.float32), target, legal, soft


def augment(board, legal, soft, permutations, rng):
    p = permutations[rng.integers(8, size=len(board))]
    rows = np.arange(len(board))[:, None]
    board = board[rows, p]
    legal = np.concatenate((legal[:, :64][rows, p], legal[:, 64:]), 1)
    soft = np.concatenate((soft[:, :64][rows, p], soft[:, 64:]), 1)
    return board, legal, soft


def predict(net, frozen, split):
    values, probs = [], []
    with torch.no_grad():
        for start in range(0, len(frozen[split+'_board']), 2048):
            part = slice(start, start+2048)
            v, p = net(torch.from_numpy(frozen[split+'_board'][part]), torch.from_numpy(frozen[split+'_turn'][part]))
            mask = torch.from_numpy(frozen[split+'_legal'][part])
            values.append(v.numpy())
            probs.append(p.masked_fill(~mask, -torch.inf).softmax(1).numpy())
    return np.concatenate(values), np.concatenate(probs)


def metrics(v, p, frozen, split):
    vm = outcome.metric(v, np.sign(frozen[split+'_target']))
    pm = policy.metrics(p, frozen[split+'_move'], frozen[split+'_legal'])
    return dict(value=vm, policy=pm, selectionLoss=vm['outcomeMse']+pm['crossEntropy'])


def native_predictions(path, frozen, split, library):
    lib = ctypes.CDLL(str(library.resolve()))
    ptr = ctypes.POINTER(ctypes.c_int)
    lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
    lib.ot_nnue_load.restype = ctypes.c_void_p
    lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]
    lib.ot_nnue_evaluate.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ptr]
    lib.ot_nnue_policy.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    handle = lib.ot_nnue_load(str(path.resolve()).encode())
    if not handle:
        raise ValueError('Native model load failed')
    values, probs = [], []
    try:
        for b, t in zip(frozen[split+'_board'], frozen[split+'_turn']):
            cells = (ctypes.c_int*64)(*map(int, b))
            score, p = ctypes.c_int(), (ctypes.c_double*65)()
            assert lib.ot_nnue_evaluate(handle, cells, int(t), ctypes.byref(score)) == 1
            assert lib.ot_nnue_policy(handle, cells, int(t), p) == 1
            values.append(score.value/1000)
            probs.append(np.array(p))
    finally:
        lib.ot_nnue_destroy(handle)
    return np.array(values), np.array(probs)


def train(args):
    args.output.mkdir(parents=True, exist_ok=True)
    final = args.output/'nnue256-mixed-5m.otnnue'
    if final.exists():
        raise ValueError('Use a fresh output directory')
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    parent_path = ROOT/'models/nnue256-outcome-v1.9.otnnue'
    parent = parent_path.read_bytes()
    baseline = json.loads((ROOT/'models/v1.9-default.json').read_text())
    assert sha(parent_path) == baseline['modelSha256']
    source = json.loads((args.input/'report.json').read_text())
    verified = json.loads((args.input/'verification.json').read_text())
    assert verified['verified'] and source['selection']['selectedGames'] == 5000000
    assert sha(args.input/'evaluation/frozen.npz') == source['frozenEvaluationSha256']
    frozen = dict(np.load(args.input/'evaluation/frozen.npz'))
    old_path, labels_path = ROOT/'.build/training/v1/positions.npz', ROOT/'.build/training/policy/labels.npz'
    old, labels = dict(np.load(old_path)), dict(np.load(labels_path))
    old_meta = json.loads(old_path.with_suffix('.json').read_text())
    labels_meta = json.loads(labels_path.with_suffix('.json').read_text())
    assert sha(old_path) == old_meta['cacheSha256'] == labels_meta['positionsSha256']
    assert sha(labels_path) == labels_meta['cacheSha256']
    for split in ('validation', 'test'):
        for k in ('board', 'turn', 'target'):
            assert np.array_equal(old[split+'_'+k], frozen[split+'_'+k])
        for k in ('move', 'legal'):
            assert np.array_equal(labels[split+'_'+k], frozen[split+'_'+k])
    net = MixedNet(parent)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=.001)
    permutations = policy.value.symmetries()
    history = []
    total_new = total_old = 0
    progress = dict(experiment='prediction-training', complete=False, epochs=args.epochs,
                    epoch=0, completedShards=0, totalShards=64, newPositions=0, oldPositions=0)

    def update(**kwargs):
        progress.update(kwargs, seconds=round(time.monotonic()-started, 2))
        write_json(args.output/'progress.json', progress)

    def validation(epoch, shards):
        score = metrics(*predict(net, frozen, 'validation'), frozen, 'validation')
        row = dict(epoch=epoch, completedShards=shards, **score)
        history.append(row)
        print(json.dumps(row), flush=True)
        return score['selectionLoss']

    update(stage='initial-validation')
    initial = validation(0, 0)
    # Select among trained checkpoints; retain initial baseline as a comparison.
    best, best_state, best_checkpoint = float('inf'), None, None

    def step(batch):
        board, turn, target, legal, soft = batch
        board, legal, soft = augment(board, legal, soft, permutations, rng)
        v, logits = net(torch.from_numpy(board), torch.from_numpy(turn))
        logp = logits.masked_fill(~torch.from_numpy(legal), -1e9).log_softmax(1)
        loss = ((v-torch.from_numpy(target))**2).mean()-(torch.from_numpy(soft)*logp).sum(1).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        opt.zero_grad()
        loss.backward()
        opt.step()

    for epoch in range(1, args.epochs+1):
        old_order = rng.permutation(len(old['train_board']))
        old_chunks = np.array_split(old_order, 64)
        for number, bucket in enumerate(rng.permutation(64), 1):
            path = args.input/f'positions/{bucket:03d}.npz'
            shard_meta = json.loads(path.with_suffix('.json').read_text())
            assert sha(path) == shard_meta['sha256']
            with np.load(path) as archive:
                data = {k: archive[k] for k in archive.files}
            eligible = np.flatnonzero((data['split'] == 0) & ~data['frozen'])
            assert len(eligible) == shard_meta['trainPositions']
            rng.shuffle(eligible)
            for offset in range(0, len(eligible), args.batch_size):
                indices = eligible[offset:offset+args.batch_size]
                step(shard_batch(data, indices))
                total_new += len(indices)
                if offset % (args.batch_size*16) == 0:
                    update(stage='training', epoch=epoch, completedShards=number-1,
                           newPositions=total_new, oldPositions=total_old, currentBucket=int(bucket))
            for offset in range(0, len(old_chunks[number-1]), args.batch_size):
                ids = old_chunks[number-1][offset:offset+args.batch_size]
                soft = np.eye(65, dtype=np.float32)[labels['train_move'][ids]]
                step((old['train_board'][ids], old['train_turn'][ids], np.sign(old['train_target'][ids]), labels['train_legal'][ids], soft))
                total_old += len(ids)
            update(completedShards=number, newPositions=total_new, oldPositions=total_old)
            if number % 8 == 0:
                score = validation(epoch, number)
                if score < best:
                    best = score
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    best_checkpoint = dict(epoch=epoch, completedShards=number, newPositions=total_new, oldPositions=total_old)
                    torch.save(best_state, args.output/'best.pt')
                update(validation=history[-1], bestCheckpoint=best_checkpoint)
            torch.save(dict(model=net.state_dict(), optimizer=opt.state_dict(), epoch=epoch,
                            completedShards=number, newPositions=total_new, oldPositions=total_old), args.output/'latest.pt')
    assert total_new == args.epochs*source['totals']['trainPositions']
    assert total_old == args.epochs*len(old['train_board'])
    net.load_state_dict(best_state)
    binary = export(net, parent)
    assert len(binary) == len(parent) and binary[:outcome.VALUE_OFFSET] == parent[:outcome.VALUE_OFFSET]
    final.write_bytes(binary)
    torch.save(best_state, final.with_suffix('.pt'))
    update(stage='frozen-evaluation')
    report = dict(version='v1.11-mixed-5m-candidate', status='candidate', defaultPromoted=False,
                  architecture='v1.9 frozen 128->256 dual-view transform; train value 512->1 tanh and policy 512->32 clipped ReLU->65',
                  modelFormat='OTNNUE4', parent=str(parent_path.relative_to(ROOT)), parentSha256=sha(parent_path),
                  sha256=sha(final), featuresUnchanged=True, baselineVersion='v1.9',
                  epochs=args.epochs, batchSize=args.batch_size, learningRate=args.learning_rate, weightDecay=.001,
                  seed=args.seed, threads=args.threads, trainingScriptSha256=sha(__file__),
                  helperSha256={p: sha(ROOT/p) for p in ['scripts/train-deep.py', 'scripts/train-outcome.py', 'scripts/train-policy.py', 'scripts/train-nnue.py']},
                  torchVersion=torch.__version__, numpyVersion=np.__version__,
                  dataset=dict(selectedGames=5000000, newPositionsPerEpoch=source['totals']['trainPositions'],
                               oldPositionsPerEpoch=len(old['train_board']), processedNewPositions=total_new, processedOldPositions=total_old,
                               sourceReportSha256=sha(args.input/'report.json'), verificationSha256=sha(args.input/'verification.json'),
                               frozenEvaluationSha256=sha(args.input/'evaluation/frozen.npz'),
                               oldCacheSha256=sha(old_path), oldLabelsSha256=sha(labels_path),
                               filter='split == 0 and frozen == false; verified shard hashes; each eligible row once per epoch',
                               mixing='each shard followed by 1/64 of randomly permuted old training rows; equal weight per aggregate row; no occurrence weighting',
                               augmentation='random D4 per batch row, jointly transforming board/legal/soft policy',
                               targets='new value=(win-loss)/count, policy=move_count/count; old value=sign(recorded disc difference), policy=one-hot recorded move'),
                  selection='minimum frozen validation outcome MSE + recorded-move cross entropy, every 8 shards; only trained checkpoints eligible; test evaluated after selection',
                  bestCheckpoint=best_checkpoint, initialValidationLoss=initial, history=history, metrics={})
    qnet = MixedNet(binary).double()
    predictions = {}
    for split in ('validation', 'test'):
        fp = predict(net, frozen, split)
        qp = predict(qnet, frozen, split)
        native = native_predictions(final, frozen, split, args.library)
        base = native_predictions(parent_path, frozen, split, args.library)
        error = float(np.max(np.abs(native[0]-qp[0])))
        perr = float(np.max(np.abs(native[1]-qp[1])))
        assert error <= .000501 and perr < 1e-10, (error, perr)
        report['metrics'][split] = dict(float=metrics(*fp, frozen, split),
                                             candidateNative=metrics(*native, frozen, split),
                                             baselineNative=metrics(*base, frozen, split),
                                             cppParityPositions=len(qp[0]), cppMaxValueError=error, cppMaxPolicyError=perr,
                                             maxValueQuantizationError=float(np.max(np.abs(fp[0]-qp[0]))))
        if split == 'test':
            predictions.update(candidate_value=native[0], candidate_policy=native[1], baseline_value=base[0], baseline_policy=base[1],
                               board=frozen['test_board'], turn=frozen['test_turn'], target=np.sign(frozen['test_target']), move=frozen['test_move'])
    np.savez_compressed(args.output/'test-predictions.npz', **predictions)
    with (args.output/'test-predictions.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['index', 'turn', 'outcome', 'recordedMove', 'baselineValue', 'candidateValue', 'baselineTopMove', 'candidateTopMove'])
        for i in range(len(predictions['target'])):
            writer.writerow([i, int(predictions['turn'][i]), int(predictions['target'][i]), int(predictions['move'][i]),
                             predictions['baseline_value'][i], predictions['candidate_value'][i],
                             int(predictions['baseline_policy'][i].argmax()), int(predictions['candidate_policy'][i].argmax())])
    report.update(seconds=time.monotonic()-started, predictionsSha256=sha(args.output/'test-predictions.npz'))
    write_json(args.output/'report.json', report)
    update(stage='complete', complete=True, metrics=report['metrics'])
    print(json.dumps(dict(complete=True, metrics=report['metrics'], seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT/'.build/training/egaroucid-v0002/full-5m-v1')
    p.add_argument('--output', type=Path, default=ROOT/'.build/training/mixed-5m-v1')
    p.add_argument('--library', type=Path, default=ROOT/'.build/native-engineering/libothello.dylib')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=8192)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--learning-rate', type=float, default=.0003)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.threads, args.learning_rate) <= 0:
        p.error('Positive training parameters required')
    train(args)
