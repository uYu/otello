#!/usr/bin/env python3
"""Train an independent Pattern outcome candidate against the approved v1.9 baseline."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from pattern_model import PatternNet, encode, legal_mask, export, parity, DIM, ROWS

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_spec = importlib.util.spec_from_file_location('pattern_legacy_training', ROOT/'scripts/train-pattern.py')
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
prepare, predict = legacy.prepare, legacy.predict


def metric(prediction, target, policy_target, mask):
    value, wdl, logits = prediction
    target = np.sign(target)
    masked = np.where(mask, logits, -1e9)
    truth = np.where(target > 0, 0, np.where(target < 0, 1, 2))
    probs = torch.softmax(torch.from_numpy(wdl), dim=1).numpy()
    logs = torch.log_softmax(torch.from_numpy(masked), dim=1).numpy()
    non_draw = target != 0
    return dict(mse=float(np.mean((value-target)**2)), outcomeMae=float(np.mean(np.abs(value-target))),
        nonDrawSignAccuracy=float(np.mean(np.sign(value[non_draw]) == target[non_draw])),
        wdlAccuracy=float(np.mean(wdl.argmax(1) == truth)),
        wdlCrossEntropy=float(-np.log(np.maximum(1e-12, probs[np.arange(len(target)), truth])).mean()),
        policyTop1=float(np.mean(masked.argmax(1) == policy_target)),
        policyCrossEntropy=float(-logs[np.arange(len(target)), policy_target].mean()),
        uniformPolicyCrossEntropy=float(np.log(mask.sum(1)).mean()))


def train(args):
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    metadata = json.loads(args.cache.with_suffix('.json').read_text())
    expected = json.loads((ROOT/'models/nnue256-v1.json').read_text())['dataset']['cacheSha256']
    if sha(args.cache) != expected or expected != metadata['cacheSha256']:
        raise ValueError('Exact frozen v1 cache required')
    data = dict(np.load(args.cache))
    if (args.work/'policy.npz').exists():
        provenance = json.loads((args.work/'policy.json').read_text())
        if provenance['cacheSha256'] != expected or provenance['gamesSha256'] != sha(args.games) or provenance['policySha256'] != sha(args.work/'policy.npz'):
            raise ValueError('Policy cache provenance mismatch')
        labels = dict(np.load(args.work/'policy.npz'))
    else:
        labels, provenance = prepare(args, data, metadata)
    masks = {}
    for split in labels:
        masks[split] = legal_mask(data[split+'_board'], data[split+'_turn'])
        if not masks[split][np.arange(len(labels[split])), labels[split]].all():
            raise ValueError('Illegal policy target')
    active = json.loads((ROOT/'models/default-model.json').read_text())
    baseline_path = ROOT/active['baselineManifest']
    baseline = json.loads(baseline_path.read_text())
    assert baseline['version'] == 'v1.9' and baseline['userApproved'] and baseline['status'] == 'frozen'
    assert sha(ROOT/baseline['defaultModel']) == baseline['modelSha256']
    net = PatternNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    board, turn, target = [data['train_'+k] for k in ('board', 'turn', 'target')]
    if args.train_positions:
        indices = np.random.default_rng(args.seed).choice(len(board), min(args.train_positions, len(board)), replace=False)
        board, turn, target = board[indices], turn[indices], target[indices]
        policy_labels, train_masks = labels['train'][indices], masks['train'][indices]
    else:
        policy_labels, train_masks = labels['train'], masks['train']
    grid = np.arange(64).reshape(8, 8)
    permutations = np.stack([np.rot90(np.fliplr(grid) if mirror else grid, rotation).reshape(64)
                             for mirror in (False, True) for rotation in range(4)])
    inverse = np.argsort(permutations, axis=1)
    phase = (64-np.count_nonzero(board, axis=1)).astype(np.float32)/60
    best, best_state, history = float('inf'), None, []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        net.train(); losses = []
        for tensor_indices in torch.randperm(len(board)).split(args.batch_size):
            indices = tensor_indices.numpy()
            sym = np.random.randint(8, size=len(indices)); perm = permutations[sym]
            batch = np.take_along_axis(board[indices], perm, axis=1)
            code = torch.from_numpy(encode(batch, turn[indices]).astype(np.int64))
            disc, wdl, policy = net(code, torch.from_numpy(phase[indices]))
            truth = torch.from_numpy(np.sign(target[indices]))
            next_move = torch.from_numpy(inverse[sym, policy_labels[indices]])
            mask = torch.from_numpy(np.take_along_axis(train_masks[indices], perm, axis=1))
            outcome = torch.where(truth > 0, 0, torch.where(truth < 0, 1, 2)).long()
            loss = F.mse_loss(disc, truth)+0.1*F.cross_entropy(wdl, outcome)+0.1*F.cross_entropy(policy.masked_fill(~mask, -1e9), next_move)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5); optimizer.step()
            losses.append(float(loss.detach()))
        validation = predict(net, data['validation_board'], data['validation_turn'])
        scores = metric(validation, data['validation_target'], labels['validation'], masks['validation'])
        # Frozen value validation selects the checkpoint, without looking at test or matches.
        if scores['mse'] < best:
            best = scores['mse']; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        history.append(dict(epoch=epoch+1, seconds=time.perf_counter()-started, loss=float(np.mean(losses)), **scores))
        print(json.dumps(history[-1]), flush=True)
        args.work.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, args.work/'best.pt')
        (args.work/'progress.json').write_text(json.dumps(dict(experiment='pattern-outcome-training', complete=False, epoch=epoch+1, epochs=args.epochs, seconds=time.perf_counter()-started, validation=scores), indent=2)+'\n')
    net.load_state_dict(best_state); net.eval()
    binary = export(net, args.output, outcome=True)
    torch.save(net.state_dict(), args.output.with_suffix('.pt'))
    report = dict(version='v1.25-pattern-outcome-candidate', status='candidate', defaultPromoted=False,
                  baselineVersion=baseline['version'], baselineManifest=str(baseline_path.relative_to(ROOT)),
                  baselineManifestSha256=sha(baseline_path), baselineModelSha256=baseline['modelSha256'],
                  initialization='random; same architecture as OTPAT1, no inherited NNUE weights',
                  valueTarget='side-to-move win/draw/loss +1/0/-1; sign of frozen recorded final disc target',
                  terminalBackup='outcome +1/0/-1; proof stopping inherited from v1.9', architecture='38 ternary line/corner lookups -> 32D embeddings; edge/interior/corner mean pools + empty/60 -> 97->32 clipped ReLU -> disc/WDL/global-conditioned per-square policy',
                  formatMagic='OTPAT2', features=38, embeddingRows=ROWS, embeddingDimension=DIM,
                  parameters=sum(p.numel() for p in net.parameters()), bytes=len(binary), sha256=sha(args.output),
                  precision='int16 weights and embeddings, int32 biases/accumulators, scale 256, straight-through quantization training',
                  seed=args.seed, epochs=args.epochs, trainPositions=len(board), batchSize=args.batch_size, threads=args.threads,
                  learningRate=args.learning_rate, weightDecay=0.0001, loss='outcome MSE + 0.1 WDL CE + 0.1 legal-masked policy CE',
                  checkpointSelection='minimum frozen validation outcome MSE', bestEpoch=min(history, key=lambda r: r['mse'])['epoch'],
                  torchVersion=torch.__version__, numpyVersion=np.__version__, device='cpu',
                  seconds=time.perf_counter()-started, dataset=metadata, policyDataset=provenance, history=history, metrics={},
                  scriptSha256={p.name: sha(p) for p in [Path(__file__), ROOT/'scripts/pattern_model.py', ROOT/'scripts/train-pattern.py']},
                  sourceAttribution='WTHOR/Logistello; original attribution and usage constraints remain in data/README.md and manifests. Independently implemented Othello architecture inspired by Rapfi.')
    for split in ('validation', 'test'):
        pred = predict(net, data[split+'_board'], data[split+'_turn'])
        report['metrics'][split] = metric(pred, data[split+'_target'], labels[split], masks[split])
        if split == 'test':
            n = min(args.parity_positions, len(pred[0]))
            report['cppParity'] = parity(args.library, args.output, data['test_board'][:n], data['test_turn'][:n], tuple(p[:n] for p in pred))
    report['cppParity']['maxOutcomeError'] = report['cppParity'].pop('maxDiscError')
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.work/'progress.json').write_text(json.dumps(dict(experiment='pattern-outcome-training', complete=True, epoch=args.epochs, epochs=args.epochs, seconds=report['seconds'], validation=report['metrics']['validation']), indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('history', 'dataset', 'policyDataset')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    suffix = 'dylib' if sys.platform == 'darwin' else 'so'
    p.add_argument('--cache', type=Path, default=ROOT/'.build/training/v1/positions.npz')
    p.add_argument('--games', type=Path, default=ROOT/'.build/training/games.jsonl')
    p.add_argument('--work', type=Path, default=ROOT/'.build/training/pattern-outcome-v1.25')
    p.add_argument('--library', type=Path, default=ROOT/f'.build/native/libothello.{suffix}')
    p.add_argument('--output', type=Path, default=ROOT/'models/pattern32-outcome-v1.25.otpat')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--train-positions', type=int, default=0)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=20261029)
    p.add_argument('--parity-positions', type=int, default=6000)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.threads, args.parity_positions) < 1 or args.train_positions < 0 or args.learning_rate <= 0:
        p.error('Invalid training budget')
    if args.output.exists():
        p.error('Refusing to overwrite an existing model; choose a new --output')
    train(args)
