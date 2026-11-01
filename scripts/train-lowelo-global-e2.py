#!/usr/bin/env python3
"""Train the complete v2 architecture from scratch on the fixed 10k low-Elo cache."""
import hashlib
import json
import struct
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pattern_global_policy import GlobalPolicyNet, permutations
from pattern_model import encode, legal_mask, parity
from pattern_structure import export

ROOT = Path(__file__).resolve().parents[1]
NAME = 'lowelo-global-e2-lr0003-v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    started = time.perf_counter()
    model = ROOT / f'models/{NAME}.otpat'
    report = ROOT / f'models/{NAME}-training.json'
    checkpoint = model.with_suffix('.pt')
    assert not any(p.exists() for p in [model, report, checkpoint])
    work = ROOT / f'.build/training/{NAME}'
    work.mkdir(parents=True, exist_ok=True)
    cache = ROOT / '.build/training/beginner-low-elo1300-v1/positions.npz'
    data_report = ROOT / 'models/beginner-low-elo1300-v1-data.json'
    metadata = json.loads(data_report.read_text())
    assert sha(cache) == metadata['cacheSha256']
    data = dict(np.load(cache))
    assert len(data['train_board']) == 10000
    preserved = {p: sha(ROOT / p) for p in ['models/default-model.json', 'models/v2-default.json',
        'models/pattern32-global-policy-v2.otpat', 'models/beginner-default.json',
        'models/second-level-selection-v2.json', 'src/game/wasm/othello.mjs']}
    torch.set_num_threads(2)
    torch.manual_seed(20261101)
    torch.use_deterministic_algorithms(True)
    net = GlobalPolicyNet(1, freeze_baseline=False)
    assert all(p.requires_grad for p in net.parameters())
    assert sum(p.numel() for p in net.parameters()) == 916987
    initial = {k: p.detach().clone() for k, p in net.named_parameters()}
    fingerprint = hashlib.sha256(b''.join(p.numpy().tobytes() for p in initial.values())).hexdigest()
    rng = np.random.default_rng(20261101)
    perms = permutations(8)
    inverses = np.argsort(perms, axis=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=.0003)

    def predict(board):
        values, outcomes, policies = [], [], []
        net.eval()
        with torch.no_grad():
            for offset in range(0, len(board), 250):
                b = board[offset:offset + 250]
                phase = (64 - np.count_nonzero(b, axis=1)).astype(np.float32) / 60
                v, w, p = net(torch.from_numpy(encode(b, np.ones(len(b), np.int8)).astype(np.int64)), torch.from_numpy(phase))
                values.append(v.numpy()); outcomes.append(w.numpy()); policies.append(p.numpy())
        return tuple(np.concatenate(x) for x in [values, outcomes, policies])

    def metrics(split):
        v, w, p = predict(data[split + '_board'])
        mask, moves = data[split + '_legal'], data[split + '_move']
        target = np.sign(data[split + '_target'])
        p = torch.from_numpy(p).masked_fill(~torch.from_numpy(mask), -1e9)
        return dict(positions=len(moves), policyCrossEntropy=float(F.cross_entropy(p, torch.from_numpy(moves))),
            policyTop1=float((p.argmax(1).numpy() == moves).mean()), outcomeMse=float(np.mean((v - target)**2)),
            wdlAccuracy=float((w.argmax(1) == np.where(target > 0, 0, np.where(target < 0, 1, 2))).mean()),
            uniformPolicyCrossEntropy=float(np.log(mask.sum(1)).mean()))

    for split in ['train', 'validation', 'test']:
        assert np.array_equal(legal_mask(data[split+'_board'], np.ones(len(data[split+'_board']), np.int8)), data[split+'_legal'])
    history = [dict(epoch=0, validation=metrics('validation'))]
    for epoch in range(1, 3):
        net.train()
        losses = []
        for rows in rng.permutation(10000).reshape(-1, 250):
            sym = rng.integers(8)
            perm, inverse = perms[sym], inverses[sym]
            b = data['train_board'][rows][:, perm]
            codes = torch.from_numpy(encode(b, np.ones(len(b), np.int8)).astype(np.int64))
            phase = torch.from_numpy((64 - np.count_nonzero(b, axis=1)).astype(np.float32) / 60)
            value, wdl, policy = net(codes, phase)
            target = torch.from_numpy(np.sign(data['train_target'][rows]))
            outcome = torch.where(target > 0, 0, torch.where(target < 0, 1, 2)).long()
            mask = torch.from_numpy(data['train_legal'][rows][:, perm])
            move = torch.from_numpy(inverse[data['train_move'][rows]].astype(np.int64))
            components = [F.mse_loss(value, target), F.cross_entropy(wdl, outcome),
                          F.cross_entropy(policy.masked_fill(~mask, -1e9), move)]
            loss = components[0] + .1 * components[1] + .1 * components[2]
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5); optimizer.step()
            losses.append([float(x.detach()) for x in components])
        row = dict(epoch=epoch, trainLossComponents=np.mean(losses, axis=0).tolist(), validation=metrics('validation'))
        history.append(row)
        print(json.dumps(row), flush=True)
    net.eval()
    torch.save(net.state_dict(), checkpoint)
    base = export(net, work / 'base.otpat')
    assert base[:8] == b'OTPAT4\0\0'
    binary = b'OTPAT9\0\0' + base[8:36] + struct.pack('<3I', 38, 16, 64) + base[36:]
    for layer in [net.global_down, net.global_up]:
        for tensor, scale, dtype in [(layer.weight, 256, '<i2'), (layer.bias, 65536, '<i4')]:
            x = tensor.detach().numpy().astype(np.float64) * scale
            x = np.sign(x) * np.floor(np.abs(x) + .5)
            limits = np.iinfo(dtype)
            assert np.isfinite(x).all() and x.min() >= limits.min and x.max() <= limits.max
            binary += x.astype(dtype).tobytes()
    model.write_bytes(binary)
    checks = {}
    for split in ['validation', 'test']:
        boards = data[split + '_board']
        checks[split] = parity(ROOT / '.build/native/libothello.dylib', model, boards,
                              np.ones(len(boards), np.int8), predict(boards))
    changed = {k: not torch.equal(v, dict(net.named_parameters())[k]) for k, v in initial.items()}
    assert all(changed.values())
    for path, digest in preserved.items():
        assert sha(ROOT / path) == digest
    boards = data['test_board'][:12]
    v, w, p = predict(boards)
    result = dict(version=NAME, status='candidate-not-promoted', complete=True,
        architecture='Exact v2 OTPAT9: 38 D4-tied patterns, 32 channels, three mean pools, 97->32 trunk, value/WDL/conditional policy plus D4-tied global 38->16->64 correction',
        parameters=916987, allParametersTrainable=True, allParametersTrainedFromScratch=True,
        pretrainedWeightsLoaded=False, initialization='Original v2 random initialization scheme, including zero global output correction; no checkpoint loaded',
        initialWeightsSha256=fingerprint, allParameterTensorsChanged=changed,
        epochs=2, optimizerUpdates=80, batchSize=250, learningRate=.0003, optimizer='Adam', gradientClip=5,
        seed=20261101, trainPositions=10000, dataReport=str(data_report.relative_to(ROOT)),
        dataReportSha256=sha(data_report), cacheSha256=sha(cache),
        loss='Outcome MSE +0.1 WDL CE +0.1 legal recorded low-Elo move CE (original v2 coefficients)',
        target='Recorded actor move and recorded final WDL; no synthetic errors or stronger-engine relabeling',
        augmentation='One uniformly chosen D4 view per shuffled 250-row batch',
        selection='Fixed final epoch2; no validation/test checkpoint selection',
        inference='One native quantized v2 network policy map per position, legal mask, model softmax sampling T=1; no book/MCTS/heuristic/uniform mixture; value/WDL heads trained but unused for selection',
        model=str(model.relative_to(ROOT)), modelSha256=sha(model), checkpoint=str(checkpoint.relative_to(ROOT)), checkpointSha256=sha(checkpoint),
        history=history, test=metrics('test'), nativeParity=checks,
        fixtures=[dict(board=b.astype(int).tolist(), turn=1, value=float(value), logits=policy.tolist()) for b, value, policy in zip(boards, v, p)],
        sourceSha256={s: sha(ROOT / s) for s in ['scripts/train-lowelo-global-e2.py', 'scripts/pattern_global_policy.py', 'scripts/pattern_model.py', 'scripts/pattern_structure.py']},
        preservedSha256=preserved, dataNotices='data/README.md', seconds=time.perf_counter()-started,
        environment=dict(torch=torch.__version__, numpy=np.__version__, threads=2, device='cpu'),
        limits='Architecture and loss differ from the small policy-only model. Match results are required; no human strength inference from loss.')
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['model', 'parameters', 'epochs', 'test', 'nativeParity', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
