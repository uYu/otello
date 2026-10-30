#!/usr/bin/env python3
"""Also preserve/evaluate the fully traversed epoch, beside validation selection."""
import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mixed', ROOT/'scripts/train-mixed-5m.py')
mixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mixed)


def evaluate(args):
    torch.set_num_threads(2)
    parent_path = ROOT/'models/nnue256-outcome-v1.9.otnnue'
    parent = parent_path.read_bytes()
    checkpoint = torch.load(args.run/'latest.pt', weights_only=True)
    assert checkpoint['completedShards'] == 64
    frozen = dict(np.load(ROOT/'.build/training/egaroucid-v0002/full-5m-v1/evaluation/frozen.npz'))
    net = mixed.MixedNet(parent)
    net.load_state_dict(checkpoint['model'])
    path = args.run/'nnue256-mixed-5m-full-epoch.otnnue'
    assert not path.exists()
    binary = mixed.export(net, parent)
    path.write_bytes(binary)
    torch.save(checkpoint['model'], path.with_suffix('.pt'))
    qnet = mixed.MixedNet(binary).double()
    report = dict(version='v1.11-mixed-5m-full-epoch-candidate', status='candidate', defaultPromoted=False,
                  selection='final weights after every eligible new and original training row; supplementary pre-specified comparison, no test tuning',
                  sha256=mixed.sha(path), parentSha256=mixed.sha(parent_path),
                  epoch=checkpoint['epoch'], newPositions=checkpoint['newPositions'], oldPositions=checkpoint['oldPositions'],
                  trainingReportSha256=mixed.sha(args.run/'report.json'), evaluationScriptSha256=mixed.sha(__file__), metrics={})
    for split in ('validation', 'test'):
        fp = mixed.predict(net, frozen, split)
        qp = mixed.predict(qnet, frozen, split)
        native = mixed.native_predictions(path, frozen, split, args.library)
        base = mixed.native_predictions(parent_path, frozen, split, args.library)
        verr = float(abs(qp[0]-native[0]).max())
        perr = float(abs(qp[1]-native[1]).max())
        assert verr <= .000501 and perr < 1e-10
        report['metrics'][split] = dict(float=mixed.metrics(*fp, frozen, split),
                                             candidateNative=mixed.metrics(*native, frozen, split),
                                             baselineNative=mixed.metrics(*base, frozen, split),
                                             cppParityPositions=len(qp[0]), cppMaxValueError=verr, cppMaxPolicyError=perr)
        if split == 'test':
            np.savez_compressed(args.run/'full-epoch-test-predictions.npz',
                                board=frozen['test_board'], turn=frozen['test_turn'], target=np.sign(frozen['test_target']),
                                move=frozen['test_move'], candidate_value=native[0], candidate_policy=native[1],
                                baseline_value=base[0], baseline_policy=base[1])
            with (args.run/'full-epoch-test-predictions.csv').open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['index', 'turn', 'outcome', 'recordedMove', 'baselineValue', 'candidateValue', 'baselineTopMove', 'candidateTopMove'])
                for i in range(len(native[0])):
                    writer.writerow([i, int(frozen['test_turn'][i]), int(np.sign(frozen['test_target'][i])), int(frozen['test_move'][i]),
                                     base[0][i], native[0][i], int(base[1][i].argmax()), int(native[1][i].argmax())])
    report['predictionsSha256'] = mixed.sha(args.run/'full-epoch-test-predictions.npz')
    mixed.write_json(args.run/'full-epoch-report.json', report)
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'.build/training/mixed-5m-v1')
    parser.add_argument('--library', type=Path, default=ROOT/'.build/native-engineering/libothello.dylib')
    evaluate(parser.parse_args())
