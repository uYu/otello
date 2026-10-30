#!/usr/bin/env python3
"""Export fixed two-convolution policy/outcome weights, never retrain."""
import argparse
import importlib.util
import struct
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('suite', ROOT/'scripts/train-cnn-policy-suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def export(net, path):
    c = net.conv1.out_channels
    if c not in (8, 16, 32) or net.fc.out_features not in (16, 32, 64):
        raise ValueError('Unsupported policy CNN dimensions')
    arrays = [p.detach().cpu().numpy().astype('<f4') for p in net.parameters()]
    if not all(np.isfinite(a).all() and abs(a).max() <= 1000 for a in arrays):
        raise ValueError('Invalid CNN weights')
    binary = struct.pack('<8sIIIII', b'OTCNN3\0\0', 8, 2, c, net.fc.out_features, 65)+b''.join(a.tobytes() for a in arrays)
    if path.exists():
        if path.read_bytes() != binary:
            raise ValueError('Refusing to replace a different exported model')
    else:
        path.write_bytes(binary)
    return binary


def main(args):
    torch.set_num_threads(2)
    bundle = torch.load(args.model, map_location='cpu', weights_only=True)
    net = suite.CnnPolicyValue(bundle['channels'], bundle['headWidth']).eval()
    net.load_state_dict(bundle['state_dict'])
    export(net, args.output)
    frozen = dict(np.load(ROOT/'.build/training/egaroucid-v0002/full-5m-v1/evaluation/frozen.npz'))
    report = dict(model=str(args.output.resolve().relative_to(ROOT)), sha256=suite.helpers.sha(args.output),
                  checkpoint=str(args.model.resolve().relative_to(ROOT)), checkpointSha256=suite.helpers.sha(args.model),
                  exportScriptSha256=suite.helpers.sha(__file__), librarySha256=suite.helpers.sha(args.library),
                  format='OTCNN3', valueTarget='side-to-move outcome +1/0/-1', channels=bundle['channels'], metrics={})
    for split in ('validation', 'test'):
        fp = suite.predict(net, frozen, split, torch.device('cpu'))
        native = suite.helpers.native_predictions(args.output, frozen, split, args.library)
        verr, perr = float(abs(fp[0]-native[0]).max()), float(abs(fp[1]-native[1]).max())
        assert verr < .00051 and perr < .00001, (verr, perr)
        report['metrics'][split] = dict(positions=len(fp[0]), maxValueError=verr, maxPolicyError=perr,
                                             native=suite.helpers.metrics(*native, frozen, split))
    suite.helpers.write_json(args.output.with_suffix('.export.json'), report)
    print(report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--library', type=Path, default=ROOT/'.build/native-light-cnn-v1.20/libothello.dylib')
    main(p.parse_args())
