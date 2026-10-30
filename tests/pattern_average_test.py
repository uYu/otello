"""Check trainable-only checkpoint means and Pattern export parity."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('pattern_average', ROOT/'scripts/train-pattern-average.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class CheckpointAverageTest(unittest.TestCase):
    def test_mean_buffers_and_export(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            net = a.PatternNet()
            parameters = set(dict(net.named_parameters()))
            original = net.state_dict()
            paths = []
            for index, value in enumerate((.125, .375)):
                state = {k: torch.full_like(v, value) if k in parameters else v.clone()
                         for k, v in original.items()}
                path = directory/f'{index}.pt'
                torch.save(state, path); paths.append(path)
            mean = a.average_checkpoints(paths)
            for key, value in mean.items():
                self.assertTrue(torch.equal(value, torch.full_like(value, .25) if key in parameters else original[key]))
            net.load_state_dict(mean)
            path = directory/'mean.otpat'
            binary = a.export(net, path, outcome=True)
            self.assertEqual(binary[:8], b'OTPAT2\0\0')
            board = np.zeros((1, 64), dtype=np.int8)
            board[0, [28, 35]] = 1; board[0, [27, 36]] = -1
            turn = np.ones(1, dtype=np.int8)
            pred = a.helper.predict(net, board, turn)
            result = a.parity(a.LIB, path, board, turn, pred)
            self.assertLess(result['maxPolicyLogitError'], 2e-5)
            # A single checkpoint must remain exactly itself.
            single = a.average_checkpoints(paths[:1])
            source = torch.load(paths[0], weights_only=True)
            self.assertTrue(all(torch.equal(single[k], source[k]) for k in single))
            state = dict(source); state['pool'] = source['pool']+1
            bad = directory/'bad.pt'; torch.save(state, bad)
            with self.assertRaisesRegex(ValueError, 'buffer'):
                a.average_checkpoints([paths[0], bad])
            state = dict(source); state['head.weight'] = source['head.weight'][:, :-1]
            torch.save(state, bad)
            with self.assertRaisesRegex(ValueError, 'shape/dtype'):
                a.average_checkpoints([bad])
            state = dict(source); state['head.bias'] = torch.full_like(source['head.bias'], float('nan'))
            torch.save(state, bad)
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                a.average_checkpoints([bad])
            with self.assertRaises(ValueError):
                a.average_checkpoints([])


if __name__ == '__main__':
    unittest.main()
