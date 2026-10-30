import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from pattern_model import encode, parity
from pattern_structure import StructureNet
from pattern_phase_value import PhaseValueNet, export


class PhaseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        suffix = 'dylib' if sys.platform == 'darwin' else 'so'
        self.library = Path(os.environ.get('OTELLO_PATTERN_LIBRARY', ROOT/f'.build/native-pattern-phase-value-v1.46/libothello.{suffix}'))
        rng = np.random.default_rng(46146)
        self.board = rng.integers(-1, 2, (512, 64), dtype=np.int8)
        # Every valid empty count, including both phase boundaries.
        for i in range(len(self.board)):
            self.board[i] = rng.choice([-1, 1], 64)
            self.board[i, rng.choice(64, i % 61, replace=False)] = 0
        self.turn = np.array([1, -1]*256, np.float32)
        self.phase = torch.from_numpy((64-np.count_nonzero(self.board, axis=1)).astype(np.float32)/60)
        self.codes = torch.from_numpy(encode(self.board, self.turn).astype(np.int64))

    def test_initial_baseline_equivalence_and_gradients(self):
        torch.manual_seed(20261029);base = StructureNet('symmetry')
        torch.manual_seed(20261029);net = PhaseValueNet()
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, net.state_dict()[key]), key)
        expected = base(self.codes, self.phase);actual = net(self.codes, self.phase)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        sum(x.square().sum() for x in actual).backward()
        self.assertGreater(float(net.early_value.weight.grad.abs().sum()), 0)
        self.assertGreater(float(net.head.weight.grad[:4].abs().sum()), 0)
        self.assertGreater(float(net.head.weight.grad[4:].abs().sum()), 0)

    def test_phase_arithmetic_policy_invariance_and_native(self):
        torch.manual_seed(46146);net = PhaseValueNet().eval()
        early = torch.full((61, 4), 257/256);late = torch.full((61, 4), -129/256)
        phase = torch.arange(61).float()/60
        mixed = net.blend(early, late, phase)
        n = np.clip(np.arange(61)-36, 0, 12)
        wanted = (n*257+(12-n)*-129)/12
        wanted = np.sign(wanted)*np.floor(np.abs(wanted)+.5)
        np.testing.assert_array_equal(mixed[:, 0].numpy(), wanted/256)
        with torch.no_grad():
            policy_before = net(self.codes, self.phase)[2].clone()
            net.embedding.weight.mul_(8)
            # Keep the shared part for policy comparison, then alter only value.
            policy_before = net(self.codes, self.phase)[2].clone()
            net.early_value.weight.uniform_(-1, 1);net.early_value.bias.uniform_(-1, 1)
            net.head.weight[:4].uniform_(-1, 1);net.head.bias[:4].uniform_(-1, 1)
            expected = tuple(x.numpy() for x in net(self.codes, self.phase))
            torch.testing.assert_close(torch.from_numpy(expected[2]), policy_before, atol=0, rtol=0)
            grid = np.arange(64).reshape(8, 8)
            for mirror in [False, True]:
                for rotation in range(4):
                    perm = np.rot90(np.fliplr(grid) if mirror else grid, rotation).reshape(64)
                    prediction = tuple(x.numpy() for x in net(torch.from_numpy(encode(self.board[:, perm], self.turn).astype(np.int64)), self.phase))
                    np.testing.assert_array_equal(prediction[0], expected[0]);np.testing.assert_array_equal(prediction[1], expected[1])
                    np.testing.assert_allclose(prediction[2][:, np.argsort(perm)], expected[2], atol=2e-6, rtol=0)
            swapped = net(torch.from_numpy(encode(-self.board, -self.turn).astype(np.int64)), self.phase)
            for a, b in zip(swapped, expected):
                np.testing.assert_array_equal(a.numpy(), b)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.otpat';binary = export(net, path)
            self.assertEqual(len(binary), 1833972)
            self.assertLess(parity(self.library, path, self.board, self.turn, expected)['maxPolicyLogitError'], 2e-5)


if __name__ == '__main__':
    unittest.main()
