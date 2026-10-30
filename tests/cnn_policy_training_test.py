import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cnn_suite', ROOT/'scripts/train-cnn-policy-suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class CnnPolicyTests(unittest.TestCase):
    def test_two_conv_joint_gradients_color_and_dynamic_export(self):
        torch.set_num_threads(2)
        torch.manual_seed(54)
        board = torch.from_numpy(np.random.default_rng(54).integers(-1, 2, (9, 64), dtype=np.int8))
        turn = torch.tensor([1., -1., 1., -1., 1., -1., 1., -1., 1.])
        for channels in (8, 16, 32):
            net = suite.CnnPolicyValue(channels)
            self.assertEqual(sum(isinstance(m, torch.nn.Conv2d) for m in net.modules()), 2)
            value, policy = net(board, turn)
            self.assertEqual(tuple(policy.shape), (9, 65))
            for a, b in zip((value, policy), net(-board, -turn)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            loss = (value-1).square().mean()+torch.nn.functional.cross_entropy(policy, torch.arange(9))
            loss.backward()
            for layer in (net.conv1, net.conv2, net.value, net.policy):
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
            with tempfile.TemporaryDirectory() as folder:
                path = str(Path(folder)/'model.pt')
                torch.jit.trace(net.eval(), (board[:2], turn[:2])).save(path)
                loaded = torch.jit.load(path)
                for a, b in zip(net(board, turn), loaded(board, turn)):
                    torch.testing.assert_close(a, b)

    def test_d4_recorded_move_and_pass_coordinates(self):
        perms = torch.from_numpy(suite.helpers.policy.value.symmetries())
        inverse = torch.argsort(perms, 1)
        for symmetry in range(8):
            for move in range(64):
                destination = inverse[symmetry, move]
                self.assertEqual(perms[symmetry, destination].item(), move)
            moves = torch.tensor([64])
            mapped = torch.where(moves == 64, 64, inverse[symmetry, moves.clamp(max=63)])
            self.assertEqual(mapped.item(), 64)


if __name__ == '__main__':
    unittest.main()
