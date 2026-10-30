import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mixed', ROOT/'scripts/train-mixed-5m.py')
mixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mixed)


class MixedTrainingTest(unittest.TestCase):
    def test_soft_labels_from_noncontiguous_csr_rows(self):
        data = dict(own=np.array([1, 2, 4], np.uint64),
                    opponent=np.array([8, 8, 8], np.uint64),
                    legal=np.array([48, 16, 96], np.uint64),
                    count=np.array([2, 1, 4], np.uint32),
                    outcome_counts=np.array([[1, 0, 1], [0, 0, 1], [1, 1, 2]], np.uint32),
                    policy_offsets=np.array([0, 2, 3, 5], np.uint64),
                    policy_move=np.array([4, 5, 4, 5, 6], np.uint8),
                    policy_count=np.array([1, 1, 1, 1, 3], np.uint32))
        board, turn, value, legal, soft = mixed.shard_batch(data, np.array([2, 0]))
        np.testing.assert_allclose(value, [.25, 0])
        np.testing.assert_allclose(soft[0, [5, 6]], [.25, .75])
        np.testing.assert_allclose(soft[1, [4, 5]], [.5, .5])
        self.assertFalse((soft[~legal] > 0).any())
        np.testing.assert_equal(board[:, 3], [-1, -1])
        np.testing.assert_equal(turn, [1, 1])
        b, l, p = mixed.augment(board, legal, soft, mixed.policy.value.symmetries(), np.random.default_rng(7))
        np.testing.assert_allclose(p.sum(1), 1)
        self.assertFalse((p[~l] > 0).any())
        for i in range(2):
            self.assertTrue(any(np.array_equal(b[i], board[i, perm]) and
                                np.array_equal(p[i, :64], soft[i, perm]) and
                                np.array_equal(l[i, :64], legal[i, perm])
                                for perm in mixed.policy.value.symmetries()))

    def test_approved_model_roundtrip_and_frozen_transform(self):
        parent = (ROOT/'models/nnue256-outcome-v1.9.otnnue').read_bytes()
        net = mixed.MixedNet(parent)
        self.assertEqual(mixed.export(net, parent), parent)
        self.assertTrue(all(not p.requires_grad for p in net.transform.parameters()))
        self.assertTrue(all(p.requires_grad for p in net.head.parameters()))
        self.assertTrue(all(p.requires_grad for p in net.hidden.parameters()))
        self.assertTrue(all(p.requires_grad for p in net.policy.parameters()))


if __name__ == '__main__':
    unittest.main()
