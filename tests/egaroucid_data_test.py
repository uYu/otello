"""Independent replay checks for the large-corpus preprocessing boundary."""
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('egaroucid_prepare', ROOT / 'scripts/prepare-egaroucid-data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def flips(board, turn, move):
    if board[move]:
        return []
    result = []
    x, y = move % 8, move // 8
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
        xx, yy = x + dx, y + dy
        ray = []
        while 0 <= xx < 8 and 0 <= yy < 8 and board[yy * 8 + xx] == -turn:
            ray.append(yy * 8 + xx)
            xx, yy = xx + dx, yy + dy
        if ray and 0 <= xx < 8 and 0 <= yy < 8 and board[yy * 8 + xx] == turn:
            result += ray
    return result


def game(seed):
    rng = random.Random(seed)
    board = np.zeros(64, dtype=np.int8)
    board[28] = board[35] = 1
    board[27] = board[36] = -1
    turn, passes, moves, states = 1, 0, [], []
    while True:
        legal = [m for m in range(64) if flips(board, turn, m)]
        if not legal:
            turn = -turn
            legal = [m for m in range(64) if flips(board, turn, m)]
            if not legal:
                break
            passes += 1
        move = rng.choice(legal)
        states.append((board.copy(), turn, move))
        changed = flips(board, turn, move)
        board[changed] = board[move] = turn
        moves.append(move)
        turn = -turn
    return ''.join(chr(97 + m % 8) + str(m // 8 + 1) for m in moves), states, int(board.sum()), passes


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.directory.name)
        cls.converter = cls.folder / 'convert'
        subprocess.run(['c++', '-std=c++17', '-O2', '-I' + str(ROOT / 'cpp'),
                        str(ROOT / 'scripts/prepare-egaroucid.cpp'), '-o', str(cls.converter)], check=True)
        cls.transcript, cls.states, cls.diff, cls.passes = game(4)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def convert(self, transcript, random_prefix=8, holdout=b''):
        (self.folder / 'holdout.bin').write_bytes(holdout)
        result = subprocess.run([str(self.converter), str(random_prefix), str(self.folder / 'games.bin'),
                                 str(self.folder / 'samples.bin'), str(self.folder / 'holdout.bin'),
                                 '100000000', '20261027'], input=transcript, text=True,
                                capture_output=True, check=True)
        return json.loads(result.stdout), np.fromfile(self.folder / 'samples.bin', dtype=module.DTYPE)

    def test_replay_passes_random_boundary_and_rejects(self):
        # Try fixed independent games until an opening-owned train game with a pass.
        for seed in range(100):
            transcript, states, diff, passes = game(seed)
            report, rows = self.convert(transcript + '\r\n')
            if len(rows) and passes:
                break
        else:
            self.fail('No train game with pass in fixture seeds')
        self.assertEqual(report['accepted'], 1)
        self.assertEqual(report['forcedPasses'], passes)
        self.assertEqual(len(rows), len(states) - 8)
        for row in rows:
            ply = int(row['ply'])
            b, turn, move = states[ply]
            key = module.canonical_keys(b[None, :], np.array([turn]))[0]
            self.assertEqual((int(row['own']), int(row['opp'])), tuple(map(int, key)))
            self.assertEqual(int(row['disc']), diff * turn)
            relative = np.array([((int(row['own']) >> i) & 1) - ((int(row['opp']) >> i) & 1) for i in range(64)])
            mask = sum(1 << m for m in range(64) if flips(relative, 1, m))
            self.assertEqual(int(row['legal']), mask)
            self.assertTrue(mask & (1 << int(row['move'])))
        blocked = np.array([[rows[0]['own'], rows[0]['opp']]], dtype='<u8').tobytes()
        report2, rows2 = self.convert(transcript + '\n', holdout=blocked)
        self.assertEqual(report2['excludedHoldout'], 1)
        self.assertEqual(len(rows2), len(rows) - 1)
        report, _ = self.convert(transcript + '\n' + transcript[:-2] + '\n' + 'a1a1\n' + 'xx\n')
        self.assertEqual(report['accepted'], 1)
        self.assertEqual(sum(report['rejected'].values()), 3)
        self.assertEqual(report['rejected']['incomplete'], 1)

    def test_sample_aggregation_preserves_holdouts(self):
        for seed in range(100):
            transcript, states, _, _ = game(seed)
            _, rows = self.convert(transcript + '\n')
            if len(rows):
                break
        rows = rows[:1]
        repeated = rows.copy()
        other = rows.copy()
        other['game'] += 1
        other['outcome'] = -rows['outcome']
        other['disc'] = -rows['disc']
        legal_moves = [m for m in range(64) if int(rows[0]['legal']) & (1 << m)]
        other['move'] = legal_moves[-1]
        groups = self.folder / 'groups'
        groups.mkdir(exist_ok=True)
        np.concatenate([rows, repeated, other]).tofile(groups / '08.samples.bin')
        frozen = {}
        labels = {}
        for split in ('test', 'validation'):
            b = np.zeros((1, 64), dtype=np.int8)
            b[0, 0], b[0, 63] = 1, -1
            frozen.update({split + '_board': b, split + '_turn': np.ones(1, dtype=np.float32),
                           split + '_target': np.array([0.5], dtype=np.float32)})
            labels[split + '_move'] = np.zeros(1, dtype=np.int16)
            labels[split + '_legal'] = np.ones((1, 65), dtype=bool)
        np.savez(self.folder / 'frozen-labels.npz', **labels)
        report = module.build_cache(SimpleNamespace(output=self.folder, seed=3, positions_per_stage=10,
                                    frozen_labels=self.folder / 'frozen-labels.npz'), {8: {}}, frozen)
        self.assertEqual(report['positions'], 1)
        self.assertEqual(report['trajectoryDuplicatesRemoved'], 1)
        data = np.load(self.folder / 'cache/positions.npz')
        policy = np.load(self.folder / 'cache/labels.npz')
        self.assertEqual(float(data['train_outcome'][0]), 0)
        self.assertEqual(int(data['train_count'][0]), 2)
        self.assertAlmostEqual(float(policy['train_policy'][0].sum()), 1)
        np.testing.assert_array_equal(data['test_target'], frozen['test_target'])


if __name__ == '__main__':
    unittest.main()
