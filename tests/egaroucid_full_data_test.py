import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from egaroucid_data_test import game

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


full = load('full_prepare', 'process-egaroucid-games.py')
sampled = load('sampled_prepare', 'prepare-egaroucid-data.py')


class FullCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.directory.name)
        cls.converter = cls.folder / 'expand'
        subprocess.run(['c++', '-std=c++17', '-O2', '-I' + str(ROOT / 'cpp'),
                        str(ROOT / 'scripts/expand-egaroucid.cpp'), '-o', str(cls.converter)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_exhaustive_expansion_and_symmetry(self):
        transcript, states, diff, _ = game(4)
        record = np.zeros(1, dtype=full.GAME)
        record['moves'] = 255
        record['moves'][0, :len(states)] = [move for _, _, move in states]
        record['random'], record['length'], record['disc'], record['split'] = 8, len(states), diff, 1
        record.tofile(self.folder / 'games.bin')
        blocked = sampled.canonical_keys(states[8][0][None, :], np.array([states[8][1]]))
        (self.folder / 'frozen.bin').write_bytes(blocked.tobytes())
        result = subprocess.run([str(self.converter), str(self.folder / 'games.bin'),
                                 str(self.folder / 'buckets'), str(self.folder / 'frozen.bin'), '4'],
                                capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        rows = np.concatenate([np.fromfile(self.folder / f'buckets/{i}.bin', dtype=full.OBS) for i in range(4)])
        rows.sort(order='ply')
        self.assertEqual(len(rows), len(states)-8)
        self.assertEqual(report['positions'], len(rows))
        self.assertEqual(report['frozenObservations'], 1)
        self.assertTrue((rows['split'] == 1).all())  # Expansion keeps every source split.
        for row in rows:
            board, turn, _ = states[int(row['ply'])]
            expected = sampled.canonical_keys(board[None, :], np.array([turn]))[0]
            self.assertEqual((int(row['own']), int(row['opp'])), tuple(map(int, expected)))
            self.assertEqual(int(row['disc']), diff*turn)

    def test_exact_counts_and_split_isolation(self):
        rows = np.zeros(4, dtype=full.OBS)
        rows['own'], rows['opp'], rows['legal'], rows['ply'] = 1, 2, (1 << 4) | (1 << 5), 8
        rows['move'] = [4, 5, 5, 4]
        rows['outcome'] = [1, -1, 0, 1]
        rows['disc'] = [2, -30, 0, 64]
        rows['split'] = [0, 0, 0, 1]
        arrays, stats = full.aggregate_rows(rows)
        self.assertEqual(stats['distinctBoards'], 1)
        self.assertEqual(stats['positionSplitPairs'], 2)
        self.assertEqual(stats['trainPositions'], 1)
        i = int(np.flatnonzero(arrays['split'] == 0)[0])
        self.assertEqual(int(arrays['count'][i]), 3)
        np.testing.assert_array_equal(arrays['outcome_counts'][i], [1, 1, 1])
        self.assertEqual(int(arrays['disc_sum'][i]), -28)
        a, b = map(int, arrays['policy_offsets'][i:i+2])
        self.assertEqual(dict(zip(map(int, arrays['policy_move'][a:b]), map(int, arrays['policy_count'][a:b]))), {4: 1, 5: 2})
        rows['frozen'] = 1
        _, stats = full.aggregate_rows(rows)
        self.assertEqual(stats['trainPositions'], 0)
        rows['move'][0] = 63
        with self.assertRaisesRegex(ValueError, 'Illegal policy'):
            full.aggregate_rows(rows)


if __name__ == '__main__':
    unittest.main()
