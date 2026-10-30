"""Training cache isolation and frozen benchmark protocol regression tests."""
from contextlib import redirect_stdout
import ctypes
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import subprocess
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / ('.build/native/libothello.' + ('dylib' if sys.platform == 'darwin' else 'so'))
spec = importlib.util.spec_from_file_location('nnue_trainer', ROOT / 'scripts/train-nnue.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class TurnArchitecture(unittest.TestCase):
    def test_initiative_and_color_relabeling(self):
        net = trainer.ValueNet(32, 'turn')
        with torch.no_grad():
            net.head.weight.zero_()
            net.head.bias.fill_(0.25)
        board = torch.zeros((2, 64), dtype=torch.int8)
        board[:, 28] = 1
        board[:, 27] = -1
        turn = torch.tensor([1.0, -1.0])
        output = net(trainer.features(board), turn)
        self.assertTrue(torch.all(output > 0))
        torch.testing.assert_close(output, net(trainer.features(-board), -turn), rtol=0, atol=0)

    def test_both_formats_match_cpp_and_reject_truncated_heads(self):
        lib = ctypes.CDLL(str(LIBRARY))
        lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
        lib.ot_nnue_load.restype = ctypes.c_void_p
        lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]
        lib.ot_nnue_evaluate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        torch.manual_seed(781)
        board = np.random.default_rng(781).integers(-1, 2, size=(12, 64), dtype=np.int8)
        turn = np.array([1, -1] * 6)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.otnnue'
            for architecture in ('legacy', 'turn'):
                for hidden in (32, 256):
                    net = trainer.ValueNet(hidden, architecture)
                    arrays = trainer.quantized(net)
                    binary = struct.pack('<8sIIII', b'OTNNUE1\0' if architecture == 'legacy' else b'OTNNUE2\0', 128, hidden, 256, 256)
                    binary += b''.join(x.tobytes() for x in arrays)
                    path.write_bytes(binary)
                    handle = lib.ot_nnue_load(str(path).encode())
                    self.assertTrue(handle)
                    try:
                        expected = trainer.quantized_predict(arrays, board, turn) * 1000
                        for cells, actor, value in zip(board, turn, expected):
                            result = ctypes.c_int()
                            self.assertEqual(lib.ot_nnue_evaluate(handle, (ctypes.c_int * 64)(*map(int, cells)), int(actor), ctypes.byref(result)), 1)
                            self.assertLessEqual(abs(result.value - value), 0.501)
                    finally:
                        lib.ot_nnue_destroy(handle)
                    path.write_bytes(binary[:-1])
                    self.assertFalse(lib.ot_nnue_load(str(path).encode()))


class TrainingIsolation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.lib = ctypes.CDLL(str(LIBRARY))
        ptr = ctypes.POINTER(ctypes.c_int)
        self.lib.ot_legal.argtypes = [ptr, ctypes.c_int, ptr]
        self.lib.ot_play.argtypes = [ptr, ctypes.c_int, ctypes.c_int, ptr]
        self.cells_type, self.out_type = ctypes.c_int * 64, ctypes.c_int * 136
        rows = []
        for index in range(12):
            cells = self.cells_type()
            cells[28] = cells[35] = 1
            cells[27] = cells[36] = -1
            turn, moves, rng = 1, [], random.Random(index + 1200)
            while True:
                out = self.out_type()
                n = self.lib.ot_legal(cells, turn, out)
                if not n and not self.lib.ot_legal(cells, -turn, out):
                    break
                move = out[rng.randrange(n)] if n else -1
                self.assertEqual(self.lib.ot_play(cells, turn, move, out), 1)
                moves.append(move)
                cells, turn = self.cells_type(*out[:64]), out[64]
                if out[65]:
                    break
            rows.append(dict(id=str(index), source='wthor' if index % 2 else 'logistello',
                             split=('train', 'validation', 'test')[index % 3], complete=True,
                             moves=moves, blackResult=sum(cells)))
        self.games = self.folder / 'games.jsonl'
        self.games.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def tearDown(self):
        self.temp.cleanup()

    def arguments(self, name):
        return SimpleNamespace(seed=20261027, max_games=30, max_positions=60, games=self.games,
                               library=LIBRARY, cache=self.folder / name,
                               holdout_cache=None, all_training_games=False, train_positions=0)

    def prepare(self, args):
        with redirect_stdout(io.StringIO()):
            return trainer.prepare(args)

    @staticmethod
    def keys(data, split):
        permutations = trainer.symmetries()
        return {min((board * turn).astype(np.int8)[p].tobytes() for p in permutations)
                for board, turn in zip(data[split + '_board'], data[split + '_turn'])}

    def test_expansion_keeps_holdouts_and_excludes_symmetric_transpositions(self):
        base_args = self.arguments('base.npz')
        original = self.prepare(base_args)
        args = self.arguments('expanded.npz')
        args.holdout_cache = base_args.cache
        args.all_training_games, args.train_positions = True, 24
        expanded = self.prepare(args)
        self.assertEqual(len(expanded['train_board']), 24)
        for split in ('validation', 'test'):
            for key in ('board', 'turn', 'target'):
                np.testing.assert_array_equal(original[split + '_' + key], expanded[split + '_' + key])
            self.assertFalse(self.keys(expanded, 'train') & self.keys(expanded, split))
        repeated = self.prepare(args)
        for key in expanded:
            np.testing.assert_array_equal(expanded[key], repeated[key])

    def test_incompatible_holdout_is_rejected(self):
        args = self.arguments('base.npz')
        self.prepare(args)
        args.holdout_cache, args.cache = args.cache, self.folder / 'expanded.npz'
        args.seed += 1
        with self.assertRaisesRegex(ValueError, 'different data snapshot or seed'):
            self.prepare(args)

    def test_corrupt_cache_is_rejected_before_training(self):
        args = self.arguments('base.npz')
        self.prepare(args)
        with args.cache.open('ab') as output:
            output.write(b'corrupt')
        args.prepare, args.threads = False, 2
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            trainer.train(args)

    def test_candidate_match_uses_same_mcts_and_replays(self):
        output = self.folder / 'candidate-match.json'
        model = ROOT / 'models/nnue256-v1.2.otnnue'
        subprocess.run([sys.executable, str(ROOT / 'scripts/compare-nnue-candidate.py'),
                        '--model', str(model), '--baseline-manifest', str(ROOT/'models/v1.7-default.json'), '--library', str(LIBRARY), '--games', '2',
                        '--milliseconds', '2', '--output', str(output)],
                       capture_output=True, text=True, check=True, timeout=30)
        report = json.loads(output.read_text())
        self.assertTrue(report['complete'])
        self.assertEqual(report['games'], 2)
        self.assertEqual({c['search'] for c in report['configs']}, {'mcts'})
        self.assertEqual(len({c['modelSha256'] for c in report['configs']}), 1)
        self.assertEqual(report['records'][0]['seed'], report['records'][1]['seed'])
        self.assertNotEqual(report['records'][0]['black'], report['records'][1]['black'])
        # Resume only a completed-game checkpoint, never the partial live board.
        first=report['records'][0]
        report.update(complete=False,games=1,records=[first])
        output.write_text(json.dumps(report))
        checkpoint=output.read_bytes()
        common=[sys.executable,str(ROOT/'scripts/compare-nnue-candidate.py'),
                '--model',str(model),'--baseline-manifest',str(ROOT/'models/v1.7-default.json'),'--library',str(LIBRARY),'--games','2',
                '--output',str(output),'--resume']
        rejected=subprocess.run([*common,'--milliseconds','3'],capture_output=True,text=True,timeout=30)
        self.assertNotEqual(rejected.returncode,0)
        self.assertIn('provenance mismatch',rejected.stderr)
        self.assertEqual(output.read_bytes(),checkpoint)
        subprocess.run([*common,'--milliseconds','2'],capture_output=True,text=True,check=True,timeout=30)
        resumed=json.loads(output.read_text())
        self.assertTrue(resumed['complete'])
        self.assertEqual(resumed['games'],2)
        self.assertEqual(resumed['records'][0],first)
        self.assertEqual(resumed['runnerSegments'][-1]['firstGame'],2)
        self.assertEqual(resumed['runnerSegments'][-1]['lastGame'],2)
        self.assertEqual(resumed['resumeHistory'][0]['previousReportSha256'],hashlib.sha256(checkpoint).hexdigest())
        self.assertIsNone(resumed['currentGame'])

    def test_policy_labels_training_export_and_cpp_parity(self):
        args = self.arguments('positions.npz')
        self.prepare(args)
        labels = self.folder / 'labels.npz'
        subprocess.run([sys.executable, str(ROOT/'scripts/prepare-policy-data.py'),
                        '--cache', str(args.cache), '--games', str(self.games),
                        '--library', str(LIBRARY), '--output', str(labels)],
                       check=True,capture_output=True,text=True,timeout=30)
        data=dict(np.load(args.cache));policy=dict(np.load(labels))
        for split in ('train','validation','test'):
            for board,turn,move,mask in zip(data[split+'_board'],data[split+'_turn'],policy[split+'_move'],policy[split+'_legal']):
                cells=self.cells_type(*map(int,board));out=self.out_type()
                n=self.lib.ot_legal(cells,int(turn),out)
                self.assertEqual(set(np.flatnonzero(mask)),set(out[:n]))
                self.assertIn(int(move),out[:n])
                for permutation in trainer.symmetries():
                    new_move=int(np.argsort(permutation)[move])
                    transformed=self.cells_type(*map(int,board[permutation]))
                    n=self.lib.ot_legal(transformed,int(turn),out)
                    self.assertIn(new_move,out[:n])
        output=self.folder/'policy.otnnue'
        subprocess.run([sys.executable,str(ROOT/'scripts/train-policy.py'),
                        '--cache',str(args.cache),'--labels',str(labels),'--output',str(output),
                        '--library',str(LIBRARY),'--epochs','1','--batch-size','16'],
                       check=True,capture_output=True,text=True,timeout=30)
        report=json.loads(output.with_suffix('.json').read_text())
        self.assertTrue(report['valueBytesUnchanged'] and report['valueParity'])
        self.assertLess(report['cppMaxProbabilityError'],1e-10)
        self.assertEqual(report['cppParityPositions'],6)
        self.lib.ot_nnue_load.argtypes=[ctypes.c_char_p];self.lib.ot_nnue_load.restype=ctypes.c_void_p
        binary=output.read_bytes()
        bad_dims=bytearray(binary);struct.pack_into('<I',bad_dims,67612,31)
        for invalid in (binary[:-1],binary+b'x',bad_dims):
            output.write_bytes(invalid)
            self.assertFalse(self.lib.ot_nnue_load(str(output).encode()))

    def test_comparison_reuses_openings_and_preserves_replayable_records(self):
        bank = self.folder / 'openings.json'
        model = ROOT / 'models/nnue256-v1.otnnue'
        for ms in (1, 2):
            output = self.folder / f'match-{ms}.json'
            subprocess.run(['python3', str(ROOT / 'scripts/compare-evaluators.py'), '--model', str(model),
                            '--library', str(LIBRARY),
                            '--games', '2', '--milliseconds', str(ms), '--openings', str(bank),
                            '--output', str(output)], cwd=ROOT, capture_output=True, text=True, check=True)
            report = json.loads(output.read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(report['modelSha256'], hashlib.sha256(model.read_bytes()).hexdigest())
            self.assertEqual(report['games'], 2)
            self.assertEqual(report['records'][0]['opening'], report['records'][1]['opening'])
            for game in report['records']:
                cells, turn = self.cells_type(), 1
                cells[28] = cells[35] = 1
                cells[27] = cells[36] = -1
                for move in game['opening'] + game['moves']:
                    out = self.out_type()
                    self.assertEqual(self.lib.ot_play(cells, turn, move, out), 1)
                    cells, turn = self.cells_type(*out[:64]), out[64]
                self.assertTrue(out[65])
                self.assertEqual(sum(cells) * game['modelSide'], game['modelDiscDifference'])
            if ms == 1:
                first_bank = bank.read_bytes()
                first_hash = report['openingBankSha256']
            else:
                self.assertEqual(first_bank, bank.read_bytes())
                self.assertEqual(first_hash, report['openingBankSha256'])


if __name__ == '__main__':
    unittest.main()
