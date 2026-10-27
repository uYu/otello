"""Native dataset/model boundary tests; run after the CMake build."""
import ctypes
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / '.build/native/otello'
CONVERTER = ROOT / '.build/native/otello_dataset'


class TrainingBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = ctypes.CDLL(str(ROOT / ('.build/native/libothello.' + ('dylib' if sys.platform == 'darwin' else 'so'))))
        ptr = ctypes.POINTER(ctypes.c_int)
        cls.cells_type = ctypes.c_int * 64
        cls.out_type = ctypes.c_int * 136
        cls.lib.ot_legal.argtypes = [ptr, ctypes.c_int, ptr]
        cls.lib.ot_play.argtypes = [ptr, ctypes.c_int, ctypes.c_int, ptr]
        cls.lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
        cls.lib.ot_nnue_load.restype = ctypes.c_void_p
        cls.lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]
        cls.lib.ot_nnue_evaluate.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ptr]
        cls.lib.ot_nnue_search.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int, ctypes.c_uint, ptr]
        cls.lib.ot_nnue_error.restype = ctypes.c_char_p
        rng = random.Random(20261027)
        for _ in range(100):
            cells = cls.cells_type()
            cells[28] = cells[35] = 1
            cells[27] = cells[36] = -1
            turn, moves, signs, passes = 1, [], [], 0
            while True:
                out = cls.out_type()
                n = cls.lib.ot_legal(cells, turn, out)
                if not n:
                    if not cls.lib.ot_legal(cells, -turn, out):
                        break
                    move = -1
                    passes += 1
                else:
                    move = out[rng.randrange(n)]
                    moves.append(move)
                    signs.append(turn)
                if cls.lib.ot_play(cells, turn, move, out) != 1:
                    raise RuntimeError('Invalid fixture')
                cells, turn = cls.cells_type(*out[:64]), out[64]
                if out[65]:
                    break
            if passes and len(moves) == 60:
                cls.moves, cls.signs, cls.final_diff = moves, signs, sum(cells)
                break
        else:
            raise RuntimeError('No forced-pass fixture found')

    def archive(self, records):
        header = bytearray(16)
        struct.pack_into('<I', header, 4, len(records))
        header[12] = 8
        return header + b''.join(records)

    def wthor_record(self, score=None):
        record = bytearray(68)
        record[6] = record[7] = (64 + (self.final_diff if score is None else score)) // 2
        record[8:] = bytes((m // 8 + 1) * 10 + m % 8 + 1 for m in self.moves)
        return record

    def test_corpora_passes_symmetry_and_score_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            wtb, gam, output = [folder / name for name in ('sample.wtb', 'sample.gam', 'games.jsonl')]
            # A bad-score copy must not suppress a later valid transcript during dedup.
            wrong = self.final_diff - 2 if self.final_diff > -64 else self.final_diff + 2
            wtb.write_bytes(self.archive([self.wthor_record(wrong), self.wthor_record()]))
            def line(moves):
                return ''.join(('+' if side == 1 else '-') + chr(97 + m % 8) + str(m // 8 + 1)
                               for m, side in zip(moves, self.signs)) + f': {self.final_diff:+03d} 10\n'
            gam.write_text(line(self.moves) + line([63 - m for m in self.moves]) + '+a1: 0 10\n')
            run = subprocess.run([str(CONVERTER), '--input', str(wtb), '--input', str(gam), '--output', str(output)],
                                 capture_output=True, text=True, check=True)
            stats = json.loads(run.stdout)
            self.assertEqual(stats['accepted'], 1)
            self.assertEqual(stats['duplicates'], 2)
            self.assertEqual(stats['scoreMismatch'], 1)
            self.assertEqual(stats['rejected'], 1)
            rows = [json.loads(row) for row in output.read_text().splitlines()]
            self.assertIn(-1, rows[0]['moves'])
            self.assertTrue(rows[0]['complete'])
            self.assertEqual(rows[0]['blackResult'], self.final_diff)

    def test_archive_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder) / 'bad.wtb'
            bad.write_bytes(self.archive([self.wthor_record()])[:-1])
            run = subprocess.run([str(CONVERTER), '--input', str(bad), '--output', str(Path(folder) / 'out.jsonl')],
                                 capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('length', run.stderr)

    def test_nnue_loader_and_c_api(self):
        for width in (32, 256):
            header = struct.pack('<8sIIII', b'OTNNUE1\0', 128, width, 256, 256)
            weights = [((i * 17 + j * 13) % 61 - 30) for i in range(128) for j in range(width)]
            bias = [50] * width
            head = [((j * 19) % 101 - 50) for j in range(width)]
            binary = header + struct.pack('<' + 'h' * len(weights), *weights)
            binary += struct.pack('<' + 'i' * width, *bias) + struct.pack('<' + 'h' * width, *head)
            with tempfile.TemporaryDirectory() as folder:
                model = Path(folder) / 'model with spaces.otnnue'
                model.write_bytes(binary)
                handle = self.lib.ot_nnue_load(str(model).encode())
                self.assertTrue(handle)
                try:
                    cells, score, opposite = self.cells_type(), ctypes.c_int(), ctypes.c_int()
                    cells[28] = cells[35] = 1
                    cells[27] = cells[36] = -1
                    out = self.out_type()
                    self.assertEqual(self.lib.ot_play(cells, 1, 19, out), 1)
                    cells = self.cells_type(*out[:64])
                    self.assertEqual(self.lib.ot_nnue_evaluate(handle, cells, 1, ctypes.byref(score)), 1)
                    self.assertEqual(self.lib.ot_nnue_evaluate(handle, cells, -1, ctypes.byref(opposite)), 1)
                    self.assertEqual(score.value, -opposite.value)
                    board = ''.join('B' if c == 1 else 'W' if c == -1 else '.' for c in cells)
                    run = subprocess.run([str(ENGINE), 'analyze', '--network', str(model), '--board', board,
                                          '--turn', '-1', '--time', '60000', '--depth', '3'],
                                         capture_output=True, text=True, check=True)
                    expected = json.loads(run.stdout)
                    self.assertEqual(self.lib.ot_nnue_search(handle, cells, -1, 0, 3, 2, 1, out), 1)
                    self.assertEqual((out[0], out[1]), (self.coord(expected['move']), expected['score']))
                    self.assertEqual(self.lib.ot_nnue_evaluate(handle, None, 1, ctypes.byref(score)), -1)
                    self.assertTrue(self.lib.ot_nnue_error())
                finally:
                    self.lib.ot_nnue_destroy(handle)
                variants = [b'garbage', binary[:-1], binary + b'x',
                            struct.pack('<8sIIII', b'OTNNUE1\0', 128, 1024, 256, 256) + binary[24:]]
                for invalid in variants:
                    model.write_bytes(invalid)
                    self.assertFalse(self.lib.ot_nnue_load(str(model).encode()))
                    self.assertTrue(self.lib.ot_nnue_error())

    @staticmethod
    def coord(text):
        return (int(text[1]) - 1) * 8 + ord(text[0]) - 65


if __name__ == '__main__':
    unittest.main()
