import ctypes
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('resnet_trainer', ROOT / 'scripts/train-resnet.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class ResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1532)
        suffix = 'dylib' if sys.platform == 'darwin' else 'so'
        self.library = ROOT / f'.build/native-v1.5/libothello.{suffix}'

    def test_batchnorm_folding_residual_skip_and_color_views(self):
        board = np.random.default_rng(1532).integers(-1, 2, (32, 64), dtype=np.int8)
        turn = np.array([1, -1] * 16, dtype=np.float32)
        for blocks in (2, 4):
            net = trainer.ResValueNet(blocks)
            # Non-default running statistics are essential to check BN folding.
            for _ in range(4):
                net(torch.from_numpy(board), torch.from_numpy(turn))
            prediction = trainer.predict(net, board, turn)
            np.testing.assert_array_equal(prediction, trainer.predict(net, -board, -turn))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.otres'
                binary = trainer.export(net, path)
                self.assertEqual(len(binary), 28 + 4 * (4866 + 18496 * blocks))
                self.assertLessEqual(trainer.parity(self.library, path, board, turn, prediction), 0.51)
            # A zero residual branch must preserve the input through its skip.
            net.eval()
            for parameter in net.tower.parameters():
                parameter.data.zero_()
            with torch.no_grad():
                x = torch.relu(net.stem(torch.stack((torch.from_numpy(board) == 1,
                                                      torch.from_numpy(board) == -1), dim=1).float().reshape(-1, 2, 8, 8)))
                torch.testing.assert_close(net.tower(x), x, rtol=0, atol=0)
            prediction = trainer.predict(net, board, turn)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'skip.otres'
                trainer.export(net, path)
                self.assertLessEqual(trainer.parity(self.library, path, board, turn, prediction), 0.51)

    def test_reject_malformed_headers_and_weights(self):
        net = trainer.ResValueNet().eval()
        lib = ctypes.CDLL(str(self.library))
        lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
        lib.ot_nnue_load.restype = ctypes.c_void_p
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.otres'
            binary = trainer.export(net, path)
            malformed = [binary[:-1], binary + b'extra', binary[:16]]
            for offset, values in [(8, (7,)), (12, (3,)), (16, (8,)), (20, (0, 3, 5)), (24, (32,))]:
                for value in values:
                    bad = bytearray(binary)
                    bad[offset:offset+4] = struct.pack('<I', value)
                    malformed.append(bytes(bad))
            for value in (float('nan'), float('inf'), 1001.0):
                bad = bytearray(binary)
                bad[28:32] = struct.pack('<f', value)
                malformed.append(bytes(bad))
            for bad in malformed:
                path.write_bytes(bad)
                self.assertFalse(lib.ot_nnue_load(str(path).encode()))


if __name__ == '__main__':
    unittest.main()
