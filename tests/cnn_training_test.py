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
spec = importlib.util.spec_from_file_location('cnn_trainer', ROOT / 'scripts/train-cnn.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class CnnTests(unittest.TestCase):
    def test_shape_color_relabeling_and_cpp_export(self):
        torch.manual_seed(54)
        net = trainer.CnnValueNet().eval()
        self.assertEqual(sum(p.numel() for p in net.parameters()), 17185)
        board = np.random.default_rng(54).integers(-1, 2, size=(32, 64), dtype=np.int8)
        turn = np.array([1.0, -1.0] * 16, dtype=np.float32)
        prediction = trainer.predict(net, board, turn)
        np.testing.assert_array_equal(prediction, trainer.predict(net, -board, -turn))
        suffix = 'dylib' if sys.platform == 'darwin' else 'so'
        library = ROOT / f'.build/native-v1.4/libothello.{suffix}'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.otcnn'
            binary = trainer.export(net, path)
            self.assertEqual(len(binary), 68764)
            self.assertLessEqual(trainer.parity(library, path, board, turn, prediction), 0.501)

    def test_malformed_models_are_rejected(self):
        net = trainer.CnnValueNet().eval()
        suffix = 'dylib' if sys.platform == 'darwin' else 'so'
        lib = ctypes.CDLL(str(ROOT / f'.build/native-v1.4/libothello.{suffix}'))
        lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
        lib.ot_nnue_load.restype = ctypes.c_void_p
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.otcnn'
            binary = trainer.export(net, path)
            bad_channels = bytearray(binary)
            bad_channels[16:20] = struct.pack('<I', 9)
            nan = bytearray(binary)
            nan[24:28] = struct.pack('<f', float('nan'))
            for malformed in (binary[:-1], binary + b'extra', bytes(bad_channels), bytes(nan)):
                path.write_bytes(malformed)
                self.assertFalse(lib.ot_nnue_load(str(path).encode()))


if __name__ == '__main__':
    unittest.main()
