import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('versions', Path(__file__).resolve().parents[1] / 'scripts/compare-model-versions.py')
versions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(versions)


class VersionComparisonTests(unittest.TestCase):
    def test_opening_and_illegal_pass(self):
        board = versions.initial()
        self.assertEqual(versions.legal(board, 1), [19, 26, 37, 44])
        with self.assertRaises(ValueError):
            versions.play(board, 1, -1)
        after, turn = versions.play(board, 1, 19)
        self.assertEqual((after.count(1), after.count(-1), turn), (4, 1, -1))
        self.assertEqual(board.count(1), 2)

    def test_full_game_replay_and_corruption(self):
        board, turn, moves, decisions = versions.initial(), 1, [], []
        while True:
            legal = versions.legal(board, turn)
            if not legal and not versions.legal(board, -turn):
                break
            move = legal[0] if legal else -1
            if move >= 0:
                decisions.append(dict(turn=turn, move=move, engine='v1.2' if turn == 1 else 'v1.1'))
            moves.append(move)
            board, turn = versions.play(board, turn, move)
        difference = sum(board)
        record = dict(focusSide=1, moves=moves, decisions=decisions, finalBoard=board,
                      focusDiscDifference=difference,
                      outcome='win' if difference > 0 else 'loss' if difference < 0 else 'draw')
        self.assertEqual(versions.replay(record), len(moves))
        record['focusDiscDifference'] += 1
        with self.assertRaises(AssertionError):
            versions.replay(record)


if __name__ == '__main__':
    unittest.main()
