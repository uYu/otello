import importlib.util
from pathlib import Path
import sys
import json
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('search_matrix',ROOT/'scripts/compare-search-matrix.py')
matrix=importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class SearchMatrixTests(unittest.TestCase):
    def test_both_models_and_searches_through_native_abi(self):
        suffix='dylib' if sys.platform=='darwin' else 'so'
        library=ROOT/f'.build/native-v1.6/libothello.{suffix}'
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'matrix.json'
            subprocess.run([sys.executable,str(ROOT/'scripts/compare-search-matrix.py'),
                            '--library',str(library),'--milliseconds','5','--games-per-pair','2',
                            '--output',str(report)],check=True,capture_output=True,timeout=30)
            result=json.loads(report.read_text())
            self.assertTrue(result['complete'])
            self.assertEqual(result['games'],12)
            for game in result['records']:matrix.replay(game)
            for name in matrix.BY_ID:self.assertEqual(result['results'][name]['games'],6)

    def test_score_table_has_symmetric_pair_results(self):
        records=[dict(black='nnue_ab',white='cnn_mcts',winner='nnue_ab',blackDiscDifference=12,decisions=[]),
                 dict(black='cnn_mcts',white='nnue_ab',winner=None,blackDiscDifference=0,decisions=[])]
        totals,pairs=matrix.aggregate(records)
        self.assertEqual(totals['nnue_ab']['scoreRate'],.75)
        self.assertEqual(totals['cnn_mcts']['scoreRate'],.25)
        self.assertEqual(totals['nnue_ab']['discDifference'],12)
        self.assertEqual(totals['cnn_mcts']['discDifference'],-12)
        self.assertEqual(pairs['nnue_ab']['cnn_mcts'],dict(win=1,draw=1,loss=0))
        self.assertEqual(pairs['cnn_mcts']['nnue_ab'],dict(win=0,draw=1,loss=1))


if __name__=='__main__':unittest.main()
