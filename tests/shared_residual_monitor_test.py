import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import shared_residual_experiment as e
spec=importlib.util.spec_from_file_location('shared_residual_watch',ROOT/'scripts/watch-matches.py')
watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)


class MonitorTests(unittest.TestCase):
    def test_next_epoch_retains_last_and_best_metrics(self):
        old=(e.WORK,e.PROTOCOL)
        try:
            with tempfile.TemporaryDirectory() as folder:
                e.WORK=Path(folder);e.PROTOCOL=e.WORK/'protocol.json';e.write(e.PROTOCOL,dict(startedUnix=time.time()))
                score=dict(mse=.72,policyTop1=.41,policyCrossEntropy=1.59,wdlCrossEntropy=.9)
                e.progress('训练中',1,15,validation=score)
                e.progress('训练中',1,15,'下一轮开始')
                latest=e.read(e.WORK/'progress.json')
                self.assertEqual(latest['validation']['mse'],.72)
                self.assertEqual(latest['bestValidation']['epoch'],1)
                e.progress('训练中',2,15,validation=dict(score,mse=.73))
                latest=e.read(e.WORK/'progress.json');self.assertEqual(latest['validationEpoch'],2)
                self.assertEqual(latest['bestValidation']['epoch'],1)
                config=dict(experiments=[dict(name='测试训练',report=str(e.WORK/'progress.json'))])
                text=watch.display(config)
                self.assertIn('0.730000',text);self.assertIn('41.00%',text);self.assertIn('1.590000',text)
                self.assertIn('当前最佳：第 1 轮',text)
        finally:e.WORK,e.PROTOCOL=old


if __name__=='__main__':unittest.main()
