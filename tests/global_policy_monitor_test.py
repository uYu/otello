import importlib.util
from pathlib import Path
import tempfile
import json
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('global_policy_watch',ROOT/'scripts/watch-matches.py')
watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)


class GlobalMonitorTests(unittest.TestCase):
    def test_displays_policy_selection_and_previous_completed_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'progress.json'
            path.write_text(json.dumps(dict(experiment='global-policy-training',stage='训练中',epoch=3,epochs=15,seconds=20,
                validation=dict(epoch=3,policyCrossEntropy=1.58,policyTop1=.42),
                bestValidation=dict(epoch=2,policyCrossEntropy=1.57,policyTop1=.43),
                baselineValidation=dict(policyCrossEntropy=1.59,policyTop1=.41))))
            text=watch.display(dict(experiments=[dict(name='策略训练',report=str(path))]))
            self.assertIn('最近完成第 3 轮',text);self.assertIn('当前最佳：第 2 轮',text)
            self.assertIn('1.580000',text);self.assertIn('1.570000',text);self.assertIn('42.00%',text)
            self.assertIn('价值分支冻结',text)

    def test_full_training_reports_value_without_claiming_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'progress.json'
            path.write_text(json.dumps(dict(experiment='global-policy-training',stage='全网训练中',epoch=2,epochs=15,seconds=20,freezeBaseline=False,
                validation=dict(epoch=2,policyCrossEntropy=1.58,policyTop1=.42,mse=.73),
                baselineValidation=dict(policyCrossEntropy=1.59,policyTop1=.41,mse=.72))))
            text=watch.display(dict(experiments=[dict(name='全网训练',report=str(path))]))
            self.assertIn('验证胜负 MSE 0.730000',text);self.assertIn('全网从头训练',text)
            self.assertNotIn('价值分支冻结',text)


if __name__=='__main__':unittest.main()
