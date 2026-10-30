"""Outcome format, frozen policy, value parity and tiny end-to-end training."""
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
LIBRARY=ROOT/('.build/native-outcome/libothello.'+('dylib' if sys.platform=='darwin' else 'so'))
spec=importlib.util.spec_from_file_location('outcome_trainer',ROOT/'scripts/train-outcome.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
PARENT=ROOT/'models/nnue256-policy-v1.8.otnnue'


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib=ctypes.CDLL(str(LIBRARY));ptr=ctypes.POINTER(ctypes.c_int)
        cls.lib.ot_nnue_load.argtypes=[ctypes.c_char_p];cls.lib.ot_nnue_load.restype=ctypes.c_void_p
        cls.lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p]
        cls.lib.ot_nnue_evaluate.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ptr]
        cls.lib.ot_nnue_policy.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
        cls.lib.ot_nnue_mcts_play.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*3+[ptr,ptr,ptr,ctypes.POINTER(ctypes.c_double)]

    def test_export_value_parity_and_color_rename(self):
        parent=PARENT.read_bytes();torch.manual_seed(1919);net=trainer.OutcomeNet(parent)
        self.assertEqual(sum(p.numel() for p in net.parameters() if p.requires_grad),513)
        with torch.no_grad():net.head.bias.fill_(0.41)
        binary=trainer.export(net,parent)
        self.assertEqual(binary[:8],b'OTNNUE4\0')
        self.assertEqual(binary[8:trainer.VALUE_OFFSET],parent[8:trainer.VALUE_OFFSET])
        self.assertEqual(binary[trainer.VALUE_END:],parent[trainer.VALUE_END:])
        boards=np.random.default_rng(91).integers(-1,2,size=(16,64),dtype=np.int8)
        turns=np.array([1,-1]*8,dtype=np.float32)
        with torch.no_grad():
            a=net(torch.from_numpy(boards),torch.from_numpy(turns))
            b=net(torch.from_numpy(-boards),torch.from_numpy(-turns))
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'outcome.otnnue';path.write_bytes(binary)
            handle=self.lib.ot_nnue_load(str(path).encode());old=self.lib.ot_nnue_load(str(PARENT).encode())
            self.assertTrue(handle and old)
            expected=trainer.quantized_predict(binary,boards,turns)*1000
            try:
                for board,turn,want in zip(boards,turns,expected):
                    cells=(ctypes.c_int*64)(*map(int,board));score=ctypes.c_int()
                    self.assertEqual(self.lib.ot_nnue_evaluate(handle,cells,int(turn),ctypes.byref(score)),1)
                    self.assertLessEqual(abs(score.value-want),0.501)
                    a,b=(ctypes.c_double*65)(),(ctypes.c_double*65)()
                    self.assertEqual(self.lib.ot_nnue_policy(handle,cells,int(turn),a),1)
                    self.assertEqual(self.lib.ot_nnue_policy(old,cells,int(turn),b),1)
                    self.assertEqual(list(a),list(b))
                # Read the format's semantics through terminal MCTS, not by name.
                for board,turn,want in [([1]*64,-1,-100000),([1]*32+[-1]*32,1,0),([1]*64,1,100000)]:
                    self.lib.ot_mcts_clear_cache();out=(ctypes.c_int*136)();p=(ctypes.c_int*6)();visits=(ctypes.c_int*64)();stats=(ctypes.c_double*72)()
                    self.assertEqual(self.lib.ot_nnue_mcts_play(handle,(ctypes.c_int*64)(*board),turn,0,1,19,0,32,64,out,p,visits,stats),1)
                    self.assertEqual(out[1],want);self.assertTrue(out[5])
            finally:
                self.lib.ot_nnue_destroy(handle);self.lib.ot_nnue_destroy(old)
            invalid_dimensions=bytearray(binary);struct.pack_into('<I',invalid_dimensions,trainer.VALUE_END,64)
            for invalid in (binary[:-1],binary+b'x',invalid_dimensions,b'OTNNUE5\0'+binary[8:]):
                path.write_bytes(invalid)
                self.assertFalse(self.lib.ot_nnue_load(str(path).encode()))

    def test_tiny_training_preserves_policy_and_exports_outcome(self):
        with np.load(ROOT/'.build/training/v1/positions.npz') as original:
            data={s+'_'+k:original[s+'_'+k][:n] for s,n in [('train',64),('validation',16),('test',16)] for k in ('board','turn','target')}
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);cache=folder/'positions.npz';output=folder/'outcome.otnnue'
            np.savez_compressed(cache,**data)
            cache.with_suffix('.json').write_text(json.dumps(dict(seed=20261027,cacheSha256=hashlib.sha256(cache.read_bytes()).hexdigest())))
            subprocess.run([sys.executable,str(ROOT/'scripts/train-outcome.py'),'--cache',str(cache),
                            '--parent',str(PARENT),'--baseline-manifest',str(ROOT/'models/v1.8-default.json'),
                            '--output',str(output),'--library',str(LIBRARY),'--epochs','1','--batch-size','16'],
                           check=True,capture_output=True,text=True,timeout=30)
            report=json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(report['cppParityPositions'],16)
            self.assertEqual(report['cppPolicyMaxError'],0)
            self.assertEqual(report['modelFormat'],'OTNNUE4')
            self.assertTrue(report['featuresUnchanged'] and report['policyBytesUnchanged'])

    def test_exact_outcome_audit_requires_proof_not_rounded_estimate(self):
        spec=importlib.util.spec_from_file_location('outcome_comparison',ROOT/'scripts/compare-nnue-candidate.py')
        comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
        board=comparison.match.rules.initial()
        decision=dict(valueTarget='outcome',exact=True,score=100000,move=19,
            candidates=[dict(move=19,score=100000,value=1,visits=2),dict(move=26,score=1000,value=1,visits=999),
                        dict(move=37,score=0,value=0,visits=3),dict(move=44,score=0,value=0,visits=3)],
            selection=dict(seed=19,eligible=1,temperatureMilli=0,lossLimit=0,bestScore=100000,scoreLoss=0,sampled=False))
        comparison.audit_decision(board,1,decision)
        decision['move']=26
        with self.assertRaises(AssertionError):comparison.audit_decision(board,1,decision)


if __name__=='__main__':unittest.main()
