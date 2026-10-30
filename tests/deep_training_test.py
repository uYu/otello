"""Shared residual trunk identity, inference parity and tiny training integration."""
import ctypes,hashlib,importlib.util,json,struct,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('deep_trainer',ROOT/'scripts/train-deep.py');trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
PARENT=ROOT/'models/nnue256-outcome-v1.9.otnnue';LIBRARY=ROOT/'.build/native-deep/libothello.dylib'
class DeepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.lib=ctypes.CDLL(str(LIBRARY));ptr=ctypes.POINTER(ctypes.c_int)
        cls.lib.ot_nnue_load.argtypes=[ctypes.c_char_p];cls.lib.ot_nnue_load.restype=ctypes.c_void_p;cls.lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p]
        cls.lib.ot_nnue_evaluate.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ptr];cls.lib.ot_nnue_policy.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
    def test_identity_and_nonzero_trunk_parity_and_malformed(self):
        parent=PARENT.read_bytes();torch.manual_seed(1010);net=trainer.DeepNet(parent)
        self.assertEqual(sum(p.numel() for p in net.parameters()),68770)
        self.assertEqual(sum(p.numel() for p in net.parameters() if p.requires_grad),17185)
        with np.load(ROOT/'.build/training/v1/positions.npz') as data:
            boards=data['test_board'][:32];turns=data['test_turn'][:32]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'deep.otnnue';old=self.lib.ot_nnue_load(str(PARENT).encode())
            try:
                for identity in (True,False):
                    if not identity:
                        with torch.no_grad():net.up.weight.uniform_(-.04,.04);net.up.bias.uniform_(-.02,.02)
                    binary=trainer.export(net,parent);self.assertEqual(len(binary),138864)
                    self.assertEqual(binary[8:trainer.outcome.VALUE_OFFSET],parent[8:trainer.outcome.VALUE_OFFSET])
                    self.assertEqual(binary[67612:104936],parent[67612:])
                    path.write_bytes(binary);handle=self.lib.ot_nnue_load(str(path).encode());self.assertTrue(handle)
                    qnet=trainer.quantized_net(binary)
                    with torch.no_grad():
                        values,logits=qnet(torch.from_numpy(boards),torch.from_numpy(turns))
                        renamed=qnet(torch.from_numpy(-boards),torch.from_numpy(-turns))
                        torch.testing.assert_close(values,renamed[0],rtol=0,atol=0);torch.testing.assert_close(logits,renamed[1],rtol=0,atol=0)
                    try:
                        for i,(board,turn) in enumerate(zip(boards,turns)):
                            cells=(ctypes.c_int*64)(*map(int,board));score=ctypes.c_int();before=ctypes.c_int();a,b=(ctypes.c_double*65)(),(ctypes.c_double*65)()
                            self.assertEqual(self.lib.ot_nnue_evaluate(handle,cells,int(turn),ctypes.byref(score)),1)
                            self.assertLessEqual(abs(score.value-values[i].item()*1000),.501)
                            self.assertEqual(self.lib.ot_nnue_policy(handle,cells,int(turn),a),1)
                            if identity:
                                self.lib.ot_nnue_evaluate(old,cells,int(turn),ctypes.byref(before));self.lib.ot_nnue_policy(old,cells,int(turn),b)
                                self.assertEqual(score.value,before.value);self.assertLess(np.max(np.abs(np.array(a)-np.array(b))),1e-12)
                            else:
                                probs=logits[i].numpy();legal=np.array(a)>0;probs=np.where(legal,probs,-np.inf);probs=np.exp(probs-probs.max());probs/=probs.sum()
                                self.assertLess(np.max(np.abs(np.array(a)-probs)),1e-10)
                    finally:self.lib.ot_nnue_destroy(handle)
                malformed=bytearray(binary);struct.pack_into('<I',malformed,104936,128)
                for bad in (binary[:-1],binary+b'x',malformed):
                    path.write_bytes(bad);self.assertFalse(self.lib.ot_nnue_load(str(path).encode()))
            finally:self.lib.ot_nnue_destroy(old)
    def test_tiny_joint_training(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);cache=folder/'positions.npz';labels=folder/'labels.npz';output=folder/'deep.otnnue'
            with np.load(ROOT/'.build/training/v1/positions.npz') as original,np.load(ROOT/'.build/training/policy/labels.npz') as original_labels:
                data={s+'_'+k:original[s+'_'+k][:n] for s,n in [('train',64),('validation',16),('test',16)] for k in ('board','turn','target')}
                lab={s+'_'+k:original_labels[s+'_'+k][:n] for s,n in [('train',64),('validation',16),('test',16)] for k in ('move','legal')}
            np.savez_compressed(cache,**data);np.savez_compressed(labels,**lab)
            cache.with_suffix('.json').write_text(json.dumps(dict(seed=20261027,cacheSha256=trainer.sha(cache))))
            labels.with_suffix('.json').write_text(json.dumps(dict(positionsSha256=trainer.sha(cache),cacheSha256=trainer.sha(labels))))
            subprocess.run([sys.executable,str(ROOT/'scripts/train-deep.py'),'--cache',str(cache),'--labels',str(labels),'--output',str(output),'--epochs','1','--batch-size','16'],check=True,capture_output=True,text=True,timeout=30)
            report=json.loads(output.with_suffix('.json').read_text());self.assertEqual(report['cppParityPositions'],16);self.assertTrue(report['featuresUnchanged'] and report['policyBytesUnchanged'])
if __name__=='__main__':unittest.main()
