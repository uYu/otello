#!/usr/bin/env python3
"""Validation-only phase breakdown for a staged-head hypothesis; no model changes."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from pattern_model import legal_mask
spec=importlib.util.spec_from_file_location('phase_profile_native_helpers',ROOT/'scripts/benchmark-pattern-int8.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    output=ROOT/'models/v1.45-phase-validation-profile.json';assert not output.exists()
    data=dict(np.load(ROOT/'.build/training/v1/positions.npz'));labels=dict(np.load(ROOT/'.build/training/v1.8/policy.npz'));placed=np.count_nonzero(data['validation_board'],axis=1)-4;train_placed=np.count_nonzero(data['train_board'],axis=1)-4;mask=legal_mask(data['validation_board'],data['validation_turn']);rows=[]
    library=ROOT/'.build/native-v1.37-reference/libothello.dylib'
    for name,path in [('hard-baseline','models/pattern32-symmetry-v1.37.otpat'),('mixed-soft','models/pattern32-symmetry-mixed-soft-v1.45.otpat')]:
        lib,handle=h.load(library,ROOT/path)
        try:prediction=h.native(lib,handle,data['validation_board'],data['validation_turn'])
        finally:lib.ot_nnue_destroy(handle)
        phases=[]
        for lo,hi in [(9,18),(21,36),(39,54)]:
            selected=(placed>=lo)&(placed<=hi);training=(train_placed>=lo)&(train_placed<=hi);target=np.sign(data['validation_target'][selected])
            phases.append(dict(placements=[lo,hi],validationPositions=int(selected.sum()),trainPositions=int(training.sum()),zeroValueMse=float(np.mean(target**2)),**h.metrics(prediction[selected],data['validation_target'][selected],labels['validation'][selected],mask[selected])))
        rows.append(dict(name=name,model=path,sha256=sha(ROOT/path),phases=phases))
    report=dict(status='validation-only-diagnostic',results=rows,validationUnchanged=True,testUsed=False,newModelTrained=False,sourceSha256={'scripts/profile-phase-validation.py':sha(__file__)},limits='Stage difficulty and label uncertainty differ naturally; high early-game error alone is not evidence that separate phase heads help. This profile is descriptive, does not change validation labels or pick a head design.')
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(rows,ensure_ascii=False))


if __name__=='__main__':main()
