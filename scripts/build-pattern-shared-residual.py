#!/usr/bin/env python3
"""Build v1.48 in an isolated source tree; never rewrite default C++/Wasm."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/pattern-shared-residual-v1.48'
SOURCE=WORK/'source'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replace(source,old,new):
    assert source.count(old)==1,old
    return source.replace(old,new)


def main():
    if not SOURCE.exists():
        SOURCE.mkdir(parents=True)
        for name in ['cpp','tests','benchmarks']:shutil.copytree(ROOT/name,SOURCE/name)
        shutil.copy2(ROOT/'CMakeLists.txt',SOURCE/'CMakeLists.txt')
        (WORK/'original-source-sha256.json').write_text(json.dumps({str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file()},indent=2)+'\n')
        h=SOURCE/'cpp/pattern.hpp';text=h.read_text()
        text=replace(text,'    static const std::array<Shape,Count>& shapes();','''    // OTPAT8: once-per-position shared residual trunk.
    int residualWidth=16;
    std::array<std::int16_t,16*32> residualDown{},residualUp{};
    std::array<std::int32_t,16> residualDownBias{};
    std::array<std::int32_t,32> residualUpBias{};
    static const std::array<Shape,Count>& shapes();''')
        text=replace(text,'bool phasePolicy=false);','bool phasePolicy=false,bool sharedResidual=false);');h.write_text(text)
        p=SOURCE/'cpp/pattern.cpp';text=p.read_text();text='#include "pattern-shared-residual.hpp"\n'+text
        text=replace(text,'bool phaseValue,bool phasePolicy){','bool phaseValue,bool phasePolicy,bool sharedResidual){')
        text=replace(text,'    if(dim!=Dim||','    const auto residualWidth=sharedResidual?read(in,4):16;\n    if((sharedResidual&&(!structural||variant!=1||(residualWidth!=8&&residualWidth!=16)))||dim!=Dim||')
        text=replace(text,'    if(variant==1||variant==3){','    if(sharedResidual){model->variant=6;model->residualWidth=int(residualWidth);}\n    if(variant==1||variant==3){')
        text=replace(text,'    if(in.get()!=EOF)throw std::runtime_error("Trailing bytes in pattern model");return model;',
            '''    if(sharedResidual){i16(model->residualDown,model->residualWidth*32);i32(model->residualDownBias,model->residualWidth);i16(model->residualUp,32*model->residualWidth);i32(model->residualUpBias,32);}
    if(in.get()!=EOF)throw std::runtime_error("Trailing bytes in pattern model");return model;''')
        text=replace(text,'        row=0;const int headCount=variant==3?4:Outputs;',
            '''        if(variant==6)applySharedResidual(residualDown.data(),residualDownBias.data(),residualUp.data(),residualUpBias.data(),residualWidth,h.data());
        row=0;const int headCount=variant==3?4:Outputs;''')
        text=replace(text,'bool Pattern::operator==(const Pattern& b) const{return ',
            'bool Pattern::operator==(const Pattern& b) const{return residualWidth==b.residualWidth&&residualDown==b.residualDown&&residualDownBias==b.residualDownBias&&residualUp==b.residualUp&&residualUpBias==b.residualUpBias&&')
        p.write_text(text);shutil.copy2(ROOT/'benchmarks/pattern-shared-residual.hpp',SOURCE/'cpp/pattern-shared-residual.hpp')
        p=SOURCE/'cpp/nnue.cpp';text=p.read_text()
        text=replace(text,'format==std::string("OTPAT7\\0\\0",8))){','format==std::string("OTPAT7\\0\\0",8)||format==std::string("OTPAT8\\0\\0",8))){')
        text=replace(text,'format==std::string("OTPAT7\\0\\0",8),format==std::string("OTPAT5\\0\\0",8)',
            'format==std::string("OTPAT7\\0\\0",8)||format==std::string("OTPAT8\\0\\0",8),format==std::string("OTPAT5\\0\\0",8)')
        text=replace(text,'format==std::string("OTPAT7\\0\\0",8));','format==std::string("OTPAT7\\0\\0",8),format==std::string("OTPAT8\\0\\0",8));');p.write_text(text)
        p=SOURCE/'cpp/main.cpp';text=p.read_text();text=replace(text,'    if(network&&network->pattern&&network->pattern->variant==3)',
            '    if(network&&network->pattern&&network->pattern->variant==6)return "othello-pattern32-shared-residual-outcome-v8";\n    if(network&&network->pattern&&network->pattern->variant==3)');p.write_text(text)
        p=SOURCE/'CMakeLists.txt';p.write_text(p.read_text()+'''\nadd_executable(otello_shared_residual_benchmark benchmarks/mlp-inference-study.cpp)
target_link_libraries(otello_shared_residual_benchmark PRIVATE othello)
''')
    portable='--portable' in sys.argv
    build=ROOT/('.build/native-pattern-shared-residual-portable-v1.48' if portable else '.build/native-pattern-shared-residual-v1.48')
    subprocess.run(['cmake','-S',str(SOURCE),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DOTELLO_PATTERN_SIMD='+('OFF' if portable else 'ON')],check=True)
    subprocess.run(['cmake','--build',str(build),'-j','2'],check=True)
    subprocess.run(['ctest','--test-dir',str(build),'--output-on-failure'],check=True)


if __name__=='__main__':main()
