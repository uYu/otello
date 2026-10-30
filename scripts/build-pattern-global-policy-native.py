#!/usr/bin/env python3
"""Build OTPAT9 global-policy inference in an isolated native source tree."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/pattern-global-policy-native-v1.49';SOURCE=WORK/'source'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replace(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def main():
    if not SOURCE.exists():
        SOURCE.mkdir(parents=True)
        for name in ['cpp','tests','benchmarks']:shutil.copytree(ROOT/name,SOURCE/name)
        shutil.copy2(ROOT/'CMakeLists.txt',SOURCE/'CMakeLists.txt')
        (WORK/'original-source-sha256.json').write_text(json.dumps({str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file()},indent=2)+'\n')
        p=SOURCE/'cpp/pattern.hpp';text=p.read_text()
        text=replace(text,'    static const std::array<Shape,Count>& shapes();','''    // OTPAT9 global policy: fixed38->16->64; expanded D4-tied weights.
    std::array<std::int16_t,16*38> globalDown{};
    std::array<std::int32_t,16> globalDownBias{};
    std::array<std::int16_t,64*16> globalUp{};
    std::array<std::int32_t,64> globalUpBias{};
    static const std::array<Shape,Count>& shapes();''')
        text=replace(text,'bool phasePolicy=false);','bool phasePolicy=false,bool globalPolicy=false);');p.write_text(text)
        p=SOURCE/'cpp/pattern.cpp';text='#include "pattern-global-policy.hpp"\n'+p.read_text()
        text=replace(text,'bool phaseValue,bool phasePolicy){','bool phaseValue,bool phasePolicy,bool globalPolicy){')
        text=replace(text,'    if(dim!=Dim||','''    const auto globalInputs=globalPolicy?read(in,4):38,globalWidth=globalPolicy?read(in,4):16,globalOutputs=globalPolicy?read(in,4):64;
    if((globalPolicy&&(!structural||variant!=1||globalInputs!=38||globalWidth!=16||globalOutputs!=64))||dim!=Dim||''')
        text=replace(text,'    if(variant==1||variant==3){','    if(globalPolicy)model->variant=6;\n    if(variant==1||variant==3){')
        text=replace(text,'    if(in.get()!=EOF)throw std::runtime_error("Trailing bytes in pattern model");return model;',
            '''    if(globalPolicy){i16(model->globalDown,16*38);i32(model->globalDownBias,16);i16(model->globalUp,64*16);i32(model->globalUpBias,64);}
    if(in.get()!=EOF)throw std::runtime_error("Trailing bytes in pattern model");return model;''')
        needle='        if(targets){saved=entries.data()+active*Dim;destinations[active++]=targets;}'
        # Only structural inference retains every ordered row for the global head.
        begin=text.index('Pattern::Prediction Pattern::predictStructural(')
        prefix,suffix=text[:begin],text[begin:]
        suffix=replace(suffix,needle,'        if(targets||variant==6){saved=entries.data()+active*Dim;destinations[active++]=targets;}')
        suffix=replace(suffix,'    maskedPolicy(result,entries.data(),destinations.data(),active,mask);return result;',
            '''    if(variant==6){applyGlobalPolicy(result,*this,entries.data(),destinations.data(),mask,geometry.counts,geometry.buckets);return result;}
    maskedPolicy(result,entries.data(),destinations.data(),active,mask);return result;''')
        text=prefix+suffix
        text=replace(text,'bool Pattern::operator==(const Pattern& b) const{return ',
            'bool Pattern::operator==(const Pattern& b) const{return globalDown==b.globalDown&&globalDownBias==b.globalDownBias&&globalUp==b.globalUp&&globalUpBias==b.globalUpBias&&')
        p.write_text(text);shutil.copy2(ROOT/'benchmarks/pattern-global-policy.hpp',SOURCE/'cpp/pattern-global-policy.hpp')
        p=SOURCE/'cpp/nnue.cpp';text=p.read_text()
        text=replace(text,'format==std::string("OTPAT7\\0\\0",8))){','format==std::string("OTPAT7\\0\\0",8)||format==std::string("OTPAT9\\0\\0",8))){')
        text=replace(text,'format==std::string("OTPAT7\\0\\0",8),format==std::string("OTPAT5\\0\\0",8)',
            'format==std::string("OTPAT7\\0\\0",8)||format==std::string("OTPAT9\\0\\0",8),format==std::string("OTPAT5\\0\\0",8)')
        text=replace(text,'format==std::string("OTPAT7\\0\\0",8));','format==std::string("OTPAT7\\0\\0",8),format==std::string("OTPAT9\\0\\0",8));');p.write_text(text)
        p=SOURCE/'cpp/main.cpp';text=p.read_text();text=replace(text,'    if(network&&network->pattern&&network->pattern->variant==5)',
            '    if(network&&network->pattern&&network->pattern->variant==6)return "othello-pattern32-global-policy-outcome-v9";\n    if(network&&network->pattern&&network->pattern->variant==5)');p.write_text(text)
        p=SOURCE/'CMakeLists.txt';p.write_text(p.read_text()+'''\nadd_executable(otello_global_policy_benchmark benchmarks/mlp-inference-study.cpp)
target_link_libraries(otello_global_policy_benchmark PRIVATE othello)
''')
    shutil.copy2(ROOT/'tests/global_policy_native.cpp',SOURCE/'tests/global_policy_native.cpp')
    cmake=SOURCE/'CMakeLists.txt'
    if 'add_executable(otello_global_policy_tests' not in cmake.read_text():
        cmake.write_text(cmake.read_text()+f'''\nadd_executable(otello_global_policy_tests tests/global_policy_native.cpp)
target_link_libraries(otello_global_policy_tests PRIVATE othello)
add_test(NAME global_policy_native COMMAND otello_global_policy_tests "{ROOT/'models/pattern32-global-policy16-full-v1.49.otpat'}")
''')
    portable='--portable' in sys.argv
    build=ROOT/('.build/native-pattern-global-policy-portable-v1.49' if portable else '.build/native-pattern-global-policy-v1.49')
    subprocess.run(['cmake','-S',str(SOURCE),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DOTELLO_PATTERN_SIMD='+('OFF' if portable else 'ON')],check=True)
    subprocess.run(['cmake','--build',str(build),'-j','2'],check=True)
    subprocess.run(['ctest','--test-dir',str(build),'--output-on-failure'],check=True)


if __name__=='__main__':main()
