#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export EM_CACHE="${EM_CACHE:-$PWD/.build/em-cache}"
python3 scripts/embed-default-model.py --check
em++ cpp/pattern.cpp cpp/direct_policy.cpp -DOTELLO_PATTERN_SIMD=1 -DOTELLO_PATTERN_FAST_CODES=1 -DOTELLO_PATTERN_BLOCK4=1 -DOTELLO_PATTERN_LEGAL_POLICY=1 -std=c++17 -O3 -msimd128 -sSTACK_SIZE=1048576 -fexceptions -sMODULARIZE=1 -sEXPORT_ES6=1 -sSINGLE_FILE=1 -sENVIRONMENT=web,worker,node -sALLOW_MEMORY_GROWTH=1 -sFILESYSTEM=0 -sEXPORTED_RUNTIME_METHODS=HEAP32,HEAPU8,HEAPF64 -sEXPORTED_FUNCTIONS='["_ot_policy_input","_ot_policy_output","_ot_policy_model_sha256","_ot_policy_infer"]' -o src/game/wasm/direct-policy.mjs
