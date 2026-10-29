#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if ! command -v em++ >/dev/null 2>&1; then
  if [ -f .build/emsdk/emsdk_env.sh ]; then . .build/emsdk/emsdk_env.sh; else
    echo "Emscripten is required only to rebuild the browser adapter; native builds use CMake." >&2
    exit 1
  fi
fi
export EM_CACHE="${EM_CACHE:-$PWD/.build/em-cache}"
python3 scripts/embed-default-model.py --check
em++ cpp/search.cpp cpp/move_policy.cpp cpp/nnue.cpp cpp/pattern.cpp cpp/cnn.cpp cpp/resnet.cpp cpp/mcts.cpp cpp/default_engine.cpp cpp/api.cpp -std=c++17 -O3 -msimd128 -sSTACK_SIZE=1048576 -fexceptions -sMODULARIZE=1 -sEXPORT_ES6=1 -sSINGLE_FILE=1 -sENVIRONMENT=web,worker,node -sALLOW_MEMORY_GROWTH=1 -sFILESYSTEM=0 -sEXPORTED_RUNTIME_METHODS=HEAP32,HEAPU8 -sEXPORTED_FUNCTIONS='["_ot_input","_ot_output","_ot_legal","_ot_play","_ot_search","_ot_search_clear_cache","_ot_default_search","_ot_default_evaluate","_ot_default_model_sha256","_ot_mcts_clear_cache"]' -o src/game/wasm/othello.mjs
