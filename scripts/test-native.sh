#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p .build
if [ "$(uname -s)" = Darwin ]; then
  set -- -isystem "$(xcrun --show-sdk-path)/usr/include/c++/v1"
else set --; fi
${CXX:-clang++} "$@" -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer tests/native.cpp cpp/search.cpp cpp/nnue.cpp cpp/api.cpp -o .build/test-native
.build/test-native
