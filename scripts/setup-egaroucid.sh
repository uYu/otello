#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python3 scripts/data-assets.py restore --only egaroucid-book
revision=2a65c87860be999373088ad833aa00b0844883b6
source_dir="$PWD/.build/deps/Egaroucid"
mkdir -p .build/deps
if [ ! -d "$source_dir/.git" ]; then
  git init "$source_dir"
  git -C "$source_dir" remote add origin https://github.com/Nyanyan/Egaroucid.git
fi
if ! git -C "$source_dir" cat-file -e "$revision^{commit}" 2>/dev/null; then
  git -C "$source_dir" fetch --depth 1 origin "$revision"
fi
current_revision=$(git -C "$source_dir" rev-parse HEAD 2>/dev/null || true)
if [ "$current_revision" != "$revision" ] || [ ! -f "$source_dir/include/egaroucid/egaroucid.h" ]; then
  git -C "$source_dir" checkout --detach "$revision"
fi
cmp .build/data/egaroucid/book.egbk3 "$source_dir/bin/resources/book.egbk3"
cp .build/data/egaroucid/book.egbk3 "$source_dir/bin/resources/book.egbk3"
cmake -S . -B .build/native-egaroucid -DCMAKE_BUILD_TYPE=Release -DOTELLO_WITH_EGAROUCID=ON
cmake --build .build/native-egaroucid -j 2
ctest --test-dir .build/native-egaroucid --output-on-failure
