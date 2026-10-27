# Otello workspace guide

## Restore the actual data first

The repository contains two versioned `.tar.gz` snapshots, with per-archive and
per-file SHA-256 checksums in `data/manifest.json`. They contain real data, not
Git LFS pointers or download placeholders. Use Python 3; no training packages
or network access are required to restore them.

```sh
python3 scripts/data-assets.py restore
# Or verify without replacing local data:
python3 scripts/data-assets.py verify
```

- `training-v1` restores only raw WTHOR/Logistello files and source checksums to
  `.build/training/raw/`. Generated JSONL, decompressed text and caches are excluded.
- `egaroucid-book` restores the exact v1 opening book, license and source record
  to `.build/data/egaroucid/`. The native setup script installs this verified book
  into the Egaroucid resource directory.
- `--only training-v1` / `--only egaroucid-book` selects one snapshot.
- Restore skips identical files and refuses conflicting files. Use `--force`
  only when intentionally replacing local data with the committed snapshot.
- Preserve source attribution and usage notes from `data/README.md` and the
  manifests. These datasets do not acquire a new permissive license here.

## Build and check

```sh
cmake -S . -B .build/native -DCMAKE_BUILD_TYPE=Release
cmake --build .build/native -j 2
ctest --test-dir .build/native --output-on-failure
npm ci
npm test
npm run build
```

The rules, hashing, search and NNUE inference are C++; the web app uses a Wasm
adapter. Native CLI/C ABI usage does not depend on WebAssembly or Node.

## Train and evaluate

Install `requirements-training.txt` for rebuilding caches, training and training
tests. After restoring the raw sources and building the native core, regenerate
derived files without training a model; the command checks all three v1 hashes.

```sh
npm run training:rebuild
npm run training:run
npm run setup:egaroucid
npm run training:baseline
# Separate terminal:
python3 scripts/watch-matches.py
```

On Linux, pass `--library .build/native/libothello.so` to the Python trainer or
comparison CLI; their original v1 defaults use the macOS `.dylib` path. The
baseline runner chooses its library suffix for the current platform.

`models/v1-baseline.json` records measured v1 weights, source hashes and results.
Keep the measured C++ sources and model artifacts consistent with those hashes;
create a new model/baseline version when intentionally changing the algorithm.
The book-assisted experiment uses book lookup followed by the same NNUE search
on a miss, never an Egaroucid search fallback.

Keep `node_modules/`, `.build/`, `dist/`, logs and `*.live.json` out of Git. Repack
data with `python3 scripts/data-assets.py pack` only for an intentional snapshot
update; routine builds and tests do not need to regenerate the archives. Data
archives contain original sources only; rebuild all derived JSONL/caches locally.
