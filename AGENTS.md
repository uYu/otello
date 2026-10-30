# Otello workspace guide

## Active baseline and promotion policy

- Continue model/search optimization from user-approved v1.36: the Pattern
  policy/outcome model `models/pattern32-outcome-v1.25.otpat` (OTPAT2), v1.3
  opening variety, and learned-prior PUCT MCTS with finite FPU (parent mean
  minus 0.15). `models/default-model.json` selects the active model and
  `models/v1.36-default.json` records the frozen default baseline.
- Train and evaluate candidates against the latest user-approved default. Keep
  candidate artifacts/reports separate; better validation or match results alone
  do not authorize changing the default.
- When the user says to make a candidate the default (for example,
  “作为默认模型”), that authorizes promoting it and freezing it as the new
  baseline. Update default entry points, embedded weights when applicable,
  version metadata, source/model hashes, evaluation records, and documentation
  together. Preserve previous baselines and use the newly approved baseline for
  subsequent comparisons.
- User decision (2026-10-27): promote the policy-head model to the default and
  freeze it as the new v1.8 baseline. Preserve both measured reports: 1s/20 games
  yielded 11W/2D/7L; 100ms/100 games yielded 29W/2D/69L. Keep time budgets
  separate and do not rewrite candidate-era reports or claim no regression.
- User decision (2026-10-27): promote v1.9 outcome model and freeze it as the new
  baseline after 18W/2D/0L at 1s over 20 games against v1.8. Preserve the experiment
  as measured; no additional match or retraining is required for this promotion.
- User decision (2026-10-29): promote v1.25 Pattern outcome as the default and
  freeze it as the new baseline after 18W/0D/2L at 100ms/20 games and
  12W/5D/3L at 2s/20 games against frozen v1.9. Keep both time budgets and
  candidate-era reports separate; no new training or strength match is needed.
- The v1.9 and v1.25 scalar value target is side-to-move win/draw/loss (+1/0/-1), with
  matching MCTS terminal backups and proof stopping. The v1.9 features and learned
  policy bytes are inherited unchanged from v1.8; v1.25 uses its independently
  trained Pattern encoding and policy head. The previous v1.8 disc-value baseline
  remains frozen; do not mix its disc/64 terminal values with outcome values.
- Every future comparison must provide the user a live monitor. Keep one stable
  entry point, `sh scripts/watch-match.sh`, and point `models/current-experiment.json`
  at the current comparison. Show candidate-perspective W/D/L once, current
  board and final board; use ASCII `*` for black, `o` for white and `.` for empty.
  Update after each move, and retain the final board after completion.
- Web deployment defaults to the verified Egaroucid opening book, always enabled
  with no user-facing off switch. Query it before built-in AI on every move;
  a verified miss continues with the approved v1.25 Pattern search. Loading failure is an
  error/retry, not a silent bypass. Keep source hashes, GPL notices and original
  book data with the derived browser assets. Model-only evaluation protocols
  remain separate from this deployed book-assisted flow.

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


## User-approved v1.36 engineering baseline (2026-10-30)

The active baseline is `models/v1.36-default.json`, selected by
`models/default-model.json`. It retains the exact v1.25 INT16 Pattern outcome
weights and enables fused current-side inference, NEON/SSE2 with portable fallback,
packed shape encoding, four-row reuse and legal-only policy projection. The default
node budget is 4,194,304; incremental cache and integer rounding remain disabled.
The previous v1.25 baseline and candidate-era reports stay frozen.

Controlled 2s sampling measured 2.158x simulations/s and 1.868x evaluations/s.
The 4M capacity match against frozen v1.25 at 2s/20 games finished 7W/7D/6L
(52.5%, mean discs +2.4), with 0/603 candidate decisions hitting the node cap.
This is engineering parity, not evidence of statistically significant strength gain.
Candidate process peak RSS was 2493.9 MiB; provision memory accordingly on Linux.
Native kernels select AArch64 NEON or x86-64 SSE2; actual Linux throughput remains
unmeasured. Browser budgets stay 100ms/1s and the verified opening book stays enabled.
See `models/v1.36-pattern-capacity-comparison.md` and
`models/v1.35-pattern-speed-suite-comparison.md` for separate measured results.
