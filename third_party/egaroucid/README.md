# Egaroucid dependency

- Upstream: https://github.com/Nyanyan/Egaroucid
- Pinned revision: `2a65c87860be999373088ad833aa00b0844883b6`
- License: GPL-3.0-or-later; full text in [LICENSE](LICENSE).
- Source is fetched unmodified into `.build/deps/Egaroucid` by `scripts/setup-egaroucid.sh`.
- The opening book from that same revision is bundled in
  `data/opening-book/egaroucid-2a65c87860be.tar.gz`, with license and source record.
  `scripts/data-assets.py restore --only egaroucid-book` restores it; the setup
  script verifies and installs the identical book into the engine resources.
- Other evaluation resources are fetched from the same revision.
- Optional native builds link this dependency; ordinary browser/Wasm and builtin-only builds do not.
- `cpp/egaroucid_bridge.cpp` compiles the unmodified upstream C API source and adds
  a GPL-3.0-or-later book-only entry point in the same translation unit. It calls
  the upstream weighted Book lookup directly, with no Egaroucid search fallback.
  The fetched dependency files remain unmodified.

SHA-256 of upstream resources:

- `book.egbk3`: `0dd454d84aa656a64329930bad796c4a48a4322b7693d50b7040006383db5012`
- `eval.egev2`: `0a8acbcbc23067933627f1bcc83c3f7ae6b7ff3e603d174a3364271fcaeeef2f`
- `eval_move_ordering_end.egev`: `bd908885e8bb85e72a9be2b0399f8ca52ed40a138bcaaecca3d13710945b9a93`
- `hash/hash25.eghs`: `66f2a6828037b27aff883ed45ce3b499aa653727789d1d2ef65f1bd3f42fda3c`
