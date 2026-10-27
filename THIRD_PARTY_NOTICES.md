# Third-party components and data

- **Egaroucid:** pinned upstream revision, GPL-3.0-or-later license and resource
  hashes are recorded in [third_party/egaroucid/](third_party/egaroucid/README.md).
  The book snapshot preserves the upstream license and source record. Optional
  native builds link Egaroucid; the builtin-only engine and browser build do not.
- **Stockfish:** generic search/NNUE methodology is a reference; the Othello
  implementation and feature layout are independent. See the exact referenced
  revision and adaptation scope in [docs/STOCKFISH_FRAMEWORK.md](docs/STOCKFISH_FRAMEWORK.md).
- **tiny_nn.h:** reused unmodified from the user-provided Patchwork project;
  provenance and hash are in [cpp/model/README.md](cpp/model/README.md). The
  supplied header has no separate license declaration; it is not relabeled here.
- **WTHOR / Logistello:** source pages, hashes and usage statements accompany
  [the data snapshots](data/README.md). This repository does not grant these
  external datasets a new permissive license.
- **Frontend packages:** React, Vite, TypeScript and Lucide are installed through
  npm; exact versions and resolved sources are pinned in `package-lock.json`.
  Package licenses remain with their respective authors.
