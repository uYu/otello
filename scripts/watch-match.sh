#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
exec python3 scripts/watch-matches.py --config models/current-experiment.json "$@"
