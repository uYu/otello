#!/bin/sh
# Monitor the outcome-value candidate without controlling the match process.
set -eu
cd "$(dirname "$0")/.."
exec python3 scripts/watch-matches.py --config models/v1.9-outcome-experiments.json "$@"
