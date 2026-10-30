#!/bin/sh
# Monitor the policy candidate's 1-second / 20-game experiment.
set -eu
cd "$(dirname "$0")/.."
exec python3 scripts/watch-matches.py --config models/v1.8-policy-1s-experiments.json "$@"
