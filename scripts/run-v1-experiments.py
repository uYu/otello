#!/usr/bin/env python3
"""Run the two frozen v1 experiments; monitor with watch-matches.py."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run():
    config = json.loads((ROOT / 'models/v1-experiments.json').read_text())
    model = ROOT / 'models/nnue256-v1.otnnue'
    metadata = json.loads(model.with_suffix('.json').read_text())
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    if model_hash != metadata['sha256']:
        raise ValueError('Frozen v1 model checksum mismatch')
    library = ROOT / ('.build/native-egaroucid/libothello.' + ('dylib' if sys.platform == 'darwin' else 'so'))
    protocol = dict(version='v1', model='models/nnue256-v1.otnnue', modelSha256=model_hash,
                    dataset=metadata['dataset'], checkpointSelection='minimum validation MSE; frozen before matches',
                    nativeLibrarySha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                    comparisonScriptSha256=hashlib.sha256((ROOT / 'scripts/compare-evaluators.py').read_bytes()).hexdigest(),
                    platform=platform.platform(), settings=config, maxDepth=64, seed=20261027,
                    bookVariety=1, bookFallback='same NNUE search; zero Egaroucid search calls',
                    limits='Repeated standard initial position, alternating colors; no independent opening-sample claim.')
    (ROOT / 'models/v1-baseline-protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for item in config['experiments']:
        if hashlib.sha256(model.read_bytes()).hexdigest() != model_hash:
            raise ValueError('Model changed between experiments')
        print(item['name'], flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/compare-evaluators.py'),
                        '--library', str(library), '--model', str(model), '--start', 'initial',
                        '--experiment', item['experiment'], '--games', str(item['games']),
                        '--milliseconds', str(config['millisecondsPerMove']), '--seed', '20261027',
                        '--output', str(ROOT / item['report'])], cwd=ROOT, check=True)


if __name__ == '__main__':
    run()
