#!/usr/bin/env python3
"""Validate downloaded hashes and replay/deduplicate through the native converter."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def prepare(args):
    manifest = json.loads((args.raw / 'manifest.json').read_text())
    inputs = []
    for entry in manifest['files']:
        if Path(entry['file']).name != entry['file']:
            raise ValueError('Invalid manifest filename')
        path = args.raw / entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Download checksum mismatch: ' + str(path))
        if path.suffix.lower() == '.wtb':
            inputs.append(path)
        elif path.name == 'logbook.gam.gz':
            gam = args.raw / 'logbook.gam'
            gam.write_bytes(gzip.decompress(path.read_bytes()))
            inputs.append(gam)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(args.converter)]
    for path in sorted(inputs):
        command += ['--input', str(path)]
    command += ['--output', str(args.output)]
    run = subprocess.run(command, capture_output=True, text=True, check=True)
    json.loads(run.stdout)
    args.output.with_suffix('.report.json').write_text(run.stdout)
    print(run.stdout, end='')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--raw', type=Path, default=Path('.build/training/raw'))
    cli.add_argument('--output', type=Path, default=Path('.build/training/games.jsonl'))
    cli.add_argument('--converter', type=Path, default=Path('.build/native/otello_dataset'))
    prepare(cli.parse_args())
