#!/usr/bin/env python3
"""Pack, verify and safely restore the versioned data snapshots (stdlib only)."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/manifest.json'
BOOK_SHA = '0dd454d84aa656a64329930bad796c4a48a4322b7693d50b7040006383db5012'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in ('..', '.') for p in path.parts) or '\\' in name:
        raise ValueError('Unsafe archive path: ' + name)
    return path


def pack():
    raw = ROOT / '.build/training/raw'
    sources = json.loads((raw / 'manifest.json').read_text())
    files = {f'raw/{e["file"]}': raw / e['file'] for e in sources['files']}
    for entry in sources['files']:
        safe_name(entry['file'])
        if digest(raw / entry['file']) != entry['sha256']:
            raise ValueError('Raw source checksum mismatch: ' + entry['file'])
    files['raw/manifest.json'] = raw / 'manifest.json'
    book = ROOT / '.build/deps/Egaroucid/bin/resources/book.egbk3'
    if digest(book) != BOOK_SHA:
        raise ValueError('Opening book differs from the pinned Egaroucid resource')
    with tempfile.TemporaryDirectory(dir=ROOT / '.build', prefix='pack-data-') as folder:
        folder = Path(folder)
        book_source = folder / 'SOURCE.json'
        book_source.write_text(json.dumps(dict(upstream='https://github.com/Nyanyan/Egaroucid',
            revision='2a65c87860be999373088ad833aa00b0844883b6', path='bin/resources/book.egbk3',
            license='GPL-3.0-or-later; upstream repository license', sha256=BOOK_SHA), indent=2) + '\n')
        specs = [dict(id='training-v1', archive='data/training/otello-v1.tar.gz', target='.build/training',
                      files=files, sources='models/training-data-manifest.json',
                      usage='Source pages invite evaluation training; no separate MIT/CC redistribution grant located. Preserve attribution and source terms.'),
                 dict(id='egaroucid-book', archive='data/opening-book/egaroucid-2a65c87860be.tar.gz',
                      target='.build/data/egaroucid', files={'book.egbk3': book,
                      'LICENSE': ROOT / 'third_party/egaroucid/LICENSE', 'SOURCE.json': book_source},
                      license='GPL-3.0-or-later; original upstream repository license retained')]
        assets = []
        for spec in specs:
            archive = ROOT / spec['archive']
            archive.parent.mkdir(parents=True, exist_ok=True)
            entries = {name: dict(bytes=path.stat().st_size, sha256=digest(path))
                       for name, path in sorted(spec['files'].items())}
            with archive.open('wb') as output:
                with gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0, compresslevel=6) as compressed:
                    with tarfile.open(fileobj=compressed, mode='w|') as tar:
                        for name, path in sorted(spec['files'].items()):
                            safe_name(name)
                            info = tarfile.TarInfo(name)
                            info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                            with path.open('rb') as stream:
                                tar.addfile(info, stream)
            if archive.stat().st_size >= 100 * 1024 * 1024:
                raise ValueError('Snapshot exceeds the regular GitHub file limit')
            asset = {key: value for key, value in spec.items() if key != 'files'}
            asset.update(bytes=archive.stat().st_size, sha256=digest(archive), files=entries)
            assets.append(asset)
            print(f'{spec["id"]}: {archive.stat().st_size:,} bytes, SHA-256 {asset["sha256"]}', flush=True)
        MANIFEST.write_text(json.dumps(dict(version=1, snapshot='2026-10-27-v1', assets=assets), indent=2) + '\n')


def restore(verify_only, only, force):
    manifest = json.loads(MANIFEST.read_text())
    if manifest['version'] != 1:
        raise ValueError('Unsupported snapshot manifest')
    (ROOT / '.build').mkdir(exist_ok=True)
    for asset in manifest['assets']:
        if only and asset['id'] != only:
            continue
        archive = ROOT / safe_name(asset['archive'])
        if archive.stat().st_size != asset['bytes'] or digest(archive) != asset['sha256']:
            raise ValueError('Archive checksum mismatch: ' + asset['archive'])
        target_name = safe_name(asset['target'])
        if target_name.parts[0] != '.build':
            raise ValueError('Data targets must be under .build')
        target = ROOT / target_name
        expected, found = asset['files'], set()
        with tempfile.TemporaryDirectory(dir=ROOT / '.build', prefix='restore-data-') as folder:
            staging = Path(folder)
            with tarfile.open(archive, mode='r:gz') as tar:
                for member in tar:
                    name = member.name
                    safe_name(name)
                    if name not in expected or name in found or not member.isfile() or member.size != expected[name]['bytes']:
                        raise ValueError('Unexpected archive member: ' + name)
                    found.add(name)
                    output = staging / name
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, output.open('wb') as destination:
                        shutil.copyfileobj(source, destination)
                    if digest(output) != expected[name]['sha256']:
                        raise ValueError('Member checksum mismatch: ' + name)
            if found != set(expected):
                raise ValueError('Snapshot is missing files')
            staged = [p for p in staging.rglob('*') if p.is_file()]
            # Check every conflict before replacing any existing data.
            for path in staged:
                destination = target / path.relative_to(staging)
                if destination.exists() and digest(destination) != digest(path) and not force:
                    raise ValueError(f'Existing data differs: {destination}; use --force to restore the snapshot')
            if not verify_only:
                for path in staged:
                    destination = target / path.relative_to(staging)
                    if destination.exists() and digest(destination) == digest(path):
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    path.replace(destination)
            print(f'{asset["id"]}: {"verified" if verify_only else "restored"} {len(staged)} files -> {asset["target"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('pack', 'verify', 'restore'))
    parser.add_argument('--only', choices=('training-v1', 'egaroucid-book'))
    parser.add_argument('--force', action='store_true', help='Replace conflicting local snapshot files')
    args = parser.parse_args()
    if args.command == 'pack':
        pack()
    else:
        restore(args.command == 'verify', args.only, args.force)
