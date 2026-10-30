#!/usr/bin/env python3
"""Download the official v0002 transcripts locally; never package or redistribute."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ASSETS = (
    ('Egaroucid_Train_Data_v0002_0.zip', 1303950421,
     '7123a56b647985b17f90b864e00ae5851cb015296315432965f4ad224ca7c87a'),
    ('Egaroucid_Train_Data_v0002_1.zip', 1339651484,
     '9bbda568b8c3163c9d7e15c3d5a4b5d47693401d1b094928c1a8077b3538b73b'),
)
SOURCE = 'https://github.com/Nyanyan/Egaroucid/releases/tag/training_data_v0002'
TERMS = 'https://www.egaroucid.nyanyan.dev/en/technology/train-data/'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(folder):
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, size, expected in ASSETS:
        path = folder / name
        url = SOURCE.replace('/tag/', '/download/') + '/' + name
        if not path.exists():
            partial = path.with_suffix('.zip.partial')
            for attempt in range(8):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == size:
                    break
                if offset > size:
                    raise ValueError('Oversized partial download: ' + str(partial))
                headers = {'User-Agent': 'Otello-local-training/1.0'}
                if offset:
                    headers['Range'] = f'bytes={offset}-{size - 1}'
                request = urllib.request.Request(url + '?local_fetch=' + str(time.time_ns()), headers=headers)
                try:
                    with urllib.request.urlopen(request, timeout=45) as response:
                        if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                            raise ValueError('Server did not honor resume range')
                        reported = time.monotonic()
                        with partial.open('ab' if offset else 'wb') as output:
                            while block := response.read(1024 * 1024):
                                output.write(block)
                                offset += len(block)
                                if offset > size:
                                    raise ValueError('Download exceeds pinned release size')
                                if time.monotonic() - reported >= 10:
                                    print(json.dumps(dict(file=name, downloaded=offset, total=size,
                                                          percent=round(100 * offset / size, 1))), flush=True)
                                    reported = time.monotonic()
                    if offset == size:
                        break
                except (OSError, TimeoutError) as error:
                    print(json.dumps(dict(file=name, retry=attempt + 1, error=str(error))), flush=True)
                    if attempt == 7:
                        raise
                    time.sleep(min(2 ** attempt, 15))
            if not partial.exists() or partial.stat().st_size != size or sha256(partial) != expected:
                raise ValueError('Release size/SHA-256 mismatch: ' + name)
            partial.replace(path)
        if path.stat().st_size != size or sha256(path) != expected:
            raise ValueError('Existing archive differs from pinned release: ' + name)
        entries.append(dict(file=name, url=url, bytes=size, sha256=expected))
        print(json.dumps(dict(verified=name, bytes=size)), flush=True)
    manifest = dict(version='egaroucid-v0002', games=52000000,
                    source=SOURCE, terms=TERMS,
                    usage='Local AI development allowed; redistribution of the data prohibited. '
                          'Credit Takuto Yamana / Egaroucid Free Training Data. '
                          'Exclude the first N random placements in folder N from training labels.',
                    verifiedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(), files=entries)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.build/training/egaroucid-v0002/raw'))
    fetch(parser.parse_args().output)
