#!/usr/bin/env python3
"""Fetch only the two selected public Othello corpora, with provenance/hashes."""
import argparse
import concurrent.futures
import datetime
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import struct
import urllib.parse
import urllib.request

WTHOR = 'https://www.ffothello.org/informatique/la-base-wthor/'
LOGISTELLO = 'https://skatgame.net/mburo/logbook.gam.gz'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(value for key, value in attrs if key == 'href' and value)


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Otello-research/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def fetch(folder):
    folder.mkdir(parents=True, exist_ok=True)
    parser = Links()
    page = get(WTHOR)
    parser.feed(page.decode('utf-8'))
    (folder / 'wthor-source.html').write_bytes(page)
    urls = sorted({urllib.parse.urljoin(WTHOR, link) for link in parser.links
                   if re.fullmatch(r'WTH_\d{4}\.wtb', Path(urllib.parse.urlparse(link).path).name, re.I)})
    if not urls:
        raise ValueError('No annual WTHOR files found on the official page')

    def download(url):
        path = folder / Path(urllib.parse.urlparse(url).path).name
        data = path.read_bytes() if path.exists() else get(url)
        if path.suffix.lower() == '.wtb':
            if len(data) < 16:
                raise ValueError(f'Truncated archive: {url}')
            games = struct.unpack_from('<I', data, 4)[0]
            if len(data) != 16 + 68 * games or data[12] not in (0, 8) or data[13] != 0:
                raise ValueError(f'Invalid 8x8 archive: {url}')
        else:
            raw = gzip.decompress(data)
            games = sum(bool(line.strip()) for line in raw.splitlines())
            (folder / 'logbook.gam').write_bytes(raw)
        path.write_bytes(data)
        return dict(file=path.name, url=url, bytes=len(data), games=games,
                    sha256=hashlib.sha256(data).hexdigest())

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        entries = list(executor.map(download, urls + [LOGISTELLO]))
    manifest = dict(fetchedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    sources=[WTHOR, 'https://skatgame.net/mburo/log.html'], files=entries,
                    usage='Both source pages invite evaluation training; no independent MIT/CC data license found. Keep attribution; raw data stays local.')
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(files=len(entries), games=sum(e['games'] for e in entries),
                          downloadedBytes=sum(e['bytes'] for e in entries),
                          logistelloRawBytes=(folder / 'logbook.gam').stat().st_size)))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, default=Path('.build/training/raw'))
    fetch(cli.parse_args().output)
