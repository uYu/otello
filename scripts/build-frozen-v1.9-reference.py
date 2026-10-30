#!/usr/bin/env python3
"""Restore checksum-verified frozen v1.9 source and build without modifying it."""
import hashlib,json,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'.build/v1.9-frozen-reference/source'
BUILD=ROOT/'.build/native-v1.9-reference'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    manifest_path=ROOT/'models/archive/v1.9-default-sources.json'
    manifest=json.loads(manifest_path.read_text());archive=ROOT/manifest['archive']
    assert sha(archive)==manifest['sha256']
    SOURCE.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as source:
        for name,digest in manifest['files'].items():
            member=source.getmember(name);assert member.isfile() and not Path(name).is_absolute()
            path=SOURCE/name;assert path.resolve().is_relative_to(SOURCE.resolve())
            content=source.extractfile(member).read();assert hashlib.sha256(content).hexdigest()==digest,name
            if path.exists():assert sha(path)==digest,name
            else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    subprocess.run(['cmake','-S',str(SOURCE),'-B',str(BUILD),'-DCMAKE_BUILD_TYPE=Release'],check=True)
    subprocess.run(['cmake','--build',str(BUILD),'-j','2'],check=True)
    subprocess.run(['ctest','--test-dir',str(BUILD),'--output-on-failure'],check=True)
    suffix='dylib' if sys.platform=='darwin' else 'so';library=BUILD/f'libothello.{suffix}'
    record=dict(complete=True,sourceManifest=str(manifest_path.relative_to(ROOT)),sourceManifestSha256=sha(manifest_path),sourceArchiveSha256=sha(archive),
        sourceFilesUnmodified=True,sourceFiles=manifest['files'],library=str(library.relative_to(ROOT)),librarySha256=sha(library),
        originalRecordedLibrarySha256=json.loads((ROOT/'models/v1.9-default.json').read_text())['nativeLibrarySha256'],
        note='Rebuilt from exact frozen source on the current host; binary hash is recorded separately and is not claimed equal to the original historical build.')
    (BUILD/'frozen-build-provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(complete=True,library=record['library'],librarySha256=record['librarySha256'])))


if __name__=='__main__':main()
