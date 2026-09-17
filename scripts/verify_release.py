#!/usr/bin/env python3
"""Verify every distributed file against the public release manifest."""
import hashlib
import json
from pathlib import Path, PurePosixPath
ROOT = Path(__file__).resolve().parents[1]
def main():
    files = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text())['files']
    for rel, info in files.items():
        path = PurePosixPath(rel)
        target = ROOT / rel
        if path.is_absolute() or '..' in path.parts or target.is_symlink() or not target.is_file():
            raise SystemExit('Unsafe or missing release file: ' + rel)
        if hashlib.sha256(target.read_bytes()).hexdigest() != info['sha256']:
            raise SystemExit('Hash mismatch: ' + rel)
    print(json.dumps({'status': 'PASS', 'release_files': len(files), 'scope': 'Public file integrity; not scientific recomputation'}))
if __name__ == '__main__':
    main()
