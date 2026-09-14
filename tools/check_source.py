"""Check the exact production-derived source files included in this repository."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((ROOT/'provenance/source.json').read_text())
    for name, expected in manifest['copied_unchanged_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('production-derived source changed: '+name)
    print(json.dumps(dict(status='PASS', files=len(manifest['copied_unchanged_sha256']),
                          origin_production_build=manifest['origin_production_build'])))


if __name__ == '__main__':
    main()
