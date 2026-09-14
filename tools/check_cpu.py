"""Build independent C checkers, then run the complete CPU regression suite."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-cube-count', action='store_true')
    args = parser.parse_args()
    build = ROOT/'build'
    build.mkdir(exist_ok=True)
    cc = os.environ.get('CC', 'cc')
    for name in ('cube_count', 'cube_count_independent'):
        subprocess.run([cc, '-O3', '-Wall', '-Wextra', str(ROOT/'review'/f'{name}.c'), '-o', str(build/name)], check=True)
    subprocess.run([cc, '-O2', '-shared', '-fPIC', str(ROOT/'review/wide.c'), '-o', str(build/'wide.so')], check=True)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-q'], cwd=ROOT, check=True)
    if args.full_cube_count:
        for name in ('cube_count', 'cube_count_independent'):
            row = json.loads(subprocess.check_output([str(build/name), '1000000001', '1442249570', '619'], text=True))
            if int(row['count']) != 38322775:
                raise ArithmeticError('full interval genuine-cube count mismatch')
            print(json.dumps(dict(checker=name, status='PASS', count=row['count'])))


if __name__ == '__main__':
    main()
