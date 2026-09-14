"""Bounded, portable GPU reproduction; produces no production coverage."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pseudocube.accounting import digest, durable
from pseudocube.batch_experiment import MaskedMatcher
from pseudocube.focus import A619, B619, Config
from pseudocube.gpu import GPUFocus, Matcher, Runtime
from pseudocube.optimized import Accumulator, BATCH_WIDTH, compute_lease
from tools.run_production import compute_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--batches', type=int, default=1)
    parser.add_argument('--compare-baseline', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.batches <= 256 or not 0 <= args.first < B619:
        raise ValueError('bounded canonical scope required')
    if args.output.exists():
        raise ValueError('choose a new output path to retain previous evidence')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise ValueError('select one available GPU with CUDA_VISIBLE_DEVICES')
    config = Config(619, 10**27, 3*10**27, A619, B619)
    end = min(B619, args.first+args.batches*BATCH_WIDTH)
    runtime = Runtime(time.monotonic()+180, 8*1024**3)
    if runtime.cp.cuda.runtime.getDeviceCount() != 1:
        raise ValueError('exactly one visible device required')
    af, bf = (GPUFocus(runtime, f) for f in config.focuses())
    result = compute_lease(MaskedMatcher(runtime, config), af, bf, args.first, end)
    baseline_seconds = None
    if args.compare_baseline:
        reference = Matcher(runtime, config)
        accumulator = Accumulator(args.first, end)
        started = time.monotonic()
        for first in range(args.first, end, BATCH_WIDTH):
            stop = min(first+BATCH_WIDTH, end)
            accumulator.add(first, stop, compute_batch(reference, af, bf, first, stop, 100_000_000))
        baseline = accumulator.finish(time.monotonic()-started)
        expected = {k: v for k, v in baseline.items() if not k.endswith('_seconds')}
        actual = {k: v for k, v in result.items() if not k.endswith('_seconds')}
        if digest(expected) != digest(actual):
            raise ArithmeticError('complete batch/reference mismatch')
        baseline_seconds = baseline['elapsed_seconds']
    record = dict(status='PASS', production_coverage=0, first=str(args.first), end=str(end),
                  baseline_compared=args.compare_baseline, baseline_seconds=baseline_seconds, result=result,
                  source_manifest=json.loads((ROOT/'provenance/source.json').read_text()),
                  peak_device_bytes=runtime.peak_device, compiled_kernels=runtime.compiles)
    durable(args.output, record)
    print(json.dumps(dict(status='PASS', output=str(args.output), production_coverage=0,
                          candidates=result['valid_focused_candidates'], seconds=result['elapsed_seconds'],
                          baseline_seconds=baseline_seconds)))


if __name__ == '__main__':
    main()
