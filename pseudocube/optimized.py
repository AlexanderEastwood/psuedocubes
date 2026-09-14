"""Qualified batched bitset execution and compact, reproducible lease receipts."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from .accounting import digest, encoded
from .batch_experiment import MaskedMatcher, prepare
from .gpu import GPUFocus
from .multi import validate_result

PROFILE: dict[str, Any] = dict(engine='residue-bitset-batch-v1', tile_width=100_000_000,
    tiles_per_compute_batch=64, compute_batches_per_lease=256, receipt_storage='sqlite-gzip',
    receipt_format='compact-batch-v1', authorization='Deploy to production')
BATCH_WIDTH = PROFILE['tile_width'] * PROFILE['tiles_per_compute_batch']
LEASE_WIDTH = BATCH_WIDTH * PROFILE['compute_batches_per_lease']
COUNTS = ('focus_u', 'focus_v', 'cartesian_opportunities', 'valid_focused_candidates',
          'genuine_cubes', 'local_survivors', 'cube_checks')
TIMINGS = ('kernel_seconds', 'focus_enumeration_seconds', 'transfer_seconds',
           'verification_seconds', 'prefilter_build_seconds')


class Accumulator:
    def __init__(self, first: int, end: int) -> None:
        if not 0 <= first < end or end-first > LEASE_WIDTH:
            raise ValueError('invalid optimized lease')
        self.first, self.end, self.cursor = first, end, first
        self.counts = dict.fromkeys(COUNTS, 0)
        self.timings = dict.fromkeys(TIMINGS, 0.0)
        self.values: list[str] = []
        self.hits: list[dict[str, Any]] = []
        self.groups: list[int] = []
        self.filters: list[list[int]] | None = None
        self.transcript = hashlib.sha256()
        self.batches = self.subtiles = 0

    def add(self, first: int, end: int, row: dict[str, Any]) -> None:
        if first != self.cursor or end != min(first+BATCH_WIDTH, self.end):
            raise ValueError('gap, overlap or changed compute batch geometry')
        validate_result(first, end, row)
        expected = [(v, min(v+PROFILE['tile_width'], end)) for v in range(first, end, PROFILE['tile_width'])]
        actual = [(int(s['first']), int(s['end'])) for s in row['subtile_scopes']]
        if actual != expected:
            raise ValueError('changed arithmetic subtile geometry')
        if self.filters is None:
            self.filters = [list(group) for group in row['filter_groups']]
            self.groups = [0] * len(self.filters)
        if row['filter_groups'] != self.filters or len(row['group_survivors']) != len(self.groups):
            raise ValueError('residue filters changed during execution')
        for k in COUNTS:
            self.counts[k] += int(row[k])
        for k in TIMINGS:
            self.timings[k] += float(row.get(k, 0.0))
        for i, count in enumerate(row['group_survivors']):
            self.groups[i] += int(count)
        self.values.extend(row['all_values'])
        self.hits.extend(row['cpu_confirmed_hits'])
        self.transcript.update(encoded(row['subtile_scopes']))
        self.batches += 1
        self.subtiles += len(actual)
        self.cursor = end

    def finish(self, seconds: float) -> dict[str, Any]:
        if self.cursor != self.end:
            raise ValueError('unfinished lease cannot become coverage')
        row = dict(status='PASS', host_verified=True, receipt_format=PROFILE['receipt_format'],
            tile_width=PROFILE['tile_width'], compute_batch_width=BATCH_WIDTH,
            compute_batches=self.batches, subtiles_executed=self.subtiles,
            subtile_scopes=[dict(first=str(self.first), end=str(self.end),
                transcript_sha256=self.transcript.hexdigest(), output_sha256=digest(self.values))],
            **self.counts, **self.timings, group_survivors=self.groups, filter_groups=self.filters,
            all_values=self.values, cpu_confirmed_hits=self.hits, overflow_splits=[], elapsed_seconds=seconds,
            claim='Executed primary coverage; independent arithmetic replay required before any minimum claim.')
        validate_result(self.first, self.end, row)
        return row


def compute_lease(matcher: MaskedMatcher, af: GPUFocus, bf: GPUFocus, first: int, end: int,
                  heartbeat: Callable[[], None] = lambda: None) -> dict[str, Any]:
    started = time.monotonic()
    acc = Accumulator(first, end)
    for start in range(first, end, BATCH_WIDTH):
        stop = min(start+BATCH_WIDTH, end)
        batch = prepare(matcher.runtime, matcher.config, af, bf, start, stop, PROFILE['tile_width'])
        acc.add(start, stop, matcher.run(batch))
        heartbeat()
    return acc.finish(time.monotonic()-started)


def allocated_bytes(directory: Path) -> int:
    """Count allocated blocks; concurrent atomic replacement is allowed for .tmp only."""
    total = directory.stat().st_blocks * 512
    with os.scandir(directory) as entries:
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    total += allocated_bytes(Path(entry.path))
                else:
                    total += entry.stat(follow_symlinks=False).st_blocks * 512
            except FileNotFoundError:
                if not entry.name.endswith('.tmp'):
                    raise
    return total


def ledger_settings(directory: Path) -> tuple[str, int, dict[str, Any]]:
    import sqlite3
    with sqlite3.connect((directory/'coverage.sqlite3').as_uri()+'?mode=ro', uri=True) as db:
        state = dict(db.execute('SELECT key,value FROM meta'))
    return state.get('active_build', state['frozen_build']), int(state['batch_width']), json.loads(state.get('runtime_profile', '{}'))
