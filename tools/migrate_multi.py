#!/usr/bin/env python3
"""Import the stopped single worker's exact committed prefix into a fresh shared ledger."""
from __future__ import annotations
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pseudocube.accounting import durable
from pseudocube.multi import DEVICES, SharedFrontier, load_multi
from pseudocube.production import Frontier
from tools.run_production import frozen_build


def migrate(old: Path, new: Path, config_path: Path, build: str) -> dict[str, Any]:
    multi, raw, math = load_multi(config_path)
    if new.exists() or (old / 'handoff-to-multi.json').exists():
        raise ValueError('destination or prior handoff exists; reconcile instead of overwriting')
    # Operator first gracefully stops the old service; this lock independently
    # establishes that the previous worker no longer owns its ledger.
    with (old / 'owner.lock').open('a') as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        release = json.loads((old / 'release.json').read_text())
        if release['configuration'] != raw or release['mathematics'] != math.manifest():
            raise ValueError('legacy mathematics or configuration differs')
        staged = new.with_name(new.name + '.incoming-' + uuid.uuid4().hex)
        archive = staged / 'imported-single' / 'coverage'
        shutil.copytree(old / 'coverage', archive)
        legacy = Frontier(archive, dict(production=raw, mathematics=math.manifest()), release['build'], math.B,
                          raw['tile_width'] * raw['tiles_per_commit'])
        try:
            if legacy.db.execute("SELECT count(*) FROM work WHERE state='running'").fetchone()[0]:
                raise ValueError('legacy worker has an unfinished attempt; reconcile before migration')
            legacy.validate_last_receipt()
            audit = legacy.audit_production()
            state = legacy.state()
            receipts = {r['receipt']: r['sha'] for r in legacy.db.execute("SELECT receipt,sha FROM work WHERE state='committed'")}
            if int(state['commits']) != audit['commits']:
                raise ValueError('legacy commit count mismatch')
            totals = dict(candidates=0, cubes=0, subtiles=0)
            for name in receipts:
                result = json.loads(gzip.decompress((archive / name).read_bytes()))['result']
                totals['candidates'] += int(result['valid_focused_candidates'])
                totals['cubes'] += int(result['genuine_cubes'])
                totals['subtiles'] += len(result['subtile_scopes'])
            if any(str(value) != state[key] for key, value in totals.items()):
                raise ValueError('legacy aggregate counts differ from receipts')
        finally:
            legacy.db.close()
        source_sha = hashlib.sha256((archive / 'coverage.sqlite3').read_bytes()).hexdigest()
        certificate = dict(status='PASS', created_unix=time.time(), authorization=multi['authorization'],
                           old_state=str(old), new_state=str(new), old_release=release,
                           old_database_sha256=source_sha, receipt_sha256=receipts,
                           audit=audit, boundary_v=state['cursor'], old_state_values=state,
                           scope='Preserves primary execution evidence; no independent arithmetic replay or minimum claim.')
        certificate_sha = durable(staged / 'migration.json', certificate)
        baseline = dict(**{k: state[k] for k in ('cursor','commits','candidates','cubes','subtiles','campaign_state')},
                        certificate_sha256=certificate_sha, imported_coverage='../imported-single/coverage')
        shared = SharedFrontier(staged / 'coverage', dict(multi=multi, production=raw, mathematics=math.manifest()), build,
                                math.B, raw['tile_width'] * raw['tiles_per_commit'], DEVICES, baseline)
        shared.audit_shared()
        shared.db.close()
        # fsync the copied evidence before making the destination available.
        for path in staged.rglob('*'):
            if path.is_file():
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
        for path in [p for p in staged.rglob('*') if p.is_dir()] + [staged]:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        os.rename(staged, new)
        fd = os.open(new.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        handoff = dict(status='MIGRATED', boundary_v=state['cursor'], old_commits=state['commits'],
                       shared_state=str(new), build=build, certificate_sha256=certificate_sha)
        # A systemd ExecCondition prevents the old worker from reopening this
        # now delegated suffix after any future reboot/manual start.
        durable(old / 'handoff-to-multi.json', handoff)
        return handoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-state', type=Path, required=True)
    parser.add_argument('--new-state', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.old_state.resolve(), args.new_state.resolve(),
                             ROOT / 'configs/pseudocube_619_multi.json', frozen_build())))


if __name__ == '__main__':
    main()
