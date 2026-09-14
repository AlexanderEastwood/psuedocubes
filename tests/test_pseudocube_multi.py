from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import tempfile
import time
from typing import Any
import unittest
from unittest.mock import patch

from pseudocube.accounting import durable
from pseudocube.multi import DEVICES, SharedFrontier, load_multi
from pseudocube.production import Frontier, save_compressed
from tools.migrate_multi import migrate

ROOT = Path(__file__).resolve().parents[1]
OWNERS = ('a', 'b', 'c', 'd')
BASE = dict(cursor='0', commits='0', candidates='0', cubes='0', subtiles='0', campaign_state='searching')


def receipt(first: int, end: int, hit: bool = False) -> dict[str, Any]:
    return dict(status='PASS', host_verified=True, valid_focused_candidates=11, genuine_cubes=1,
                cpu_confirmed_hits=[dict(n='test-only-witness', valid_pseudocube=True)] if hit else [],
                subtile_scopes=[dict(first=str(first), end=str(end))])


def connection(path: Path, baseline: dict[str, Any] | None = None, end: int = 10000) -> SharedFrontier:
    return SharedFrontier(path, {'test': True}, 'build', end, 10, OWNERS, baseline)


def concurrent_worker(path: str, owner: str) -> list[tuple[int, int]]:
    db = connection(Path(path))
    scopes = []
    try:
        for _ in range(20):
            job = db.next_batch(owner)
            assert job
            time.sleep(0.002 * (OWNERS.index(owner) + 1))
            db.complete(owner, job, receipt(job[2], job[3]))
            scopes.append((job[2], job[3]))
    finally:
        db.db.close()
    return scopes


class MultiTests(unittest.TestCase):
    def test_real_concurrent_allocation_and_commits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            db = connection(path, BASE)
            db.db.close()
            with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
                rows = list(pool.map(concurrent_worker, [str(path)] * 4, OWNERS))
            scopes = sorted(scope for worker in rows for scope in worker)
            self.assertEqual(scopes, [(n, n + 10) for n in range(0, 800, 10)])
            db = connection(path)
            audit = db.audit_shared()
            self.assertEqual(audit['totals']['commits'], 80)
            self.assertEqual(audit['worker_commits'], dict.fromkeys(OWNERS, 20))
            self.assertEqual(audit['committed_v_end'], '800')
            db.db.close()

    def test_out_of_order_and_sticky_global_candidate_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = connection(Path(directory), BASE)
            a, b, c = (db.next_batch(owner) for owner in OWNERS[:3])
            assert a and b and c
            db.complete('b', b, receipt(b[2], b[3]))
            self.assertEqual(db.state()['cursor'], '0')
            self.assertEqual(db.audit_shared()['totals']['commits'], 1)
            db.complete('a', a, receipt(a[2], a[3], hit=True))
            self.assertEqual(db.state()['cursor'], '20')
            self.assertIsNone(db.next_batch('d'))
            db.complete('c', c, receipt(c[2], c[3]))
            self.assertEqual(db.state()['campaign_state'], 'candidate-found')
            self.assertEqual(db.state()['cursor'], '30')
            self.assertEqual(db.db.execute('SELECT count(*) FROM candidates').fetchone()[0], 1)
            db.audit_shared()
            db.db.close()

    def test_owner_recovery_does_not_steal_live_leases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            db = connection(path, BASE)
            a, b = db.next_batch('a'), db.next_batch('b')
            assert a and b
            with self.assertRaises(ValueError):
                db.next_batch('a')
            with self.assertRaises(ValueError):
                db.complete('c', a, receipt(a[2], a[3]))
            db.db.close()
            db = connection(path)
            self.assertEqual(db.recover('a', 'isolated test: prior owner gone'), 1)
            retry = db.next_batch('a')
            assert retry
            self.assertEqual((retry[0], retry[2], retry[3]), (a[0], a[2], a[3]))
            self.assertNotEqual(retry[1], a[1])
            with self.assertRaises(ValueError):
                db.complete('a', a, receipt(a[2], a[3]))
            db.complete('b', b, receipt(b[2], b[3]))
            db.complete('a', retry, receipt(retry[2], retry[3]))
            self.assertEqual(db.audit_shared()['totals']['commits'], 2)
            db.db.close()

    def test_orphan_receipt_is_not_coverage_and_corruption_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = connection(Path(directory), BASE)
            job = db.next_batch('a')
            assert job

            def interrupted(path: Path, value: dict[str, Any]) -> str:
                save_compressed(path, value)
                raise OSError('crash after durable receipt, before commit')

            with patch('pseudocube.multi.save_compressed', side_effect=interrupted):
                with self.assertRaises(OSError):
                    db.complete('a', job, receipt(job[2], job[3]))
            self.assertEqual(db.state()['cursor'], '0')
            self.assertEqual(db.audit_shared()['totals']['commits'], 0)
            db.fail_lease('a', job, 'isolated injected failure')
            retry = db.next_batch('a')
            assert retry
            db.complete('a', retry, receipt(retry[2], retry[3]))
            row = db.db.execute("SELECT * FROM leases WHERE state='committed'").fetchone()
            (db.directory / row['receipt']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                db.validate_owner_receipt('a')
            db.db.close()

    def test_end_tail_and_binding_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            db = connection(path, BASE, end=23)
            jobs = [db.next_batch(owner) for owner in OWNERS]
            self.assertIsNone(jobs[3])
            for owner, job in reversed(list(zip(OWNERS[:3], jobs[:3]))):
                assert job
                db.complete(owner, job, receipt(job[2], job[3]))
            self.assertEqual(db.state()['campaign_state'], 'complete')
            self.assertTrue(db.audit_shared()['full_interval_complete'])
            with self.assertRaises(ValueError):
                SharedFrontier(path, {'test': True}, 'different', 23, 10, OWNERS)
            with self.assertRaises(ValueError):
                connection(path, dict(BASE, cursor='10'), end=23)
            db.db.close()

    def test_migration_preserves_prefix_and_receipts(self) -> None:
        multi, raw, math = load_multi(ROOT / 'configs/pseudocube_619_multi.json')
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / 'old', Path(directory) / 'new'
            legacy = Frontier(old / 'coverage', dict(production=raw, mathematics=math.manifest()), 'old-build', math.B, 6400000000)
            for _ in range(2):
                job = legacy.next_batch()
                assert job
                legacy.complete(*job, receipt(job[2], job[3]))
            legacy.db.close()
            durable(old / 'release.json', dict(configuration=raw, mathematics=math.manifest(), build='old-build'))
            original = hashlib.sha256((old / 'coverage/coverage.sqlite3').read_bytes()).hexdigest()
            handoff = migrate(old, new, ROOT / 'configs/pseudocube_619_multi.json', 'new-build')
            self.assertEqual(handoff['boundary_v'], '12800000000')
            self.assertEqual(hashlib.sha256((old / 'coverage/coverage.sqlite3').read_bytes()).hexdigest(), original)
            db = SharedFrontier(new / 'coverage', dict(multi=multi, production=raw, mathematics=math.manifest()), 'new-build', math.B, 6400000000, DEVICES)
            job = db.next_batch(DEVICES[1])
            assert job
            self.assertEqual(job[2], 12800000000)
            db.complete(DEVICES[1], job, receipt(job[2], job[3]))
            self.assertEqual(db.audit_shared()['totals']['commits'], 3)
            self.assertEqual(json.loads((new / 'migration.json').read_text())['old_database_sha256'], original)
            with self.assertRaises(ValueError):
                migrate(old, new, ROOT / 'configs/pseudocube_619_multi.json', 'new-build')
            db.db.close()


if __name__ == '__main__':
    unittest.main()
