from __future__ import annotations
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from pseudocube.multi import SharedFrontier, validate_result
from pseudocube.optimized import Accumulator, BATCH_WIDTH, COUNTS, PROFILE, allocated_bytes
from tests.test_pseudocube_multi import BASE, OWNERS, receipt


def db_at(path: Path, build: str = 'old', width: int = 10, baseline: bool = False) -> SharedFrontier:
    return SharedFrontier(path, {'test': 'upgrade'}, build, 1000, width, OWNERS,
                          BASE if baseline else None, PROFILE if build == 'new' else None)


def crash_after_blob(path: str) -> None:
    db = db_at(Path(path), 'new', 100)
    job = db.next_batch('a')
    assert job
    db.db.create_function('isolated_exit', 0, lambda: os._exit(91))
    db.db.execute('CREATE TEMP TRIGGER isolated_exit AFTER INSERT ON receipt_blobs BEGIN SELECT isolated_exit(); END')
    db.complete('a', job, receipt(job[2], job[3]))


def compute_row(first: int, end: int) -> dict[str, Any]:
    return dict(status='PASS', host_verified=True, **dict.fromkeys(COUNTS, 0),
                group_survivors=[0], filter_groups=[[193]], all_values=[], cpu_confirmed_hits=[],
                subtile_scopes=[dict(first=str(v), end=str(min(v+100_000_000, end)))
                                for v in range(first, end, 100_000_000)])


class OptimizedTests(unittest.TestCase):
    def test_upgrade_preserves_old_evidence_and_rejects_old_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            old = db_at(path, baseline=True)
            job = old.next_batch('a'); assert job
            with self.assertRaises(ValueError):
                old.upgrade('new', 100, PROFILE, {'tests': 'passed'})
            old.complete('a', job, receipt(0, 10))
            old_path = path / old.db.execute('SELECT receipt FROM leases').fetchone()[0]
            original = old_path.read_bytes()
            cert = old.upgrade('new', 100, PROFILE, {'tests': 'passed'})
            self.assertEqual(cert['boundary'], '10')
            with self.assertRaises(ValueError): old.next_batch('b')
            old.db.close()
            with self.assertRaises(ValueError): db_at(path)
            db = db_at(path, 'new', 100)
            job = db.next_batch('b'); assert job
            self.assertEqual(job[2:], (10, 110))
            db.complete('b', job, receipt(10, 110))
            self.assertEqual(db.audit_shared()['totals']['commits'], 2)
            self.assertEqual(old_path.read_bytes(), original)
            db.db.execute("UPDATE receipt_blobs SET data=x'00'")
            with self.assertRaises(ValueError): db.audit_shared()
            db.db.close()

    def test_process_exit_inside_commit_retains_same_owner_and_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            old = db_at(path, baseline=True)
            old.upgrade('new', 100, PROFILE, {'tests': 'isolated fault probe'})
            old.db.close()
            child = multiprocessing.get_context('spawn').Process(target=crash_after_blob, args=(tmp,))
            child.start(); child.join(20)
            self.assertFalse(child.is_alive())
            self.assertEqual(child.exitcode, 91)
            db = db_at(path, 'new', 100)
            self.assertEqual(db.state()['cursor'], '0')
            self.assertEqual(db.db.execute('SELECT count(*) FROM receipt_blobs').fetchone()[0], 0)
            old_token = db.db.execute('SELECT token FROM leases').fetchone()[0]
            self.assertEqual(db.recover('a', 'owned isolated child exited with 91'), 1)
            job = db.next_batch('a'); assert job
            self.assertEqual(job[2:], (0, 100)); self.assertNotEqual(job[1], old_token)
            with self.assertRaises(ValueError): db.complete('a', (0, old_token, 0, 100), receipt(0, 100))
            db.complete('a', job, receipt(0, 100))
            self.assertEqual(db.audit_shared()['totals']['commits'], 1)
            db.db.close()

    def test_upgrade_certificate_and_profile_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            old = db_at(path, baseline=True)
            old.upgrade('new', 100, PROFILE, {'tests': 'passed'})
            old.db.execute("UPDATE upgrades SET certificate='{}'")
            old.db.close()
            with self.assertRaises(ValueError): db_at(path, 'new', 100)

    def test_compaction_tail_counts_and_gap_rejection(self) -> None:
        first, end = 123, 123 + BATCH_WIDTH + 200_000_001
        acc = Accumulator(first, end)
        with self.assertRaises(ValueError): acc.finish(1)
        with self.assertRaises(ValueError): acc.add(first+1, first+BATCH_WIDTH, compute_row(first+1, first+BATCH_WIDTH))
        acc.add(first, first+BATCH_WIDTH, compute_row(first, first+BATCH_WIDTH))
        acc.add(first+BATCH_WIDTH, end, compute_row(first+BATCH_WIDTH, end))
        row = acc.finish(1)
        self.assertEqual(row['subtiles_executed'], 67)
        self.assertEqual(row['compute_batches'], 2)
        validate_result(first, end, row)
        row['subtiles_executed'] = 66
        with self.assertRaises(ValueError): validate_result(first, end, row)

    def test_storage_scan_tolerates_only_vanishing_temporary_files(self) -> None:
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            entry = MagicMock()
            entry.is_dir.return_value = False
            entry.stat.side_effect = FileNotFoundError('atomic rename')
            entry.name = 'status.json.abc.tmp'
            scan = MagicMock()
            scan.__enter__.return_value = [entry]
            with patch('pseudocube.optimized.os.scandir', return_value=scan):
                self.assertEqual(allocated_bytes(path), path.stat().st_blocks*512)
                entry.name = 'committed.json.gz'
                with self.assertRaises(FileNotFoundError): allocated_bytes(path)


if __name__ == '__main__':
    unittest.main()
