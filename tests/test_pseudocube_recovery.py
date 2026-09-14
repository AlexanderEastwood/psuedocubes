from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pseudocube.accounting import Ledger


class RecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)
        self.ledger=Ledger(self.path,{'search_type':'pseudocube','p':619},'build-v1')
        self.work=self.ledger.plan([dict(kind='v-window',first='0',end='100')])[0]

    def tearDown(self) -> None:
        self.ledger.db.close();self.temp.cleanup()

    def test_atomic_output_before_commit_and_retry(self) -> None:
        token=self.ledger.claim(self.work)
        with patch('pseudocube.accounting.os.replace',side_effect=OSError('interrupted write')):
            with self.assertRaises(OSError):self.ledger.commit(self.work,token,dict(status='PASS',host_verified=True))
        self.assertEqual(self.ledger.audit()['states'],{'running':1})
        self.ledger.reclaim(self.work,token,'test process has stopped')
        new=self.ledger.claim(self.work)
        with self.assertRaises(ValueError):self.ledger.commit(self.work,token,dict(status='PASS',host_verified=True))
        self.ledger.commit(self.work,new,dict(status='PASS',host_verified=True,pairs='12'))
        with self.assertRaises(ValueError):self.ledger.claim(self.work)
        self.assertEqual(len(self.ledger.audit()['committed_results']),1)

    def test_corruption_and_configuration_fail_closed(self) -> None:
        token=self.ledger.claim(self.work)
        self.ledger.commit(self.work,token,dict(status='PASS',host_verified=True))
        (self.path/'outputs'/f'{token}.json').write_text('{}')
        with self.assertRaises(ValueError):self.ledger.audit()
        with self.assertRaises(ValueError):Ledger(self.path,{'search_type':'pseudosquare','p':619},'build-v1')
        with self.assertRaises(ValueError):self.ledger.plan([dict(first='1',end='100')])

    def test_overflow_kernel_error_split_and_no_double_count(self) -> None:
        for flag in ('overflow','kernel_error'):
            token=self.ledger.claim(self.work)
            with self.assertRaises(ValueError):self.ledger.commit(self.work,token,dict(status='PASS',host_verified=True,**{flag:True}))
        children=self.ledger.split(self.work)
        self.assertEqual(len(children),2)
        with self.assertRaises(ValueError):self.ledger.claim(self.work)
        for child in children:
            token=self.ledger.claim(child);self.ledger.commit(child,token,dict(status='PASS',host_verified=True))
        self.assertEqual(self.ledger.audit()['states'],{'split':1,'committed':2})
        scopes=[json.loads(r[0]) for r in self.ledger.db.execute('SELECT scope FROM work WHERE parent=?',(self.work,))]
        self.assertEqual(sorted((int(r['first']),int(r['end'])) for r in scopes),[(0,50),(50,100)])

    def test_persistent_budget_reservation_and_build_binding(self) -> None:
        reservation=self.ledger.reserve('gpu',600,600)
        again=Ledger(self.path,{'search_type':'pseudocube','p':619},'build-v1')
        try:
            with self.assertRaisesRegex(RuntimeError,'BUDGET_EXHAUSTED'):again.reserve('gpu',1,600)
        finally:again.db.close()
        self.ledger.settle(reservation,20)
        self.ledger.reserve('gpu',580,600)
        token=self.ledger.claim(self.work)
        self.ledger.build='different-build'
        with self.assertRaises(ValueError):self.ledger.commit(self.work,token,dict(status='PASS',host_verified=True))

    def test_validation_replays_never_add_coverage(self) -> None:
        token=self.ledger.claim(self.work)
        self.ledger.commit(self.work,token,dict(status='PASS',host_verified=True,pairs='10'))
        replay=self.ledger.begin_validation(self.work,'packed table A/B')
        self.ledger.finish_validation(replay,dict(status='PASS',host_verified=True,pairs='10'))
        audit=self.ledger.audit()
        self.assertEqual(len(audit['committed_results']),1);self.assertEqual(len(audit['validations']),1)
        with self.assertRaises(ValueError):self.ledger.plan([dict(first='0',end='100')]*2)
        self.ledger.quarantine(self.work,'injected independent-replay disagreement')
        self.assertEqual(self.ledger.audit()['states'],{'quarantined':1})
        self.assertEqual(len(self.ledger.audit()['committed_results']),0)


if __name__=='__main__':unittest.main()
