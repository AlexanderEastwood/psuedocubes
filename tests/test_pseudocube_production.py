from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from typing import Any
from pseudocube.production import Frontier,load_production


def receipt(first: int,end: int,hits: list[str] | None = None) -> dict[str,Any]:
    middle=(first+end)//2
    return dict(status='PASS',host_verified=True,valid_focused_candidates=11,genuine_cubes=0,
                cpu_confirmed_hits=hits or [],subtile_scopes=[dict(first=str(first),end=str(middle)),dict(first=str(middle),end=str(end))])


class ProductionTests(unittest.TestCase):
    def test_resume_stale_owner_and_exact_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);db=Frontier(p,{'test':True},'build',20,10)
            job=db.next_batch();assert job
            work,token,first,end=job
            with patch('pseudocube.production.os.replace',side_effect=OSError('interrupted receipt')):
                with self.assertRaises(OSError):db.complete(work,token,first,end,receipt(first,end))
            self.assertEqual(db.state()['cursor'],'0');db.db.close()
            db=Frontier(p,{'test':True},'build',20,10)
            self.assertEqual(db.recover_stopped_owner('new process holds exclusive GPU and frontier locks'),1)
            new=db.next_batch();assert new
            self.assertEqual(new[0],work)
            with self.assertRaises(ValueError):db.complete(work,token,first,end,receipt(first,end))
            db.complete(*new,receipt(new[2],new[3]));db.db.close()
            db=Frontier(p,{'test':True},'build',20,10);db.validate_last_receipt()
            final=db.next_batch();assert final;self.assertEqual(final[2:],(10,20))
            db.complete(*final,receipt(final[2],final[3]))
            self.assertIsNone(db.next_batch());self.assertTrue(db.audit_production()['full_interval_complete']);db.db.close()

    def test_candidate_stop_and_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db=Frontier(Path(directory),{'test':True},'build',20,10);job=db.next_batch();assert job
            db.complete(*job,receipt(job[2],job[3],['test-witness']))
            self.assertEqual(db.state()['campaign_state'],'candidate-found');self.assertIsNone(db.next_batch())
            p=db.directory/db.state()['last_receipt'];p.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):db.validate_last_receipt()
            db.db.close()

    def test_gap_and_build_changes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);db=Frontier(p,{'test':True},'build',20,10);job=db.next_batch();assert job
            bad=receipt(job[2],job[3]);bad['subtile_scopes'][1]['first']='6'
            with self.assertRaises(ValueError):db.complete(*job,bad)
            self.assertEqual(db.state()['cursor'],'0')
            with self.assertRaises(ValueError):Frontier(p,{'test':True},'changed-build',20,10)
            with self.assertRaises(ValueError):Frontier(p,{'test':True},'build',30,10)
            db.db.close()

    def test_production_configuration_rejects_pilot_and_changes(self) -> None:
        root=Path(__file__).resolve().parents[1];path=root/'configs/pseudocube_619_production.json'
        raw,_=load_production(path)
        with self.assertRaises(ValueError):load_production(root/'configs/pseudocube_619_pilot.json')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'
            for key,value in [('device_uuid','different'),('p',613),('hi_inclusive',str(4*10**27)),('tile_width',10**9),('layout','serial'),('stop_on_candidate',False),('authorization',{})]:
                changed=copy.deepcopy(raw);changed[key]=value;path.write_text(json.dumps(changed))
                with self.subTest(key=key),self.assertRaises(ValueError):load_production(path)


if __name__=='__main__':unittest.main()
