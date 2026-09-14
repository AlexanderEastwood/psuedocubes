from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
from typing import Any
from pseudocube.configuration import load

ROOT=Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_all_pilot_fields_are_bound(self) -> None:
        raw=json.loads((ROOT/'configs/pseudocube_619_pilot.json').read_text())
        load(ROOT/'configs/pseudocube_619_pilot.json')
        changes: list[tuple[str,Any]]=[('schema_version',True),('search_type','pseudosquare'),
            ('mode','production'),('p',613),('lo_exclusive',str(10**27-1)),
            ('hi_inclusive',str(3*10**27+1)),('deterministic_test_seed',1),
            ('production_verification_policy','APPROVED'),('focus',dict(raw['focus'],A='2')),
            ('authority',dict(raw['authority'],allow_production_search=True)),
            ('authority',dict(raw['authority'],allow_multi_gpu=0))]
        changes += [('pilot_budget',dict(raw['pilot_budget'],**{key:value})) for key,value in
            [('max_devices',2),('max_devices',1.0),('device_uuid','GPU-invalid'),
             ('max_total_gpu_allocation_wall_seconds',601),('max_total_pilot_wall_seconds',1801),
             ('max_host_working_memory_bytes',2**31+1),('max_device_memory_bytes',2**33+1),
             ('max_device_memory_fraction',0.51),('max_output_bytes',2**29+1)]]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'
            for key,value in changes:
                with self.subTest(key=key,value=value):
                    updated=copy.deepcopy(raw);updated[key]=value;path.write_text(json.dumps(updated))
                    with self.assertRaises(ValueError):load(path)


if __name__=='__main__':unittest.main()
