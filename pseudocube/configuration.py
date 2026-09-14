from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
from .focus import Config, A619, B619

LIMITS = dict(max_devices=1, max_total_gpu_allocation_wall_seconds=600,
              max_total_pilot_wall_seconds=1800, max_host_working_memory_bytes=2*1024**3,
              max_device_memory_bytes=8*1024**3, max_device_memory_fraction=0.5, max_output_bytes=512*1024**2)


def load(path: Path) -> tuple[dict[str, Any], Config]:
    raw = json.loads(path.read_text())
    expected = {'schema_version','search_type','mode','p','lo_exclusive','hi_inclusive','focus','authority','pilot_budget','deterministic_test_seed','production_verification_policy'}
    if set(raw) != expected or type(raw['schema_version']) is not int or raw['schema_version'] != 1 or raw['search_type'] != 'pseudocube' or raw['mode'] != 'pilot':
        raise ValueError('invalid pilot schema/mode; production is disabled')
    authority = dict(execute_through_stage='B', allow_production_search=False, allow_multi_gpu=False,
                     allow_range_extension=False, allow_external_communication=False)
    if raw['authority'] != authority or any(raw['authority'].get(key) is not False for key in authority if key.startswith('allow_')) or raw['production_verification_policy'] != 'UNAPPROVED':
        raise ValueError('this implementation cannot authorize production')
    budget = raw['pilot_budget']
    if set(budget) != set(LIMITS) | {'device_uuid'}:
        raise ValueError('unknown/missing budget fields')
    for key, limit in LIMITS.items():
        if isinstance(budget[key], bool) or not isinstance(budget[key], (float, int)) or not 0 < budget[key] <= limit:
            raise ValueError('invalid or expanded resource budget: ' + key)
    if type(budget['max_devices']) is not int or budget['max_devices'] != 1:
        raise ValueError('exactly one device required')
    device = budget['device_uuid']
    if device is not None and (not isinstance(device, str) or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',device)):
        raise ValueError('invalid single-device UUID')
    if type(raw['p']) is not int or raw['p'] != 619 or raw['lo_exclusive'] != str(10**27) or raw['hi_inclusive'] != str(3*10**27):
        raise ValueError('pilot range/cutoff changed; derive a separate reviewed configuration')
    if raw['focus'] != {'A':str(A619), 'B':str(B619), 'status':'baseline_requires_validation'}:
        raise ValueError('focus configuration changed')
    if type(raw['deterministic_test_seed']) is not int or raw['deterministic_test_seed'] != 619003:
        raise ValueError('test seed changed')
    return raw, Config(raw['p'], int(raw['lo_exclusive']), int(raw['hi_inclusive']), A619, B619)
