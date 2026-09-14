"""Report evidence levels without promoting a witness or partial run to an exact term."""
from __future__ import annotations
import html
from typing import Any
from review.verify_cube import verify


def integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError('integer or decimal string required')
    result = int(value)
    if str(result) != str(value):
        raise ValueError('canonical decimal integer required')
    return result


def summarize(snapshot: dict[str, Any]) -> dict[str, Any]:
    if snapshot.get('schema_version') != 1:
        raise ValueError('unsupported snapshot schema')
    phase = snapshot['campaign_state']
    if phase not in ('searching', 'candidate-found', 'complete', 'paused', 'error'):
        raise ValueError('unknown campaign state')
    domain = snapshot['domain']
    p, lo, hi, end = (integer(domain[k]) for k in ('p', 'lo_exclusive', 'hi_inclusive', 'canonical_v_end'))
    cursor, allocated, unfinished = (integer(snapshot[k]) for k in ('committed_v_end', 'allocation_v_end', 'unfinished_leases'))
    if not 0 <= lo < hi or not 0 <= cursor <= allocated <= end or unfinished < 0:
        raise ValueError('invalid scope, cursor or ownership counts')
    if phase == 'complete' and (cursor != end or allocated != end or unfinished != 0):
        raise ValueError('partial coverage cannot be reported as complete')
    witnesses = []
    numbers = [integer(value) for value in snapshot['candidate_values']]
    if len(numbers) != len(set(numbers)):
        raise ValueError('duplicate candidate')
    for n in numbers:
        if not lo < n <= hi:
            raise ValueError('candidate outside search interval')
        witness = verify(n, p)
        if not witness['valid_pseudocube']:
            raise ValueError('independent candidate verification failed')
        witnesses.append(witness)
    if phase == 'candidate-found' and not witnesses:
        raise ValueError('candidate stop has no valid witness')
    audit = snapshot.get('structural_audit')
    audited_full = False
    if audit is not None:
        if audit.get('status') != 'PASS' or integer(audit['committed_v_end']) != cursor:
            raise ValueError('audit does not match this snapshot')
        audited_full = audit.get('full_interval_complete') is True
        if audited_full and (cursor != end or unfinished != 0):
            raise ValueError('audit claims full coverage of an unfinished domain')
    if witnesses:
        label = 'Verified candidate; minimality unresolved'
        claim = f"Valid p={p} pseudocube witness: {min(numbers)}. This supplies an upper bound only."
    elif phase == 'complete' and audited_full:
        label = 'Complete primary interval search; structural audit passed'
        claim = f"Primary execution reports no noncube satisfying the p={p} predicate in ({lo}, {hi}]."
    elif phase == 'complete':
        label = 'Worker reports completion; frozen coverage audit pending'
        claim = 'No interval-exhaustion or exact-term claim is released yet.'
    else:
        label = 'Search in progress' if phase == 'searching' else 'Search paused or requires attention'
        claim = 'The configured interval is not yet established as exhausted.'
    return dict(schema_version=1, measured_utc=snapshot['measured_utc'], label=label, claim=claim,
        domain=domain, coverage_percent=100*cursor/end, committed_v_end=str(cursor),
        witnesses=witnesses, structural_audit=audit, performance=snapshot.get('performance'),
        production_build=snapshot['production_build'], minimality_proved=False,
        exact_terms=[], oeis_submission_ready=False,
        caveat='Structural execution evidence and independently replayed arithmetic are distinct. Historical lower bounds retain their separate attribution.')


def render(summary: dict[str, Any]) -> str:
    e = lambda value: html.escape(str(value))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pseudocube search status</title></head>
<body style="margin:0;background:#f4f2ed;color:#26342d;font-family:Georgia,serif;line-height:1.6"><main style="max-width:680px;margin:36px auto;padding:28px;background:#fff">
<h1 style="line-height:1.2">{e(summary['label'])}</h1><p>Snapshot: {e(summary['measured_utc'])}</p>
<p style="padding:16px;background:#edf4ee;border-left:4px solid #32684e">{e(summary['claim'])}</p>
<p>Contiguous canonical domain covered: {summary['coverage_percent']:.5f}%.</p>
<p>Cutoff p={e(summary['domain']['p'])}; numerical interval ({e(summary['domain']['lo_exclusive'])}, {e(summary['domain']['hi_inclusive'])}].</p>
<p>Production build: <span style="overflow-wrap:anywhere;font-family:monospace">{e(summary['production_build'])}</span>.</p>
<p>{e(summary['caveat'])}</p><p>No new exact OEIS term is asserted by this report. The sequence entry, indexing and minimality evidence must be established separately.</p>
</main></body></html>'''
