from __future__ import annotations
import copy
import unittest
from typing import Any
from pseudocube.publication import render, summarize


def snapshot() -> dict[str, Any]:
    return dict(schema_version=1, measured_utc='2026-09-14T00:00:00Z', campaign_state='searching',
        domain=dict(p=619,lo_exclusive=str(10**27),hi_inclusive=str(3*10**27),canonical_v_end='100'),
        committed_v_end='20',allocation_v_end='30',unfinished_leases=1,candidate_values=[],structural_audit=None,
        production_build='test')


class PublicationTests(unittest.TestCase):
    def test_partial_search_cannot_become_exact_term(self) -> None:
        row=snapshot();summary=summarize(row)
        self.assertFalse(summary['minimality_proved']);self.assertEqual(summary['exact_terms'],[])
        row['campaign_state']='complete'
        with self.assertRaises(ValueError):summarize(row)

    def test_complete_search_requires_separate_audit_and_retains_limits(self) -> None:
        row=snapshot();row.update(campaign_state='complete',committed_v_end='100',allocation_v_end='100',unfinished_leases=0)
        self.assertIn('audit pending',summarize(row)['label'])
        row['structural_audit']=dict(status='PASS',committed_v_end='100',full_interval_complete=True,independent_arithmetic_replay=False)
        result=summarize(row)
        self.assertIn('structural audit passed',result['label'])
        self.assertFalse(result['oeis_submission_ready']);self.assertFalse(result['minimality_proved'])
        changed=copy.deepcopy(row);changed['structural_audit']['committed_v_end']='99'
        with self.assertRaises(ValueError):summarize(changed)

    def test_candidate_validation_is_not_minimality(self) -> None:
        n=674441580981249129037406633
        row=snapshot();row.update(campaign_state='candidate-found',candidate_values=[str(n)])
        row['domain'].update(p=613,lo_exclusive='0',hi_inclusive=str(10**27))
        result=summarize(row)
        self.assertTrue(result['witnesses'][0]['valid_pseudocube']);self.assertFalse(result['minimality_proved'])
        row['domain']['p']=619
        with self.assertRaises(ValueError):summarize(row)

    def test_report_escapes_ingested_fields(self) -> None:
        row=snapshot();row['measured_utc']='<script>alert(1)</script>'
        page=render(summarize(row))
        self.assertNotIn('<script>',page);self.assertIn('&lt;script&gt;',page)


if __name__ == '__main__':
    unittest.main()
