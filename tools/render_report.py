"""Render public status and an OEIS preparation record from an evidence snapshot."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pseudocube.accounting import durable
from pseudocube.publication import render, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != '.html':
        raise ValueError('report output must be HTML')
    summary = summarize(json.loads(args.snapshot.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(summary))
    durable(args.output.with_suffix('.json'), summary)
    durable(args.output.with_suffix('.oeis.json'), dict(schema_version=1, status='draft-awaiting-entry-and-claim-review',
        sequence_id=None, index_mapping_verified=False, exact_terms=[], submission_sent=False,
        evidence_claim=summary['claim'], verified_witnesses=summary['witnesses'],
        minimality_proved=False, source_build=summary['production_build']))
    print(json.dumps(dict(status='PASS', report=str(args.output), claim=summary['label'], exact_terms=0)))


if __name__ == '__main__':
    main()
