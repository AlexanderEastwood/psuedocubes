# Reporting and OEIS preparation

The p=619 search stopped on a verified candidate on September 15, 2026 (Pacific), with partial canonical coverage. See `reports/candidate-2026-09-15.html`. Every status file must record its UTC measurement time and whether it describes live counters, a frozen structural audit, or independently replayed arithmetic. Performance snapshots are observations, not completion certificates.

## On completion or a candidate-triggered stop

1. Freeze the SQLite database using online backup and retain all committed legacy receipts, imported evidence, source manifests, build-transition certificates and ownership records. Never replace or reset the production ledger.
2. Audit disjoint ownership, contiguous committed coverage, receipt hashes, source/configuration binding, totals, candidate records and unfinished leases. A process exit alone is not completion. A candidate-triggered stop can leave the rest of the interval unexplored.
3. Verify each noncube witness with `review/verify_cube.py`. Record the complete integer, cutoff, failing-prime list, cube-root test and source hash. Candidate validity provides an upper bound, not a minimum.
4. If the complete interval has no noncube output, report the tested interval and evidence level. Extending `L619,3 > 10^27` to a larger global lower bound additionally relies on the separately attributed historical computation. Independent arithmetic replay remains a separate assurance step.
5. Publish a dated report and versioned machine-readable certificate. Preserve previous snapshots and benchmarks. Add source archives/checksums and suitably reviewed reproduction evidence as release assets.
6. Confirm the actual pseudocube OEIS entry, definition, cutoff-to-index mapping and current editorial state. A002189 concerns pseudosquares. No matching pseudocube entry has been established by this repository's initial source lookup; this does not establish that none exists.
7. Prepare a comment/link update for a justified bound or witness. Populate an exact-term b-file only when each indexed term has the required minimality evidence and the sequence mapping has been independently checked. Do not put an inequality or an upper-bound candidate into a table of exact minima.

The [OEIS contributor style sheet](https://oeis.org/wiki/Style_Sheet) describes entry fields and indexing; the [b-file documentation](https://oeis.org/wiki/B-files) describes term-table files. The submission package should cite the exact source version and stable repository/release URL. Final editorial submission is a distinct action after the mathematical claim is ready for review.

## Public reporting snapshot

`results/current.json` records `schema_version`, `measured_utc`, `campaign_state`, the fixed p/interval/CRT domain, `committed_v_end`, `allocation_v_end`, `unfinished_leases`, `candidate_values`, a `structural_audit` summary (or null), and performance metrics. Counts and large integers use decimal strings.

```sh
python tools/render_report.py --snapshot results/current.json --output reports/current.html
```

The renderer verifies candidate values independently, checks domain bounds, and refuses inconsistent completion claims. It produces an adjacent `.json` record and `.oeis.json` preparation record. These records explicitly preserve the distinction between a structural audit and independent arithmetic replay. The tool deliberately generates no exact b-file automatically.

## Relation to filter-integrated jump tables

The first residual filters are already folded into a bitset that skips innermost candidate positions before wide reconstruction. This is compatible in principle with a jump table that advances directly between accepted positions. The current GPU implementation does not incorporate a monotonically ordered, incumbent-aware early-abort jump table: canonical CRT coordinates and modular wraparound require a separate bound proof. The first comparison should preserve exact outputs, wrap/carry behavior, scoped coverage and checkpoint identities before measuring end-to-end gains. This is a research direction, not a deployed feature or a measured speedup.
