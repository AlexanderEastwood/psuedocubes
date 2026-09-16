# Pseudocubes: CRT windows, residue bitsets and GPU/CPU verification

Implementation and reporting repository for the **p=619 pseudocube search**. The repository name retains the requested spelling, `psuedocubes`.

The primary search of **10^27 < x ≤ 3×10^27** completed on **September 16, 2026 at 05:12 Pacific**. The full-coverage structural audit passed, and the sole reported qualifying noncube is **2082301388064549408087113327**, independently verified for p=619. Independent arithmetic replay and the historical lower-bound assurance remain outstanding; no certified global minimum or exact OEIS term is claimed. See the [completion report](reports/completion-2026-09-16.html), [current reporting snapshot](results/current.json), [method and benchmarks](reports/method.html), and [mathematical contract](docs/mathematics.md).

Four RTX 5080s measured approximately **0.93 trillion focused candidate positions/second**, including durable checkpoint commits, in the September 14, 2026 deployment sample: about **30×** the preceding complete pipeline. These are logical candidate positions, including those excluded in bulk by bitsets, not individual wide multiplications or active CUDA-core counts. The initial remaining-range estimate was about 35 hours; it is a timestamped projection. [Machine-readable benchmarks](results/benchmarks-2026-09-14.json) distinguish compute-only and durable measurements.

## Approach

The canonical representation is `x = uB - vA`, with `0 ≤ v < B`. Coefficient-adjusted CRT wheels generate bounded focus windows without materializing the full residue sets. Each GPU shares a focus window across 64 adjacent subtiles and intersects residue bitsets for the first four residual filters before constructing wide integers. Later filters use exact arithmetic; CPUs check all emitted values and independently verify each noncube witness. Disjoint SQLite leases preserve ownership, recovery history and durable coverage.

The formulation and focus split follow [Jonathan P. Sorenson, *Sieving for pseudosquares and pseudocubes in parallel using doubly-focused enumeration and wheel datastructures*, arXiv:1001.3316v2](https://arxiv.org/abs/1001.3316v2). That source reports the historical bound `L619,3 > 10^27`; this project has not independently reproduced the earlier search. Our window reuse, GPU residue bitsets, compact checkpoints and measurements are documented separately.

## CPU reproduction

Python 3.12 and a C/C++ compiler are required. GPU tests additionally need an NVIDIA GPU and a CUDA runtime compatible with the pinned CuPy package.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_source.py
python tools/check_cpu.py
python review/verify_cube.py 674441580981249129037406633 --p 613
```

The 33 inherited tests cover complete small sets, endpoint/carry arithmetic, coefficient-adjusted residues, output overflow, disjoint leases, stale tokens, interrupted transactions and recovery. Additional publication tests ensure that partial coverage and candidate validity cannot become unsupported exact-term claims. GitHub Actions runs CPU tests and type diagnostics; it does **not** claim GPU validation.

`python tools/check_cpu.py --full-cube-count` also reruns both independent counts of the 38,322,775 eligible genuine cubes in the target interval. That is a useful control, not a proof that every noncube was enumerated.

## Bounded GPU reproduction

Use a device available to you. The portable entry point below does not connect to the production fleet or add production coverage:

```sh
python -m pip install -r requirements-gpu.txt
CUDA_VISIBLE_DEVICES=0 python tools/reproduce.py \
  --first 0 --batches 1 --compare-baseline --output artifacts/reproduction-1.json
```

Each batch spans 6.4 billion canonical v positions, internally divided into 64 subtiles of 100 million v. `--batches 256` tests one full production-sized lease. Comparison includes complete outputs, all filter populations and the deterministic execution transcript. A bounded 180-second deadline and 8 GiB/half-device memory limit apply. This command never launches a background service.

The copied historical configuration files and helper functions retain their original provenance bindings. They are regression fixtures, not portable deployment configuration. Use `tools/reproduce.py` on another machine. The SHA-256 inventory in [provenance/source.json](provenance/source.json) identifies the files copied unchanged from production build `9bdb452a5485eb3b3f40de928fa114e46b71f7cec3950e95df8734470c692530`. Public adapters are separate from that frozen build.

## Reports, completion and OEIS

`python tools/render_report.py --snapshot results/current.json --output reports/current.html` produces a timestamped HTML report and an OEIS preparation JSON. The input snapshot contains only public search/evidence fields, not access credentials or host configuration. [Reporting and OEIS workflow](docs/reporting.md) distinguishes candidate validity, interval exhaustion, historical bounds and exact terms. The appropriate pseudocube OEIS entry and indexing still need confirmation; **A002189 is the pseudosquare sequence and is not a target for this run**.

The completion ledger and historical receipts are frozen; the structural audit and witness verification passed. The [completion evidence](results/completion-2026-09-16/) records source/snapshot hashes and the exact match to two independent CPU cube-count controls. An [OEIS witness/upper-bound draft](oeis/draft.html) is prepared but has not been submitted; its destination and public evidence link still need confirmation. No b-file is fabricated from a bound or an unproved minimum. Full receipt archives remain separate from Git and can be published as versioned release assets after their provenance and public contents are reviewed.

## Source layout

| Path | Purpose |
|---|---|
| `pseudocube/`, `kernels/` | CRT enumeration, exact arithmetic, bitset matcher and durable ledger |
| `review/` | Independent witness verifier and two cube-count implementations |
| `tests/` | Arithmetic, recovery and publication regression tests |
| `tools/reproduce.py` | Portable, bounded GPU comparison |
| `tools/render_report.py` | HTML and OEIS preparation from a reporting snapshot |
| `provenance/`, `results/` | Source hashes, benchmarks and timestamped status |

The historical source and references retain their own attribution. Publication of this code is separate from any claim of a new sequence term or minimality result.
