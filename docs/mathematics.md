# Mathematical contract and coverage proof

The numerical interval is **(lo, hi]**, never [lo, hi). Coordinates and index tiles are half-open. The pilot target is p=619 and 10^27 < x <= 3*10^27. Smaller test definitions have their own configuration hashes.

## Local predicate and independent oracle

`oracle.local3` checks positivity, x mod 9 in {1,8}, coprimality to EVERY prime <=p, and x^((q-1)/3)=1 mod q for every q=1 mod 3. `qualifies3` additionally excludes exact integer cubes. The oracle uses trial-division prime generation and modular exponentiation; table generation independently uses Eratosthenes and a^3 mod q. A prime p=1 mod 3 is required by this implementation.

## Coefficient adjustment

Let A and B be coprime, x=uB-vA. For each prime-power ring q dividing A, x mod q=(B mod q)(u mod q); hence u must lie in B^-1 Cq. For a ring dividing B, v must lie in (-A)^-1 Cq. Multiplication by either inverse is a bijection. At q=9, use C9={1,8}; at a prime q=2 mod 3 use all nonzero residues. This covers every required constraint assigned to a focus. The machine-readable coverage map assigns every other prime to an explicit residual filter; 3 is covered by the complete factor 9, never two separate factors of 3.

The supplied split is the one printed in Sorenson arXiv:1001.3316v2, section 4.1:

- A=2*7*13*31*43*73*79*127*139*157*181=701856356111039402.
- B=9*19*37*61*67*97*103*109*151*163=693110504329192503.
- A has 4,212,736,819,200 allowed classes; B has 6,700,548,096,000.
- B mod 31=21 and 21^10 mod 31=5, so using raw cubic residues for u is wrong. The regression checks this explicitly.

## Canonical domain and endpoints

For each integer x there is exactly one v in [0,B) satisfying v=-x*A^-1 mod B, because A is invertible mod B. Then u=(x+vA)/B is integral. Conversely any pair with 0<=v<B reconstructs x=uB-vA, and the congruence forces that same v. Thus two canonical pairs cannot give the same x. For positive x, u is positive.

For a fixed v, lo<uB-vA<=hi is equivalent, with no rounding ambiguity, to

    floor((lo+vA)/B)+1 <= u < floor((hi+vA)/B)+1.

The GPU lower-bound search finds the first u with uB>vA+lo; it stops only when uB>vA+hi. It checks unsigned underflow before subtraction. These conventions include hi and exclude lo.

## Factored wheel and window completeness

Each ring has distinct admissible residues. Pairwise-coprime ring moduli give a bijection from mixed-radix digit tuples to allowed residues mod the focus modulus. `Focus.at` implements the CRT sum, and `Focus.rank` inverts the ring digits. Index neighborhoods used for targeted known-answer tests are actual CRT index ranges, not injected survivors.

Partition the rings into C and D. Each partial sum is reduced modulo M; summing C+D modulo M reconstructs the same full CRT sum. Each admissible full residue is represented exactly once by a tuple of ring digits. In a within-period window [a,b), the condition is either a<=C+D<b or a+M<=C+D<b+M. C,D lie in [0,M), so these two wraps exhaust the possibilities and do not overlap. For each D, two binary searches in sorted C enumerate exactly those entries. Empty halves have the one empty sum zero. Period offsets extend the bijection to coordinate windows; a window crossing a period is partitioned first. Full focus lists are never materialized: only bounded partial arrays and result buffers.

For a v tile [first,end), an inclusive u envelope is obtained from the smallest fixed-v lower bound and largest fixed-v upper bound. The matcher subsequently applies the exact numerical interval. Every eligible pair belongs to exactly one v tile. Splitting at integer middle partitions [first,end) into [first,middle) and [middle,end) without overlap or gaps. Checkpoints count a logical tile once, even if an attempt is replayed. A split parent cannot be committed or counted with its children. The pilot manifest is fixed before execution, and covers only its explicit rectangles; it does NOT certify the enclosing target interval.

For an inclusive coordinate rectangle the conservative extrema are u_min*B-v_max*A and u_max*B-v_min*A. These may be loose. A scheduler must intersect them with the numerical interval before using them to rule out a smaller hit. No production pruning is enabled in this pilot.

## Arithmetic bounds

With 0<=v<B and positive x<=hi, u<=floor((hi+(B-1)A)/B). Both coordinates fit uint64. For the baseline the products require about 119 bits, despite x requiring only 92. Config validation calculates and serializes the exact maximum u, v, uB and vA. Products, shifted bounds, and subtractions use two 64-bit limbs; multiplication uses `__umul64hi`, with explicit carry/borrow comparisons. Because A,B<2^63, focus partial additions below 2M fit uint64. Residue reconstruction uses ((high mod q)*(2^64 mod q)+(low mod q)) mod q; q<=619, so its intermediate fits uint64. Each kernel is limited to <=50,000,000 Cartesian opportunities, so per-launch uint64 counters cannot overflow.

Python integer arithmetic and independently compiled CPU __uint128_t arithmetic check deterministic extreme and random limb cases. Device native __uint128_t support and speed are not assumed.

## Cubes and independent accounting

The exact root r satisfies r^3<=n<(r+1)^3. In (lo,hi], possible cube roots are icbrt(lo)+1 through icbrt(hi). A positive r^3 satisfies all local conditions iff r is coprime to the product of all primes <=p: necessity follows from divisibility; sufficiency follows from cubic residues and r^3=±1 mod 9 for units modulo 9.

Two independently written tools count those roots: a byte-segment marking sieve with Eratosthenes primes and an odd-only bit sieve with trial-division primes. Both begin marking at ceil(segment_start/q)*q (adjusted to odd multiples in the odd sieve), including r=q. They also emit exact small root sets for comparison with brute-force gcd. The full supplied fixture is a hypothesis checked by the executed tools, not a return constant.

Cube counts, fingerprints, and candidate verification are diagnostics. Even matching the exact cube set would not by itself prove that every noncube was enumerated. A production minimum claim requires an approved independent replay/coverage policy.
