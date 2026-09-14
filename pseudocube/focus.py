"""Coefficient-adjusted factored CRT focuses and canonical coordinate tiles."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from math import gcd, prod
from typing import Any, Iterator

from .oracle import validate_p
from .tables import allowed, definition, sieve_primes

A619 = 701856356111039402
B619 = 693110504329192503
MAX = (1 << 64) - 1


def factors(n: int, p: int) -> list[int]:
    if not 1 <= n < 1 << 63:
        raise ValueError('focus modulus must be positive and below 2^63')
    result: list[int] = []
    for q in sieve_primes(p):
        if n % q:
            continue
        power = 1
        while n % q == 0:
            n //= q
            power *= q
        if (q == 3 and power != 9) or (q != 3 and power != q):
            raise ValueError('only squarefree primes and one complete factor 9 are supported')
        result.append(power)
    if n != 1:
        raise ValueError('focus contains a prime above the active cutoff')
    return result


@dataclass(frozen=True)
class Focus:
    modulus: int
    rings: tuple[tuple[int, tuple[int, ...]], ...]

    @property
    def count(self) -> int:
        return prod(len(residues) for _, residues in self.rings)

    @property
    def terms(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(a * (self.modulus // q) * pow(self.modulus // q, -1, q) % self.modulus
                           for a in residues) for q, residues in self.rings)

    def at(self, index: int) -> int:
        if not 0 <= index < self.count:
            raise ValueError('focus index out of bounds')
        value = 0
        for terms in self.terms:
            index, digit = divmod(index, len(terms))
            value = (value + terms[digit]) % self.modulus
        return value

    def rank(self, coordinate: int) -> tuple[int, int]:
        period, value = divmod(coordinate, self.modulus)
        index, radix = 0, 1
        for q, residues in self.rings:
            index += residues.index(value % q) * radix
            radix *= len(residues)
        return period, index

    def indices(self, start: int, end: int, period: int = 0) -> Iterator[int]:
        if not 0 <= start <= end <= self.count or period < 0:
            raise ValueError('invalid focus index interval')
        for index in range(start, end):
            yield period * self.modulus + self.at(index)

    def window(self, low: int, high: int) -> Iterator[int]:
        """Bounded-memory reference stream; intended for small regression focuses."""
        if low < 0 or high < low:
            raise ValueError('invalid half-open window')
        for period in range(low // self.modulus, (high - 1) // self.modulus + 1):
            for value in self.indices(0, self.count, period):
                if low <= value < high:
                    yield value

    def halves(self) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
        terms = self.terms
        best = min(range(1 << len(terms)), key=lambda mask: max(
            prod(len(t) for i, t in enumerate(terms) if mask & (1 << i)),
            prod(len(t) for i, t in enumerate(terms) if not mask & (1 << i))))
        return (tuple(t for i, t in enumerate(terms) if best & (1 << i)),
                tuple(t for i, t in enumerate(terms) if not best & (1 << i)))


@dataclass(frozen=True)
class Config:
    p: int
    lo: int
    hi: int
    A: int
    B: int

    def __post_init__(self) -> None:
        validate_p(self.p)
        if not 0 <= self.lo < self.hi <= 3 * 10**27 or gcd(self.A, self.B) != 1:
            raise ValueError('invalid interval, arithmetic bound, or coprimality')
        factors(self.A, self.p)
        factors(self.B, self.p)
        maximum_u = (self.hi + (self.B - 1) * self.A) // self.B
        if maximum_u > MAX or maximum_u * self.B >= 1 << 128:
            raise ValueError('intermediates exceed implementation bounds')

    def focuses(self) -> tuple[Focus, Focus]:
        result = []
        for modulus, coefficient in [(self.A, self.B), (self.B, -self.A)]:
            rings = tuple((q, tuple(sorted({a * pow(coefficient, -1, q) % q for a in allowed(q)})))
                          for q in factors(modulus, self.p))
            result.append(Focus(modulus, rings))
        return result[0], result[1]

    def canonical(self, n: int) -> tuple[int, int]:
        v = (-n * pow(self.A, -1, self.B)) % self.B if self.B > 1 else 0
        return (n + v * self.A) // self.B, v

    def u_interval(self, v: int) -> tuple[int, int]:
        return (self.lo + v * self.A) // self.B + 1, (self.hi + v * self.A) // self.B + 1

    def coverage(self) -> dict[str, str]:
        af, bf = factors(self.A, self.p), factors(self.B, self.p)
        return {str(q): 'A_focus' if q in af else 'B_focus' if q in bf else 'residual'
                for q in [9, *sieve_primes(self.p)] if q != 3} | {'3': 'mod9'}

    def residuals(self) -> list[int]:
        return [int(q) for q, location in self.coverage().items() if location == 'residual']

    def manifest(self) -> dict[str, Any]:
        af, bf = self.focuses()
        maximum_u = (self.hi + (self.B - 1) * self.A) // self.B
        return dict(schema_version=1, search_type='pseudocube', p=self.p, lo_exclusive=str(self.lo),
                    hi_inclusive=str(self.hi), A=str(self.A), B=str(self.B), canonical='v-in-[0,B)',
                    coverage=self.coverage(), table_sha256=definition(self.p)['sha256'],
                    focus_counts=[str(af.count), str(bf.count)],
                    arithmetic_bounds=dict(max_u=str(maximum_u), max_v=str(self.B - 1),
                        max_uB=str(maximum_u * self.B), max_vA=str((self.B - 1) * self.A),
                        x_bits=self.hi.bit_length(), product_bits=(maximum_u * self.B).bit_length()))


@dataclass(frozen=True)
class Tile:
    first: int
    end: int

    def validate(self, config: Config) -> None:
        if not 0 <= self.first < self.end <= config.B:
            raise ValueError('tile must be a nonempty canonical v interval')

    def envelope(self, config: Config) -> dict[str, str]:
        self.validate(config)
        u_min = config.u_interval(self.first)[0]
        u_max = config.u_interval(self.end - 1)[1] - 1
        return dict(u_min=str(u_min), u_max=str(u_max), v_min=str(self.first), v_max=str(self.end - 1),
                    x_min=str(u_min * config.B - (self.end - 1) * config.A),
                    x_max=str(u_max * config.B - self.first * config.A))

    def split(self) -> tuple[Tile, Tile]:
        if self.end - self.first < 2:
            raise ValueError('unsplittable tile')
        middle = (self.first + self.end) // 2
        return Tile(self.first, middle), Tile(middle, self.end)

    def manifest(self, config: Config) -> dict[str, Any]:
        return dict(kind='v-window', first=str(self.first), end=str(self.end), envelope=self.envelope(config))


def enumerate_cpu(config: Config, tile: Tile) -> Iterator[int]:
    tile.validate(config)
    af, bf = config.focuses()
    tables = {q: set(allowed(q)) for q in config.residuals()}
    for v in bf.window(tile.first, tile.end):
        low, high = config.u_interval(v)
        for u in af.window(low, high):
            n = u * config.B - v * config.A
            if all(n % q in residues for q, residues in tables.items()):
                yield n


def targeted_coordinates(config: Config, n: int, radius: int = 1) -> tuple[list[int], list[int]]:
    """Select CRT index neighborhoods; emit via normal enumeration, never inject n."""
    u, v = config.canonical(n)
    af, bf = config.focuses()
    period, ui = af.rank(u)
    vp, vi = bf.rank(v)
    if vp != 0:
        raise ValueError('noncanonical v')
    return (list(af.indices(max(0, ui - radius), min(af.count, ui + radius + 1), period)),
            list(bf.indices(max(0, vi - radius), min(bf.count, vi + radius + 1))))
