"""Deterministic cube-generated tables, independent of the oracle's pow test."""
from __future__ import annotations
import hashlib
import json
from math import isqrt
from typing import Any


def sieve_primes(p: int) -> list[int]:
    flags = bytearray(b'\1') * (p + 1)
    flags[:2] = b'\0\0'
    for q in range(2, isqrt(p) + 1):
        if flags[q]:
            flags[q*q:p+1:q] = b'\0' * ((p - q*q) // q + 1)
    return [q for q in range(2, p + 1) if flags[q]]


def allowed(q: int) -> tuple[int, ...]:
    if q == 9:
        return (1, 8)
    if q % 3 == 1:
        return tuple(sorted({a * a * a % q for a in range(1, q)}))
    return tuple(range(1, q))


def packed(q: int) -> list[int]:
    words = [0] * ((q + 31) // 32)
    for a in allowed(q):
        words[a // 32] |= 1 << (a % 32)
    return words


def definition(p: int) -> dict[str, Any]:
    tables = {str(q): list(allowed(q)) for q in [9, *sieve_primes(p)] if q != 3}
    raw = json.dumps(tables, sort_keys=True, separators=(',', ':')).encode()
    return {'tables': tables, 'sha256': hashlib.sha256(raw).hexdigest()}
