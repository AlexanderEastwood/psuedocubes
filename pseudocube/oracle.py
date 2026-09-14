"""Independent exact predicate. No focus, lookup table, or GPU imports."""
from __future__ import annotations
from functools import lru_cache
from math import isqrt


@lru_cache(maxsize=128)
def primes_upto(p: int) -> tuple[int, ...]:
    if type(p) is not int or p < 0:
        raise ValueError('prime limit must be a nonnegative integer')
    return tuple(q for q in range(2, p + 1) if all(q % d for d in range(2, isqrt(q) + 1)))


def validate_p(p: int) -> None:
    if type(p) is not int or p < 7 or p % 3 != 1 or p not in primes_upto(p):
        raise ValueError('p must be a prime congruent to 1 modulo 3')


def icbrt(n: int) -> int:
    if type(n) is not int or n < 0:
        raise ValueError('cube root requires a nonnegative integer')
    low, high = 0, 1 << ((n.bit_length() + 2) // 3)
    while low < high:
        middle = (low + high + 1) // 2
        if middle ** 3 <= n:
            low = middle
        else:
            high = middle - 1
    return low


def is_perfect_cube(n: int) -> bool:
    return n >= 0 and icbrt(n) ** 3 == n


def local3(n: int, p: int) -> bool:
    validate_p(p)
    if type(n) is not int or n <= 0 or n % 9 not in (1, 8):
        return False
    return all(n % q != 0 and (q % 3 != 1 or pow(n, (q - 1) // 3, q) == 1)
               for q in primes_upto(p))


def qualifies3(n: int, p: int) -> bool:
    return local3(n, p) and not is_perfect_cube(n)
