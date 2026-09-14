from __future__ import annotations
import ctypes
from itertools import product
import json
from math import gcd, prod
from pathlib import Path
import random
import subprocess
import unittest

from pseudocube.focus import A619, B619, Config, Tile, enumerate_cpu, targeted_coordinates
from pseudocube.oracle import icbrt, is_perfect_cube, local3, primes_upto, qualifies3
from pseudocube.tables import allowed, packed, sieve_primes

ROOT = Path(__file__).resolve().parents[1]
N613 = 674441580981249129037406633


def small_config(p: int, lo: int = 0, hi: int = 100000) -> Config:
    return Config(p, lo, hi, 14 * (19 if p >= 19 else 1), 9 * (13 if p >= 13 else 1) * (31 if p >= 31 else 1))


class ArithmeticTests(unittest.TestCase):
    def test_roots_and_invalid_inputs(self) -> None:
        self.assertEqual(icbrt(3 * 10**27), 1442249570)
        for root in [0,1,2,3,2**21,2**32-1,2**42,10**27]:
            n=root**3
            self.assertEqual(icbrt(n), root)
            self.assertTrue(is_perfect_cube(n))
            if root:
                self.assertEqual(icbrt(n-1),root-1)
            self.assertEqual(icbrt(n+1), root if root else 1)
        for n in (-1, 1.5, True):
            with self.assertRaises(ValueError):icbrt(n)  # type: ignore[arg-type]
        for p in (2,3,5,9,15,25,-7):
            with self.assertRaises(ValueError):local3(1,p)
        self.assertTrue(local3(1,619));self.assertFalse(qualifies3(1,619))
        for n in (-1,0):self.assertFalse(local3(n,619))

    def test_tables_exhaustively(self) -> None:
        primes=sieve_primes(619)
        self.assertEqual(primes,list(primes_upto(619)))
        cubic=[q for q in primes if q%3==1]
        self.assertEqual((len(primes),len(cubic),sum(q%3==2 for q in primes)),(114,54,59))
        self.assertEqual(sum(cubic),15744)
        self.assertEqual(sum(4*len(packed(q)) for q in cubic),2076)
        for q in cubic:
            residues=allowed(q);self.assertEqual(len(residues),(q-1)//3);self.assertNotIn(0,residues)
            for a in range(q):
                self.assertEqual(a in residues,pow(a,(q-1)//3,q)==1)
                self.assertEqual(bool(packed(q)[a//32] & 1<<(a%32)),a in residues)
        # The oracle exposes the failure modes of deliberately corrupted tables.
        self.assertFalse(pow(0,10,31)==1)
        self.assertNotEqual(set(allowed(31))|{0},set(allowed(31)))

    def test_adjustment_and_nondivisibility(self) -> None:
        config=Config(619,10**27,3*10**27,A619,B619)
        self.assertEqual((A619%31,B619%31,pow(21,10,31)),(0,21,5))
        for focus,coefficient in zip(config.focuses(),(B619,-A619)):
            for q,residues in focus.rings:
                for a in range(q):self.assertEqual(a in residues,(coefficient*a)%q in allowed(q))
        adjusted=dict(config.focuses()[0].rings)[31]
        self.assertNotEqual(set(adjusted),set(allowed(31)))
        for q in (5,11,617):
            n=q**3
            self.assertIn(n%9,(1,8))
            self.assertTrue(all(pow(n,(r-1)//3,r)==1 for r in primes_upto(619) if r%3==1))
            self.assertFalse(local3(n,619))
            self.assertEqual(config.coverage()[str(q)],'residual')
        self.assertEqual(config.coverage()['3'],'mod9')
        with self.assertRaises(ValueError):Config(7,0,100000,14,9*13)
        with self.assertRaises(ValueError):Config(7,0,100000,3*7,3)

    def test_complete_small_sets_and_partitions(self) -> None:
        for p in (7,13,19,31):
            config=small_config(p)
            expected={n for n in range(1,100001) if local3(n,p)}
            actual=list(enumerate_cpu(config,Tile(0,config.B)))
            self.assertEqual(len(actual),len(set(actual)));self.assertEqual(set(actual),expected)
            pieces=[]
            step=max(1,config.B//7)
            for first in range(0,config.B,step):
                pieces.extend(enumerate_cpu(config,Tile(first,min(config.B,first+step))))
            self.assertEqual(len(pieces),len(set(pieces)));self.assertEqual(set(pieces),expected)
            a,b=Tile(0,config.B).split()
            self.assertEqual(set(enumerate_cpu(config,a))|set(enumerate_cpu(config,b)),expected)
            self.assertEqual({n for n in actual if is_perfect_cube(n)},
                             {r**3 for r in range(1,icbrt(100000)+1) if gcd(r,prod(primes_upto(p)))==1})

    def test_canonical_boundaries_and_normalized_residues(self) -> None:
        randomizer=random.Random(619003)
        for config in (small_config(31),Config(619,10**27,3*10**27,A619,B619)):
            for n in [config.lo,config.lo+1,config.hi,config.hi+1,*[randomizer.randrange(config.lo,config.hi+1) for _ in range(300)]]:
                u,v=config.canonical(n)
                self.assertEqual(u*config.B-v*config.A,n);self.assertTrue(0<=v<config.B)
                low,high=config.u_interval(v)
                self.assertEqual(low<=u<high,config.lo<n<=config.hi)
                for q in (5,7,9,31,617,619):
                    self.assertEqual((u*(config.B%q)-v*(config.A%q))%q,n%q)
                envelope=Tile(v,min(v+2,config.B)).envelope(config)
                if config.lo<n<=config.hi:self.assertTrue(int(envelope['x_min'])<=n<=int(envelope['x_max']))

    def test_targeted_known_value_and_cube_emission(self) -> None:
        self.assertTrue(qualifies3(N613,613));self.assertFalse(local3(N613,619));self.assertEqual(pow(N613,206,619),366)
        for p,n in [(613,N613),(619,1000000007**3),(619,1000000009**3)]:
            self.assertTrue(local3(n,p))
            config=Config(p,max(0,n-100),min(3*10**27,n+100),A619,B619)
            us,vs=targeted_coordinates(config,n)
            emitted={u*config.B-v*config.A for u,v in product(us,vs)
                     if config.lo<u*config.B-v*config.A<=config.hi and local3(u*config.B-v*config.A,p)}
            self.assertIn(n,emitted)
            self.assertEqual(n%9,1 if n==1000000009**3 else 8 if n==1000000007**3 else n%9)
        # Endpoints are recovered/excluded by enumeration, not only by the predicate.
        for n in (1,125,343,1001):
            if not local3(n,7):continue
            self.assertIn(n,list(enumerate_cpu(small_config(7,0,n),Tile(0,9))))
            self.assertNotIn(n,list(enumerate_cpu(small_config(7,n,n+100),Tile(0,9))))

    def test_compiled_wide(self) -> None:
        library=ctypes.CDLL(str(ROOT/'build/wide.so'))
        function=library.products
        function.argtypes=[ctypes.c_uint64]*4+[ctypes.POINTER(ctypes.c_uint64)]*2
        function.restype=ctypes.c_int
        rand=random.Random(619003)
        cases=[(0,0,0,0),(2**64-1,2**64-1,0,0),(0,0,2**64-1,2**64-1),
               (2**32,2**32,1,1),(2**64-1,1,0,0)]
        cases += [tuple(rand.getrandbits(64) for _ in range(4)) for _ in range(2000)]
        for a,b,c,d in cases:
            low,high=ctypes.c_uint64(),ctypes.c_uint64()
            borrow=function(a,b,c,d,ctypes.byref(low),ctypes.byref(high))
            self.assertEqual(borrow,int(a*b<c*d));self.assertEqual(low.value+(high.value<<64),(a*b-c*d)%(1<<128))

    def test_cube_counters_small_sets(self) -> None:
        for first,last,p in [(1,1000,7),(5,31,31),(30,10000,619),(617,617,619)]:
            expected=[str(n) for n in range(first,last+1) if gcd(n,prod(primes_upto(p)))==1]
            for name in ('cube_count','cube_count_independent'):
                row=json.loads(subprocess.check_output([str(ROOT/'build'/name),str(first),str(last),str(p),'--list'],text=True))
                self.assertEqual(row['roots'],expected);self.assertEqual(int(row['count']),len(expected))


if __name__=='__main__':unittest.main()
