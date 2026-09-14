"""Standalone witness verifier: no imports from focus, tables, GPU or scheduler."""
from __future__ import annotations
import argparse
from math import isqrt
import json


def verify(n: int, p: int) -> dict[str,object]:
    primes=[q for q in range(2,p+1) if all(q%d for d in range(2,isqrt(q)+1))]
    if p not in primes or p%3!=1:raise ValueError('invalid prime cutoff')
    if n<0:raise ValueError('negative integer')
    low,high=0,n+1
    while high-low>1:
        middle=(low+high)//2
        if middle*middle*middle<=n:low=middle
        else:high=middle
    failures=[q for q in primes if n%q==0 or (q%3==1 and pow(n,(q-1)//3,q)!=1)]
    valid=n>0 and n%9 in (1,8) and not failures and low**3!=n
    return dict(n=str(n),p=p,cube_root_floor=str(low),perfect_cube=low**3==n,
                failing_primes=failures,valid_pseudocube=valid,claim='Candidate validity only; no minimality claim.')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('n',type=int);parser.add_argument('--p',type=int,default=619)
    args=parser.parse_args();row=verify(args.n,args.p);print(json.dumps(row,indent=2))
    if not row['valid_pseudocube']:raise SystemExit(1)


if __name__=='__main__':main()
