from __future__ import annotations
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

import numpy as np

from pseudocube.batch_experiment import tile_ranges
from pseudocube.focus import A619, B619, Config
from tools.batch_window_experiment import CPUFocus

ROOT=Path(__file__).resolve().parents[1]


class BatchExperimentTests(unittest.TestCase):
    def test_cpu_producer_exact_wrapping_multisets_and_overflow(self) -> None:
        rng=random.Random(61964)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            subprocess.run(['c++','-O2','-std=c++17','-shared','-fPIC',str(ROOT/'review/focus_window_cpu.cpp'),'-o',str(path/'focus_window.so')],check=True)
            for modulus in (7,113,701856356111039402):
                c=sorted({rng.randrange(modulus) for _ in range(30)} | {0,modulus-1})
                d=sorted({rng.randrange(modulus) for _ in range(30)} | {0,modulus-1})
                np.save(path/'T-C.npy',np.array(c,dtype=np.uint64))
                np.save(path/'T-D.npy',np.array(d,dtype=np.uint64))
                focus=CPUFocus(path,'T',modulus)
                for lo,hi in [(0,modulus),(1,modulus-1),(0,0),(modulus-1,modulus+1),(2*modulus+1,4*modulus-1)]:
                    actual=focus.window(lo,hi)
                    expected=sorted(period*modulus+(a+b)%modulus for period in range(lo//modulus,(hi-1)//modulus+1)
                                    for a in c for b in d if lo<=period*modulus+(a+b)%modulus<hi)
                    self.assertEqual(actual.tolist(),expected)
                with self.assertRaises(OverflowError):focus.window(0,modulus,cap=1)

    def test_batch_geometry_and_period_boundary(self) -> None:
        c=Config(619,10**27,3*10**27,A619,B619)
        for first,end in [(0,6400000000),(B619-6400000000,B619),(0,6400000001)]:
            width=100000001 if end-first>6400000000 else 100000000
            rows=tile_ranges(c,first,end,width)
            self.assertEqual(rows[0][0],first);self.assertEqual(rows[-1][1],end)
            self.assertTrue(all(a[1]==b[0] for a,b in zip(rows,rows[1:])))
            for lo,hi,umin,umax in rows:
                for v in (lo,hi-1):
                    a,b=c.u_interval(v)
                    self.assertLessEqual(umin,a);self.assertGreaterEqual(umax,b)
        with self.assertRaises(ValueError):tile_ranges(c,0,65,1)
        with self.assertRaises(ValueError):tile_ranges(c,0,1,0)


if __name__=='__main__':unittest.main()
