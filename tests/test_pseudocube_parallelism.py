from __future__ import annotations
import unittest
from pseudocube.gpu import launch_plan


class ParallelismTests(unittest.TestCase):
    def test_striped_rows_partition_every_candidate_once(self) -> None:
        # Nonzero lower-bound positions and tails smaller than a warp/stripe.
        for nu in (0,1,31,32,33,63,255,256,257,6001):
            for nv in (0,1,7,8,9,86,289,962):
                for target in (1,2,4):
                    blocks,stripes=launch_plan(nu,nv,84,'warp',target)
                    self.assertEqual(bool(blocks),bool(nu and nv))
                    if not blocks:continue
                    self.assertGreaterEqual(blocks*8,nv*stripes)
                    self.assertLess(blocks*8-nv*stripes,8)
                    for first in (0,nu//3,nu):
                        emitted=[i for stripe in range(stripes) for lane in range(32)
                                 for i in range(first+stripe*32+lane,nu,stripes*32)]
                        self.assertEqual(sorted(emitted),list(range(first,nu)))

    def test_serial_and_invalid_launches(self) -> None:
        self.assertEqual(launch_plan(6001,86,84,'serial'),(1,1))
        self.assertEqual(launch_plan(6001,962,84,'serial'),(4,1))
        for layout,target in [('unknown',2),('warp',0),('warp',True),('warp',3)]:
            with self.assertRaises(ValueError):launch_plan(10,10,84,layout,target)


if __name__=='__main__':unittest.main()
