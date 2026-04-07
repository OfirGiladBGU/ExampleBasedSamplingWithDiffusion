import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train_free_v3.ccvt_guidance import CCVTConfig, DifferentiableCCVTGuidance


class TestCCVTGuidance(unittest.TestCase):
    def test_barycenters_shape(self):
        cfg = CCVTConfig(grid_size=32, tau=0.01, num_steps=1, mode="ccvt")
        op = DifferentiableCCVTGuidance(cfg)
        points = torch.rand(2, 64, 2)
        image = torch.rand(2, 1, 64, 64)
        bary = op.compute_barycenters(points, image)
        self.assertEqual(bary.shape, points.shape)

    def test_refine_stays_in_bounds(self):
        cfg = CCVTConfig(grid_size=32, tau=0.01, num_steps=2, mode="lloyd")
        op = DifferentiableCCVTGuidance(cfg)
        points = torch.rand(1, 128, 2)
        image = torch.rand(1, 1, 64, 64)
        refined, bary = op.refine(points, image)
        self.assertTrue((refined >= 0).all())
        self.assertTrue((refined <= 1).all())
        self.assertTrue(torch.isfinite(refined).all())


if __name__ == "__main__":
    unittest.main()
