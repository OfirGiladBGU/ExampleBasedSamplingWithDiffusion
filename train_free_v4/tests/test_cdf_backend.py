import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train_free_v4.backends.cdf_backend import CDFWarpBackend, apply_cdf_inverse_warp
from train_free_v4.backends.none_backend import IdentityWarpBackend


class TestIdentityWarpBackend(unittest.TestCase):
    def test_identity_backend_preserves_points(self):
        backend = IdentityWarpBackend()
        points = torch.rand(2, 16, 2)
        image = torch.rand(2, 1, 32, 32)
        warped = backend.warp(points, image)
        torch.testing.assert_close(points, warped)


class TestCDFWarpBackend(unittest.TestCase):
    def setUp(self):
        self.backend = CDFWarpBackend(
            warp_grid_size=64,
            density_mode="dark",
            density_gamma=2.0,
            eps=1e-8,
            interpolation=True,
        )

    def test_output_shape_and_bounds(self):
        points = torch.rand(1, 64, 2)
        image = torch.rand(1, 1, 32, 32)
        warped = self.backend.warp(points, image)
        self.assertEqual(warped.shape, points.shape)
        self.assertTrue((warped >= 0).all())
        self.assertTrue((warped <= 1).all())

    def test_dark_region_attracts_points(self):
        points = torch.stack(
            torch.meshgrid(
                torch.linspace(0.05, 0.95, 16),
                torch.linspace(0.05, 0.95, 16),
                indexing="xy",
            ),
            dim=-1,
        ).reshape(1, -1, 2)

        image = torch.ones(1, 1, 64, 64)
        image[:, :, 20:44, 20:44] = 0.0

        warped = self.backend.warp(points, image)
        center = torch.tensor([0.5, 0.5]).view(1, 1, 2)
        before = torch.norm(points - center, dim=-1).mean().item()
        after = torch.norm(warped - center, dim=-1).mean().item()
        self.assertLess(after, before)

    def test_apply_cdf_inverse_warp_handles_uniform_pdf(self):
        points = torch.rand(1, 32, 2)
        pdf = torch.ones(1, 32, 32)
        pdf = pdf / pdf.sum(dim=(1, 2), keepdim=True)
        warped = apply_cdf_inverse_warp(points, pdf, interpolation=True)
        self.assertEqual(warped.shape, points.shape)
        self.assertTrue(torch.isfinite(warped).all())


if __name__ == "__main__":
    unittest.main()
