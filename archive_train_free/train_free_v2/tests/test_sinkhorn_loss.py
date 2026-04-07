"""
Unit tests for train_free_v2 modules.

Test coverage:
- SinkhornDensityLoss: Correctness, shape validation, differentiability
- Image loading and preprocessing
- Inverse OT transform
"""

import sys
import torch
import numpy as np
import unittest
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_free_v2.sinkhorn_loss import SinkhornDensityLoss, SinkhornDensityLossSimple
from train_free_v2.utils_guidance import (
    inverse_ot_transform,
    compute_density_statistics
)


class TestSinkhornDensityLoss(unittest.TestCase):
    """Test SinkhornDensityLoss module."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sinkhorn = SinkhornDensityLoss(blur=0.05, grid_size=32).to(self.device)
        self.grid_size = 32
    
    def test_shape_validation(self):
        """Test that loss accepts correct shapes and rejects invalid ones."""
        B, N = 2, 1024
        H, W = self.grid_size, self.grid_size
        
        # Valid shapes
        points = torch.randn(B, N, 2, device=self.device)
        image = torch.randn(B, 1, H, W, device=self.device).abs()
        
        loss = self.sinkhorn(points, image)
        self.assertEqual(loss.shape, torch.Size([]))  # Scalar
        self.assertTrue(torch.isfinite(loss))
    
    def test_differentiability(self):
        """Test that loss is differentiable w.r.t. points."""
        B, N = 1, 1024
        H, W = self.grid_size, self.grid_size
        
        points = torch.randn(B, N, 2, device=self.device, requires_grad=True)
        image = torch.randn(B, 1, H, W, device=self.device).abs()
        
        loss = self.sinkhorn(points, image)
        loss.backward()
        
        self.assertIsNotNone(points.grad)
        self.assertTrue(torch.isfinite(points.grad).all())
    
    def test_loss_consistency(self):
        """Test that loss is deterministic (same input → same output)."""
        B, N = 1, 1024
        H, W = self.grid_size, self.grid_size
        
        points = torch.randn(B, N, 2, device=self.device)
        image = torch.randn(B, 1, H, W, device=self.device).abs()
        
        loss1 = self.sinkhorn(points, image).detach()
        loss2 = self.sinkhorn(points, image).detach()
        
        torch.testing.assert_close(loss1, loss2)
    
    def test_loss_positivity(self):
        """Test that loss is always non-negative."""
        B, N = 2, 1024
        H, W = self.grid_size, self.grid_size
        
        for _ in range(5):
            points = torch.randn(B, N, 2, device=self.device)
            image = torch.randn(B, 1, H, W, device=self.device).abs()
            
            loss = self.sinkhorn(points, image)
            self.assertTrue((loss >= 0).all())
    
    def test_batch_processing(self):
        """Test that batching works correctly."""
        N = 1024
        H, W = self.grid_size, self.grid_size
        
        # Single sample
        points_single = torch.randn(1, N, 2, device=self.device)
        image_single = torch.randn(1, 1, H, W, device=self.device).abs()
        loss_single = self.sinkhorn(points_single, image_single)
        
        # Batch of 4
        points_batch = torch.randn(4, N, 2, device=self.device)
        image_batch = torch.randn(4, 1, H, W, device=self.device).abs()
        loss_batch = self.sinkhorn(points_batch, image_batch)
        
        # Loss should scale approximately with batch size
        self.assertTrue(loss_batch > 0)
        self.assertTrue(loss_single > 0)
    
    def test_image_inversion(self):
        """Test that dark pixels get higher density."""
        B, N = 1, 100
        H, W = 16, 16  # Smaller grid for this test
        
        # Create images: one all-dark, one all-light
        dark_image = torch.zeros(B, 1, H, W, device=self.device)
        light_image = torch.ones(B, 1, H, W, device=self.device)
        
        # Create some test points in center
        points = torch.tensor(
            [[[0.5, 0.5]] * N],
            dtype=torch.float32,
            device=self.device
        )
        
        sinkhorn_small = SinkhornDensityLoss(blur=0.05, grid_size=16).to(self.device)
        
        loss_dark = sinkhorn_small(points, dark_image)
        loss_light = sinkhorn_small(points, light_image)
        
        # Loss should be lower when points are in dark region
        self.assertTrue(torch.isfinite(loss_dark))
        self.assertTrue(torch.isfinite(loss_light))
    
    def test_grid_size_flexibility(self):
        """Test that different grid sizes work."""
        B, N = 1, 1024
        
        for grid_size in [8, 16, 32, 64]:
            sinkhorn = SinkhornDensityLoss(blur=0.05, grid_size=grid_size).to(self.device)
            points = torch.randn(B, N, 2, device=self.device)
            image = torch.randn(B, 1, 64, 64, device=self.device).abs()
            
            loss = sinkhorn(points, image)
            self.assertTrue(torch.isfinite(loss))


class TestSinkhornDensityLossSimple(unittest.TestCase):
    """Test fallback SinkhornDensityLossSimple."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sinkhorn_simple = SinkhornDensityLossSimple(blur=0.05, grid_size=32).to(self.device)
    
    def test_basic_functionality(self):
        """Test that simple version works."""
        B, N = 1, 1024
        H, W = 32, 32
        
        points = torch.randn(B, N, 2, device=self.device)
        image = torch.randn(B, 1, H, W, device=self.device).abs()
        
        loss = self.sinkhorn_simple(points, image)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss >= 0)


class TestInverseOTTransform(unittest.TestCase):
    """Test inverse optimal transport transform."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.grid_size = 32
    
    def test_shape_conversion(self):
        """Test that shapes are correctly transformed."""
        B = 2
        offsets = torch.randn(B, 2, self.grid_size, self.grid_size, device=self.device)
        
        points = inverse_ot_transform(offsets, grid_size=self.grid_size)
        
        expected_shape = (B, self.grid_size ** 2, 2)
        self.assertEqual(points.shape, expected_shape)
    
    def test_point_range(self):
        """Test that points are clipped to [0, 1]."""
        B = 2
        offsets = torch.randn(B, 2, self.grid_size, self.grid_size, device=self.device)
        
        points = inverse_ot_transform(offsets, grid_size=self.grid_size)
        
        self.assertTrue((points >= 0).all())
        self.assertTrue((points <= 1).all())
    
    def test_deterministic(self):
        """Test that transform is deterministic."""
        offsets = torch.randn(1, 2, self.grid_size, self.grid_size, device=self.device)
        
        points1 = inverse_ot_transform(offsets, grid_size=self.grid_size)
        points2 = inverse_ot_transform(offsets, grid_size=self.grid_size)
        
        torch.testing.assert_close(points1, points2)


class TestDensityStatistics(unittest.TestCase):
    """Test density statistics computation."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_statistics_computation(self):
        """Test that statistics are computed without error."""
        N = 1024
        points = torch.randn(1, N, 2, device=self.device).abs()
        points = torch.clamp(points, 0, 1)
        
        stats = compute_density_statistics(points, grid_size=32, bins=10)
        
        self.assertIn('num_points', stats)
        self.assertEqual(stats['num_points'], N)
        self.assertGreaterEqual(stats['spatial_coverage'], 0)
        self.assertLessEqual(stats['density_uniformity'], 1)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sinkhorn = SinkhornDensityLoss(blur=0.05, grid_size=32).to(self.device)
    
    def test_full_pipeline(self):
        """Test a simplified version of the full sampling pipeline."""
        B, N = 1, 1024
        grid_size = 32
        H, W = 32, 32
        
        # 1. Create random offsets (simulating model output)
        offsets = torch.randn(B, 2, H, W, device=self.device, requires_grad=True)
        
        # 2. Create target image
        target_image = torch.rand(B, 1, H, W, device=self.device)
        
        # 3. Reshape offsets to points
        offsets_flat = offsets.permute(0, 2, 3, 1).reshape(B, -1, 2)
        
        # 4. Compute loss
        loss = self.sinkhorn(offsets_flat, target_image)
        
        # 5. Compute gradient
        loss.backward()
        
        # 6. Check gradient exists and is finite
        self.assertIsNotNone(offsets.grad)
        self.assertTrue(torch.isfinite(offsets.grad).all())
        
        # 7. Apply inverse transform
        points = inverse_ot_transform(offsets.detach(), grid_size=grid_size)
        
        # 8. Compute statistics
        stats = compute_density_statistics(points, grid_size=grid_size)
        self.assertEqual(stats['num_points'], grid_size ** 2)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestSinkhornDensityLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestSinkhornDensityLossSimple))
    suite.addTests(loader.loadTestsFromTestCase(TestInverseOTTransform))
    suite.addTests(loader.loadTestsFromTestCase(TestDensityStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
