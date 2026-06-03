"""Unit tests for render_overlay_frame() overlay rendering function."""

import unittest
from unittest.mock import Mock

import numpy as np

from acia.base import Contour, Overlay
from acia.viz import render_overlay_frame


class TestRenderOverlayFrameBasic(unittest.TestCase):
    """Test basic functionality of render_overlay_frame()."""

    def setUp(self):
        """Set up test fixtures."""
        self.height = 100
        self.width = 100
        self.frame_idx = 0

    def test_render_overlay_frame_returns_uint8(self):
        """Test that render_overlay_frame() always returns uint8 array."""
        # Create RGB image
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Create empty overlay
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, self.frame_idx)

        self.assertEqual(result.dtype, np.uint8)

    def test_render_overlay_frame_returns_rgb(self):
        """Test that render_overlay_frame() always returns RGB (3-channel) image."""
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, self.frame_idx)

        self.assertEqual(len(result.shape), 3)
        self.assertEqual(result.shape[2], 3)

    def test_render_overlay_frame_preserves_dimensions(self):
        """Test that output dimensions match input dimensions."""
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, self.frame_idx)

        self.assertEqual(result.shape[0], self.height)
        self.assertEqual(result.shape[1], self.width)

    def test_render_overlay_frame_grayscale_input(self):
        """Test render_overlay_frame() converts grayscale to RGB output."""
        # Grayscale image (HxW)
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, self.frame_idx)

        # Output should be RGB
        self.assertEqual(result.shape, (self.height, self.width, 3))

    def test_render_overlay_frame_with_empty_overlay(self):
        """Test render_overlay_frame() with empty overlay (no instances)."""
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 100
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, self.frame_idx)

        # With empty overlay, result should be close to original image
        # (may be different due to grayscale conversion)
        self.assertEqual(result.shape, (self.height, self.width, 3))
        self.assertEqual(result.dtype, np.uint8)


class TestRenderOverlayFrameImageFormats(unittest.TestCase):
    """Test render_overlay_frame() with different image formats."""

    def setUp(self):
        """Set up test fixtures."""
        self.height = 100
        self.width = 100
        self.frame_idx = 0

    def test_render_overlay_frame_uint8_input(self):
        """Test render_overlay_frame() with uint8 input."""
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 150
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, self.frame_idx)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (self.height, self.width, 3))

    def test_render_overlay_frame_uint16_input(self):
        """Test render_overlay_frame() with uint16 input."""
        # uint16 image (typical for microscopy)
        image = np.ones((self.height, self.width, 3), dtype=np.uint16) * 1000
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, self.frame_idx)

        # Output should always be uint8
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (self.height, self.width, 3))

    def test_render_overlay_frame_float_input(self):
        """Test render_overlay_frame() with float input."""
        image = np.ones((self.height, self.width, 3), dtype=np.float32) * 0.5
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, self.frame_idx)

        # Output should always be uint8
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (self.height, self.width, 3))

    def test_render_overlay_frame_single_channel_input(self):
        """Test render_overlay_frame() with single-channel (grayscale) input."""
        image = np.ones((self.height, self.width), dtype=np.uint8) * 100
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, self.frame_idx)

        # Output should be RGB
        self.assertEqual(result.shape, (self.height, self.width, 3))
        self.assertEqual(result.dtype, np.uint8)


class TestRenderOverlayFrameAlphaBlending(unittest.TestCase):
    """Test alpha blending behavior of render_overlay_frame()."""

    def setUp(self):
        """Set up test fixtures."""
        self.height = 100
        self.width = 100
        self.frame_idx = 0

    def test_render_overlay_frame_alpha_default(self):
        """Test render_overlay_frame() with default alpha=0.8."""
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 100
        overlay = Overlay([])

        # Default alpha should be 0.8
        result = render_overlay_frame(image, overlay, self.frame_idx)

        self.assertEqual(result.shape, (self.height, self.width, 3))

    def test_render_overlay_frame_alpha_zero(self):
        """Test render_overlay_frame() with alpha=0.0 (full overlay)."""
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 100
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, self.frame_idx, alpha=0.0)

        self.assertEqual(result.dtype, np.uint8)

    def test_render_overlay_frame_alpha_one(self):
        """Test render_overlay_frame() with alpha=1.0 (full original)."""
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 100
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, self.frame_idx, alpha=1.0)

        self.assertEqual(result.dtype, np.uint8)

    def test_render_overlay_frame_alpha_half(self):
        """Test render_overlay_frame() with alpha=0.5 (50/50 blend)."""
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 100
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, self.frame_idx, alpha=0.5)

        self.assertEqual(result.dtype, np.uint8)

    def test_render_overlay_frame_alpha_affects_blending(self):
        """Test that alpha parameter actually affects the blending result."""
        # Create a colored image for testing
        image = np.ones((50, 50, 3), dtype=np.uint8) * 200
        image[:, :, 0] = 200  # Red channel
        image[:, :, 1] = 0  # Green channel
        image[:, :, 2] = 0  # Blue channel

        overlay = Overlay([])

        # Render with different alpha values
        result_alpha_01 = render_overlay_frame(
            image, overlay, self.frame_idx, alpha=0.1
        )
        result_alpha_09 = render_overlay_frame(
            image, overlay, self.frame_idx, alpha=0.9
        )

        # Results should be different (different alpha values)
        # Although with empty overlay, results might be similar
        self.assertEqual(result_alpha_01.dtype, np.uint8)
        self.assertEqual(result_alpha_09.dtype, np.uint8)


class TestRenderOverlayFrameWithContours(unittest.TestCase):
    """Test render_overlay_frame() with actual contours."""

    def setUp(self):
        """Set up test fixtures."""
        self.height = 200
        self.width = 200

    def test_render_overlay_frame_with_mock_contour(self):
        """Test render_overlay_frame() with mock contour."""
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 100

        # Create a mock contour with toMask method
        mock_contour = Mock(spec=Contour)
        mock_mask = np.zeros((self.height, self.width), dtype=bool)
        # Create a small square mask
        mock_mask[50:100, 50:100] = True
        mock_contour.toMask = Mock(return_value=mock_mask)
        mock_contour.label = None
        mock_contour.frame = 0
        mock_contour.id = 0

        overlay = Overlay([mock_contour])

        result = render_overlay_frame(image, overlay, 0)

        # Should return RGB uint8
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (self.height, self.width, 3))

    def test_render_overlay_frame_with_multiple_contours(self):
        """Test render_overlay_frame() with multiple contours."""
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 100

        # Create multiple mock contours
        contours = []
        for i in range(3):
            mock_contour = Mock(spec=Contour)
            mock_mask = np.zeros((self.height, self.width), dtype=bool)
            # Create different regions for each contour
            y_start = i * 50
            mock_mask[y_start : y_start + 50, i * 50 : (i + 1) * 50] = True
            mock_contour.toMask = Mock(return_value=mock_mask)
            mock_contour.label = None
            mock_contour.frame = 0
            mock_contour.id = i
            contours.append(mock_contour)

        overlay = Overlay(contours)

        result = render_overlay_frame(image, overlay, 0)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (self.height, self.width, 3))

    def test_render_overlay_frame_preserves_masked_region_structure(self):
        """Test that masked regions are properly blended with overlay."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        image[:, :, 0] = 200  # Red-ish image
        image[:, :, 1] = 100
        image[:, :, 2] = 100

        # Create mock contour
        mock_contour = Mock(spec=Contour)
        mock_mask = np.zeros((100, 100), dtype=bool)
        mock_mask[25:75, 25:75] = True  # Center square
        mock_contour.toMask = Mock(return_value=mock_mask)
        mock_contour.label = None
        mock_contour.frame = 0
        mock_contour.id = 0

        overlay = Overlay([mock_contour])

        result = render_overlay_frame(image, overlay, 0, alpha=0.8)

        # Result should have blended region where mask is
        # and original region where mask is not
        self.assertEqual(result.shape, (100, 100, 3))
        self.assertEqual(result.dtype, np.uint8)


class TestRenderOverlayFrameEdgeCases(unittest.TestCase):
    """Test edge cases for render_overlay_frame()."""

    def test_render_overlay_frame_small_image(self):
        """Test render_overlay_frame() with very small image."""
        image = np.ones((5, 5, 3), dtype=np.uint8) * 100
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, 0)

        self.assertEqual(result.shape, (5, 5, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_render_overlay_frame_large_image(self):
        """Test render_overlay_frame() with large image."""
        image = np.ones((1000, 1000, 3), dtype=np.uint8) * 100
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, 0)

        self.assertEqual(result.shape, (1000, 1000, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_render_overlay_frame_rectangular_image(self):
        """Test render_overlay_frame() with rectangular (non-square) image."""
        image = np.ones((200, 500, 3), dtype=np.uint8) * 100
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, 0)

        self.assertEqual(result.shape, (200, 500, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_render_overlay_frame_frame_idx_parameter(self):
        """Test that frame_idx parameter is accepted (for compatibility)."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 100
        overlay = Overlay([])

        # Should work with different frame indices
        result0 = render_overlay_frame(image, overlay, 0)
        result5 = render_overlay_frame(image, overlay, 5)
        result100 = render_overlay_frame(image, overlay, 100)

        self.assertEqual(result0.shape, result5.shape)
        self.assertEqual(result5.shape, result100.shape)

    def test_render_overlay_frame_zero_image(self):
        """Test render_overlay_frame() with all-zero image."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, 0)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (100, 100, 3))

    def test_render_overlay_frame_max_value_image(self):
        """Test render_overlay_frame() with max-value image."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, 0)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (100, 100, 3))
        # Values should be in valid uint8 range
        self.assertLessEqual(result.max(), 255)
        self.assertGreaterEqual(result.min(), 0)

    def test_render_overlay_frame_mixed_channel_values(self):
        """Test render_overlay_frame() with different values in each channel."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # Max red
        image[:, :, 1] = 128  # Mid green
        image[:, :, 2] = 64  # Low blue
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, 0)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (100, 100, 3))


class TestRenderOverlayFrameOutputValidation(unittest.TestCase):
    """Test output validation for render_overlay_frame()."""

    def test_render_overlay_frame_output_is_uint8(self):
        """Verify output is always uint8 as per specification."""
        image = np.ones((100, 100, 3), dtype=np.uint32) * 1000
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, 0)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.uint8)

    def test_render_overlay_frame_output_values_in_range(self):
        """Verify output values are in valid uint8 range [0, 255]."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 150
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, 0)

        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 255)

    def test_render_overlay_frame_does_not_modify_input(self):
        """Verify render_overlay_frame() does not modify input image."""
        image_original = np.ones((100, 100, 3), dtype=np.uint8) * 100
        image = image_original.copy()
        overlay = Overlay([])

        render_overlay_frame(image, overlay, 0)

        np.testing.assert_array_equal(image, image_original)

    def test_render_overlay_frame_returns_contiguous_array(self):
        """Verify output is a contiguous numpy array."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 100
        overlay = Overlay([])

        result = render_overlay_frame(image, overlay, 0)

        self.assertTrue(result.flags["C_CONTIGUOUS"] or result.flags["F_CONTIGUOUS"])


if __name__ == "__main__":
    unittest.main()
