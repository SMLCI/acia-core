"""Test cases for extract_segmentation_stacks function."""

import unittest

import numpy as np

from acia.base import Contour, Instance, Overlay
from acia.segm.local import THWCSequenceSource
from acia.segm.utils import _bbox_from_mask, extract_segmentation_stacks


class TestBboxFromMask(unittest.TestCase):
    """Test the bounding box computation helper."""

    def test_basic_bbox(self):
        """Test basic bbox computation with margin."""
        # Create a mask with object at rows 20-40, cols 10-30
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 10:30] = True

        y_slice, x_slice = _bbox_from_mask(mask, margin=5)

        # With margin=5: x=[5, 35], y=[15, 45]
        self.assertEqual(x_slice, slice(5, 35))
        self.assertEqual(y_slice, slice(15, 45))

    def test_bbox_clipping_at_origin(self):
        """Test bbox clips correctly when near origin."""
        # Object near origin at rows 3-10, cols 2-10
        mask = np.zeros((100, 100), dtype=bool)
        mask[3:10, 2:10] = True

        y_slice, x_slice = _bbox_from_mask(mask, margin=10)

        # x should clip to 0 (2-10=-8 -> 0)
        # y should clip to 0 (3-10=-7 -> 0)
        self.assertEqual(x_slice.start, 0)
        self.assertEqual(y_slice.start, 0)

    def test_bbox_clipping_at_edge(self):
        """Test bbox clips correctly when near image edge."""
        # Object near edge at rows 90-98, cols 90-98
        mask = np.zeros((100, 100), dtype=bool)
        mask[90:98, 90:98] = True

        y_slice, x_slice = _bbox_from_mask(mask, margin=10)

        # x_end should clip to 100 (98+10=108 -> 100)
        # y_end should clip to 100 (98+10=108 -> 100)
        self.assertEqual(x_slice.stop, 100)
        self.assertEqual(y_slice.stop, 100)

    def test_bbox_zero_margin(self):
        """Test bbox with zero margin."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 10:30] = True

        y_slice, x_slice = _bbox_from_mask(mask, margin=0)

        self.assertEqual(x_slice, slice(10, 30))
        self.assertEqual(y_slice, slice(20, 40))

    def test_empty_mask_returns_none(self):
        """Test that empty mask returns None."""
        mask = np.zeros((100, 100), dtype=bool)

        result = _bbox_from_mask(mask, margin=5)

        self.assertIsNone(result)


class TestExtractSegmentationStacks(unittest.TestCase):
    """Test the main extraction function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test image stack: 5 frames, 100x100, 3 channels
        self.image_stack = np.zeros((5, 100, 100, 3), dtype=np.uint8)
        # Add some identifiable pattern
        for t in range(5):
            self.image_stack[t, :, :, 0] = t * 10  # Varying red channel
        self.source = THWCSequenceSource(self.image_stack)

    def test_single_contour_extraction(self):
        """Test extraction of a single contour."""
        contours = [
            Contour([[10, 20], [30, 20], [30, 40], [10, 40]], -1, frame=0, id=1)
        ]
        overlay = Overlay(contours)

        result = extract_segmentation_stacks(self.source, overlay, margin=5, frame=0)

        self.assertIn(1, result)
        # Check dimensions: T preserved, H/W cropped
        # bbox = (10, 20, 30, 40), margin=5 -> x=[5,35], y=[15,45]
        self.assertEqual(result[1].size_t, 5)
        self.assertEqual(result[1].size_h, 30)  # 45-15
        self.assertEqual(result[1].size_w, 30)  # 35-5
        self.assertEqual(result[1].size_c, 3)

    def test_multiple_contours_extraction(self):
        """Test extraction of multiple contours."""
        contours = [
            Contour([[10, 10], [20, 10], [20, 20], [10, 20]], -1, frame=0, id=1),
            Contour([[50, 50], [60, 50], [60, 60], [50, 60]], -1, frame=0, id=2),
        ]
        overlay = Overlay(contours)

        result = extract_segmentation_stacks(self.source, overlay, margin=5, frame=0)

        self.assertEqual(len(result), 2)
        self.assertIn(1, result)
        self.assertIn(2, result)

    def test_frame_filtering(self):
        """Test that only contours from specified frame are extracted."""
        contours = [
            Contour([[10, 10], [20, 10], [20, 20], [10, 20]], -1, frame=0, id=1),
            Contour([[30, 30], [40, 30], [40, 40], [30, 40]], -1, frame=1, id=2),
            Contour([[50, 50], [60, 50], [60, 60], [50, 60]], -1, frame=0, id=3),
        ]
        overlay = Overlay(contours)

        result = extract_segmentation_stacks(self.source, overlay, margin=5, frame=0)

        self.assertEqual(len(result), 2)
        self.assertIn(1, result)
        self.assertIn(3, result)
        self.assertNotIn(2, result)

    def test_frame_none_extracts_all(self):
        """Test that frame=None extracts all contours."""
        contours = [
            Contour([[10, 10], [20, 10], [20, 20], [10, 20]], -1, frame=0, id=1),
            Contour([[30, 30], [40, 30], [40, 40], [30, 40]], -1, frame=1, id=2),
        ]
        overlay = Overlay(contours)

        result = extract_segmentation_stacks(self.source, overlay, margin=5, frame=None)

        self.assertEqual(len(result), 2)

    def test_empty_overlay(self):
        """Test with empty overlay."""
        overlay = Overlay([])

        result = extract_segmentation_stacks(self.source, overlay, margin=5, frame=0)

        self.assertEqual(result, {})

    def test_no_contours_in_frame(self):
        """Test when no contours match the specified frame."""
        contours = [
            Contour([[10, 10], [20, 10], [20, 20], [10, 20]], -1, frame=1, id=1),
        ]
        overlay = Overlay(contours)

        result = extract_segmentation_stacks(self.source, overlay, margin=5, frame=0)

        self.assertEqual(result, {})

    def test_contour_at_image_edge(self):
        """Test contour near image edge gets clipped correctly."""
        contours = [
            Contour([[0, 0], [5, 0], [5, 5], [0, 5]], -1, frame=0, id=1),
        ]
        overlay = Overlay(contours)

        result = extract_segmentation_stacks(self.source, overlay, margin=10, frame=0)

        self.assertIn(1, result)
        # Should be clipped: x=[0, 15], y=[0, 15]
        self.assertEqual(result[1].size_h, 15)
        self.assertEqual(result[1].size_w, 15)

    def test_negative_margin_raises(self):
        """Test that negative margin raises ValueError."""
        contours = [
            Contour([[10, 10], [20, 10], [20, 20], [10, 20]], -1, frame=0, id=1)
        ]
        overlay = Overlay(contours)

        with self.assertRaises(ValueError):
            extract_segmentation_stacks(self.source, overlay, margin=-5, frame=0)

    def test_zero_margin(self):
        """Test extraction with zero margin."""
        contours = [
            Contour([[10, 20], [30, 20], [30, 40], [10, 40]], -1, frame=0, id=1)
        ]
        overlay = Overlay(contours)

        result = extract_segmentation_stacks(self.source, overlay, margin=0, frame=0)

        self.assertIn(1, result)
        # Exact bbox: x=[10,30], y=[20,40]
        self.assertEqual(result[1].size_h, 20)  # 40-20
        self.assertEqual(result[1].size_w, 20)  # 30-10

    def test_pixel_values_preserved(self):
        """Test that extracted pixel values are correct."""
        # Set a specific region with known values
        image_stack = np.zeros((5, 100, 100, 3), dtype=np.uint8)
        image_stack[:, 20:40, 10:30, :] = 255
        source = THWCSequenceSource(image_stack)

        contours = [
            Contour([[10, 20], [30, 20], [30, 40], [10, 40]], -1, frame=0, id=1)
        ]
        overlay = Overlay(contours)

        result = extract_segmentation_stacks(source, overlay, margin=0, frame=0)

        # All pixels in the cropped region should be 255
        np.testing.assert_array_equal(result[1].image_stack, 255)

    def test_with_instance_objects(self):
        """Test extraction works with Instance objects too."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 10:30] = 1
        instances = [Instance(mask, frame=0, label=1, id=1)]
        overlay = Overlay(instances)

        result = extract_segmentation_stacks(self.source, overlay, margin=5, frame=0)

        self.assertIn(1, result)


class TestExtractSegmentationStacksIntegration(unittest.TestCase):
    """Integration tests for extract_segmentation_stacks."""

    def test_realistic_workflow(self):
        """Test a realistic workflow with multiple frames and contours."""
        # Create a larger image stack
        image_stack = np.random.randint(0, 255, (10, 200, 200, 1), dtype=np.uint8)
        source = THWCSequenceSource(image_stack)

        # Create contours across multiple frames
        contours = []
        for frame in range(3):
            for i in range(5):
                x, y = 20 + i * 30, 20 + frame * 50
                coords = [[x, y], [x + 20, y], [x + 20, y + 20], [x, y + 20]]
                contours.append(Contour(coords, -1, frame=frame, id=f"f{frame}_c{i}"))

        overlay = Overlay(contours)

        # Extract for frame 1
        result = extract_segmentation_stacks(source, overlay, margin=10, frame=1)

        self.assertEqual(len(result), 5)
        for key in result:
            self.assertTrue(key.startswith("f1_"))


if __name__ == "__main__":
    unittest.main()
