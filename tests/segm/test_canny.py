"""Test cases for CannySegmentationProcessor."""

import unittest

import cv2
import numpy as np
from shapely.geometry import Polygon

from acia.base import Overlay
from acia.segm.local import THWCSequenceSource
from acia.segm.processor import CannySegmentationProcessor


class TestCannyProcessorInitialization(unittest.TestCase):
    """Test processor initialization and parameter handling."""

    def test_default_initialization(self):
        """Test processor initializes with default parameters."""
        processor = CannySegmentationProcessor()

        self.assertEqual(processor.canny_low, 50)
        self.assertEqual(processor.canny_high, 150)
        self.assertEqual(processor.min_area, 50)
        self.assertEqual(processor.max_area, 100000)
        self.assertEqual(processor.blur_kernel, 5)

    def test_custom_initialization(self):
        """Test processor initializes with custom parameters."""
        processor = CannySegmentationProcessor(
            canny_low=75,
            canny_high=200,
            min_area=100,
            max_area=50000,
            blur_kernel=7,
        )

        self.assertEqual(processor.canny_low, 75)
        self.assertEqual(processor.canny_high, 200)
        self.assertEqual(processor.min_area, 100)
        self.assertEqual(processor.max_area, 50000)
        self.assertEqual(processor.blur_kernel, 7)

    def test_blur_kernel_odd_number_enforcement(self):
        """Test that blur kernel is converted to odd number."""
        processor = CannySegmentationProcessor(blur_kernel=6)

        # Even number should be converted to 7
        self.assertEqual(processor.blur_kernel, 7)
        self.assertTrue(processor.blur_kernel % 2 == 1)

    def test_blur_kernel_odd_number_preserved(self):
        """Test that odd blur kernel is preserved."""
        processor = CannySegmentationProcessor(blur_kernel=7)

        self.assertEqual(processor.blur_kernel, 7)

    def test_zero_blur_kernel_converted(self):
        """Test that blur_kernel=0 is converted to 1."""
        processor = CannySegmentationProcessor(blur_kernel=0)

        self.assertEqual(processor.blur_kernel, 1)

    def test_threshold_parameter_validation(self):
        """Test threshold parameters are stored correctly."""
        processor = CannySegmentationProcessor(canny_low=100, canny_high=300)

        self.assertGreater(processor.canny_high, processor.canny_low)

    def test_area_parameter_validation(self):
        """Test area parameters are stored correctly."""
        processor = CannySegmentationProcessor(min_area=50, max_area=10000)

        self.assertGreater(processor.max_area, processor.min_area)


class TestCannyBasicFunctionality(unittest.TestCase):
    """Test basic segmentation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = CannySegmentationProcessor()

    def test_single_frame_grayscale_input(self):
        """Test processing a single grayscale frame."""
        # Create a simple grayscale image with a white square
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image[0, 30:70, 30:70, 0] = 255

        source = THWCSequenceSource(image)
        overlay = self.processor(source)

        self.assertIsInstance(overlay, Overlay)
        # Should detect the square
        self.assertGreater(len(overlay), 0)

    def test_single_frame_color_input(self):
        """Test processing a single color (BGR) frame."""
        # Create a color image with a white square
        image = np.zeros((1, 100, 100, 3), dtype=np.uint8)
        image[0, 30:70, 30:70, :] = 255

        source = THWCSequenceSource(image)
        overlay = self.processor(source)

        self.assertIsInstance(overlay, Overlay)
        # Should detect the square after grayscale conversion
        self.assertGreater(len(overlay), 0)

    def test_multi_frame_sequence(self):
        """Test processing multiple frames."""
        # Create a sequence with 3 frames
        image = np.zeros((3, 100, 100, 1), dtype=np.uint8)
        for t in range(3):
            image[t, 30:70, 30:70, 0] = 255

        source = THWCSequenceSource(image)
        overlay = self.processor(source)

        # Check that we got contours from multiple frames
        frame_indices = [c.frame for c in overlay.contours]
        self.assertIn(0, frame_indices)
        self.assertIn(1, frame_indices)
        self.assertIn(2, frame_indices)

    def test_contours_have_correct_attributes(self):
        """Test that returned contours have correct attributes."""
        # Create a test image with a simple shape
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image[0, 40:60, 40:60, 0] = 255

        source = THWCSequenceSource(image)
        overlay = self.processor(source)

        self.assertGreater(len(overlay), 0)

        # Check first contour attributes
        for contour in overlay.contours:
            # Check coordinates shape (should be Nx2)
            self.assertEqual(len(contour.coordinates.shape), 2)
            self.assertEqual(contour.coordinates.shape[1], 2)

            # Check score is numeric
            self.assertIsInstance(contour.score, (int, float, np.number))
            self.assertGreater(contour.score, 0)

            # Check frame index
            self.assertEqual(contour.frame, 0)

            # Check id
            self.assertEqual(contour.id, -1)

    def test_contour_coordinates_are_float(self):
        """Test that contour coordinates are float32."""
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image[0, 40:60, 40:60, 0] = 255

        source = THWCSequenceSource(image)
        overlay = self.processor(source)

        for contour in overlay.contours:
            self.assertEqual(contour.coordinates.dtype, np.float32)


class TestCannyFrameProcessing(unittest.TestCase):
    """Test frame-by-frame processing."""

    def test_frame_indices_correct(self):
        """Test that frame indices are assigned correctly."""
        image = np.zeros((5, 100, 100, 1), dtype=np.uint8)
        for t in range(5):
            image[t, 30:70, 30:70, 0] = 255

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Collect all frame indices from contours
        frame_indices = {c.frame for c in overlay.contours}

        # Should have contours from all frames
        self.assertEqual(frame_indices, {0, 1, 2, 3, 4})

    def test_contours_from_different_frames_separate(self):
        """Test that contours from different frames maintain separate frame indices."""
        image = np.zeros((2, 100, 100, 1), dtype=np.uint8)
        # Frame 0: square at one location
        image[0, 20:40, 20:40, 0] = 255
        # Frame 1: square at different location
        image[1, 60:80, 60:80, 0] = 255

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Group contours by frame
        frame_0_contours = [c for c in overlay.contours if c.frame == 0]
        frame_1_contours = [c for c in overlay.contours if c.frame == 1]

        self.assertGreater(len(frame_0_contours), 0)
        self.assertGreater(len(frame_1_contours), 0)

    def test_single_frame_sequence(self):
        """Test with a single frame sequence."""
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image[0, 30:70, 30:70, 0] = 255

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # All contours should have frame=0
        for contour in overlay.contours:
            self.assertEqual(contour.frame, 0)


class TestCannyFiltering(unittest.TestCase):
    """Test contour filtering by area."""

    def test_min_area_filtering(self):
        """Test that small contours are filtered by min_area."""
        # Create image with very small noise and large shape
        image = np.zeros((1, 200, 200, 1), dtype=np.uint8)
        # Add small noise (single pixels)
        image[0, 50, 50, 0] = 255
        image[0, 51, 51, 0] = 255
        # Add larger shape
        image[0, 100:150, 100:150, 0] = 255

        # Processor with high min_area will filter out small noise
        processor = CannySegmentationProcessor(min_area=500)
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Should have only the large shape, not the noise
        self.assertGreater(len(overlay), 0)
        for contour in overlay.contours:
            self.assertGreaterEqual(contour.score, 500)

    def test_max_area_filtering(self):
        """Test that large contours are filtered by max_area."""
        # Create image with very large and small shapes
        image = np.zeros((1, 300, 300, 1), dtype=np.uint8)
        # Small shape
        image[0, 20:40, 20:40, 0] = 255
        # Very large shape
        image[0, 50:280, 50:280, 0] = 255

        # Processor with low max_area will filter out large shape
        processor = CannySegmentationProcessor(min_area=50, max_area=5000)
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # All remaining contours should be below max_area
        for contour in overlay.contours:
            self.assertLessEqual(contour.score, 5000)

    def test_no_contours_with_strict_filtering(self):
        """Test that strict filtering can result in no contours."""
        # Create image with medium-sized shape
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image[0, 30:70, 30:70, 0] = 255

        # Very restrictive filtering
        processor = CannySegmentationProcessor(min_area=10000, max_area=100000)
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # May have no contours due to strict filtering
        self.assertIsInstance(overlay, Overlay)

    def test_zero_contours_from_empty_image(self):
        """Test that empty image produces no contours."""
        # Create all-black image
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Should have no contours
        self.assertEqual(len(overlay), 0)

    def test_solidity_filtering(self):
        """Test that poorly shaped contours are filtered by solidity."""
        # Create image with diagonal line (low solidity)
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        for i in range(20):
            image[0, 20 + i, 20 + i, 0] = 255

        # Create solid square (high solidity)
        image[0, 60:80, 60:80, 0] = 255

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Should mostly detect the solid square, not the thin line
        self.assertGreater(len(overlay), 0)


class TestCannyEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def test_empty_image_sequence(self):
        """Test with empty image sequence."""
        processor = CannySegmentationProcessor()
        # Create empty sequence with shape (0, 100, 100, 1)
        image = np.zeros((0, 100, 100, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)
        overlay = processor(source)

        self.assertEqual(len(overlay), 0)
        self.assertIsInstance(overlay, Overlay)

    def test_all_black_image(self):
        """Test with all-black image (no edges)."""
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        self.assertEqual(len(overlay), 0)

    def test_all_white_image(self):
        """Test with all-white image (uniform intensity)."""
        image = np.ones((1, 100, 100, 1), dtype=np.uint8) * 255

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Uniform image should produce no edges
        self.assertEqual(len(overlay), 0)

    def test_circle_detection(self):
        """Test detection of circular shapes."""
        # Create image with a circle
        image = np.zeros((1, 200, 200, 1), dtype=np.uint8)
        cv2.circle(image[0, :, :, 0], (100, 100), 50, 255, -1)

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Should detect the circle
        self.assertGreater(len(overlay), 0)

    def test_rectangle_detection(self):
        """Test detection of rectangular shapes."""
        # Create image with rectangles
        image = np.zeros((1, 200, 200, 1), dtype=np.uint8)
        cv2.rectangle(image[0, :, :, 0], (50, 50), (150, 100), 255, -1)

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Should detect the rectangle
        self.assertGreater(len(overlay), 0)

    def test_multiple_shapes(self):
        """Test detection of multiple shapes in single image."""
        image = np.zeros((1, 300, 300, 1), dtype=np.uint8)
        # Draw circle
        cv2.circle(image[0, :, :, 0], (75, 75), 40, 255, -1)
        # Draw rectangle
        cv2.rectangle(image[0, :, :, 0], (150, 50), (250, 150), 255, -1)
        # Draw ellipse
        cv2.ellipse(image[0, :, :, 0], (150, 250), (60, 40), 45, 0, 360, 255, -1)

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Should detect multiple shapes
        self.assertGreater(len(overlay), 1)

    def test_grayscale_image_input(self):
        """Test with explicit grayscale image."""
        # Create 3D image (no explicit channel dimension)
        image_2d = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image_2d[0, 30:70, 30:70, 0] = 255

        source = THWCSequenceSource(image_2d)

        processor = CannySegmentationProcessor()
        overlay = processor(source)

        self.assertGreater(len(overlay), 0)

    def test_rgb_to_grayscale_conversion(self):
        """Test that RGB images are converted to grayscale correctly."""
        # Create RGB image with all channels having the same content
        image = np.zeros((1, 100, 100, 3), dtype=np.uint8)
        image[0, 30:70, 30:70, :] = 255  # All channels: white square

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Should detect the square after grayscale conversion
        self.assertGreater(len(overlay), 0)


class TestCannyThresholdSensitivity(unittest.TestCase):
    """Test processor sensitivity to threshold parameters."""

    def test_low_threshold_detects_more_edges(self):
        """Test that lower Canny threshold detects more edges."""
        image = np.zeros((1, 200, 200, 1), dtype=np.uint8)
        # Create gradient edge
        for y in range(100):
            image[0, 100, y, 0] = int(255 * y / 100)

        # Low threshold processor (sensitive)
        processor_low = CannySegmentationProcessor(canny_low=30, canny_high=100)
        # High threshold processor (selective)
        processor_high = CannySegmentationProcessor(canny_low=100, canny_high=200)

        source = THWCSequenceSource(image)
        overlay_low = processor_low(source)
        overlay_high = processor_high(source)

        # Low threshold should detect more (or equal) contours
        self.assertGreaterEqual(len(overlay_low), len(overlay_high))

    def test_high_threshold_detects_fewer_edges(self):
        """Test that higher Canny threshold detects fewer edges."""
        # Create complex image with varying edges
        image = np.zeros((1, 200, 200, 1), dtype=np.uint8)
        cv2.circle(image[0, :, :, 0], (100, 100), 40, 255, 2)  # Thin circle
        cv2.circle(image[0, :, :, 0], (100, 100), 80, 255, 5)  # Thicker circle

        processor_low = CannySegmentationProcessor(canny_low=30, canny_high=100)
        processor_high = CannySegmentationProcessor(canny_low=150, canny_high=300)

        source = THWCSequenceSource(image)
        overlay_low = processor_low(source)
        overlay_high = processor_high(source)

        # High threshold should detect fewer edges
        self.assertLessEqual(len(overlay_high), len(overlay_low))


class TestCannyContourFormat(unittest.TestCase):
    """Test that contours have correct format for use in Overlay."""

    def test_contour_coordinates_format(self):
        """Test that contour coordinates are (x, y) pairs."""
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image[0, 30:70, 30:70, 0] = 255

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        for contour in overlay.contours:
            coords = contour.coordinates
            # Should be Nx2 array of (x, y) pairs
            self.assertEqual(coords.shape[1], 2)
            # Coordinates should be within image bounds
            self.assertTrue(np.all(coords[:, 0] >= 0))
            self.assertTrue(np.all(coords[:, 1] >= 0))
            self.assertTrue(np.all(coords[:, 0] <= 100))
            self.assertTrue(np.all(coords[:, 1] <= 100))

    def test_contour_to_polygon_conversion(self):
        """Test that contours can be converted to Shapely polygons."""
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image[0, 30:70, 30:70, 0] = 255

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        for contour in overlay.contours:
            # Try to create a Shapely polygon (should not raise)
            polygon = Polygon(contour.coordinates)
            self.assertTrue(polygon.is_valid)

    def test_overlay_type_returned(self):
        """Test that processor returns Overlay object."""
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image[0, 30:70, 30:70, 0] = 255

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        result = processor(source)

        self.assertIsInstance(result, Overlay)

    def test_contour_count_matches_overlay_length(self):
        """Test that overlay length matches number of contours."""
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image[0, 30:70, 30:70, 0] = 255

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # len(overlay) should equal number of contours
        self.assertEqual(len(overlay), len(overlay.contours))

    def test_contour_score_type_and_value(self):
        """Test that contour score is numeric and positive."""
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image[0, 30:70, 30:70, 0] = 255

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        for contour in overlay.contours:
            # Score should be numeric
            self.assertIsInstance(contour.score, (int, float, np.number))
            # Score should be positive (represents area or confidence)
            self.assertGreater(contour.score, 0)

    def test_contour_id_consistency(self):
        """Test that all contours have consistent id value."""
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image[0, 30:70, 30:70, 0] = 255

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # All contours should have id=-1 (as per specification)
        for contour in overlay.contours:
            self.assertEqual(contour.id, -1)

    def test_multi_frame_frame_indices(self):
        """Test that frame indices are correct in multi-frame sequences."""
        image = np.zeros((3, 100, 100, 1), dtype=np.uint8)
        # Add objects to each frame
        for t in range(3):
            image[t, 30:70, 30:70, 0] = 255

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Collect frame indices
        frame_indices = [c.frame for c in overlay.contours]

        # Should have frames 0, 1, 2
        self.assertIn(0, frame_indices)
        self.assertIn(1, frame_indices)
        self.assertIn(2, frame_indices)

        # Verify each frame index is correct
        for contour in overlay.contours:
            self.assertIsInstance(contour.frame, int)
            self.assertGreaterEqual(contour.frame, 0)

    def test_empty_frame_in_multi_frame_sequence(self):
        """Test that empty frames in multi-frame sequence are handled."""
        image = np.zeros((3, 100, 100, 1), dtype=np.uint8)
        # Only add object to frame 0 and 2, leave frame 1 empty
        image[0, 30:70, 30:70, 0] = 255
        # Frame 1 is all black (empty)
        image[2, 30:70, 30:70, 0] = 255

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Should still process all frames
        frame_indices = {c.frame for c in overlay.contours}

        # Frame 1 has no contours, but frames 0 and 2 should
        self.assertIn(0, frame_indices)
        self.assertIn(2, frame_indices)

    def test_coordinates_are_2d_array(self):
        """Test that coordinates are stored as 2D numpy arrays."""
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image[0, 30:70, 30:70, 0] = 255

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        for contour in overlay.contours:
            coords = contour.coordinates
            # Must be 2D array
            self.assertEqual(len(coords.shape), 2)
            # Must have at least 3 points (triangle minimum for polygon)
            self.assertGreaterEqual(coords.shape[0], 3)
            # Must be Nx2
            self.assertEqual(coords.shape[1], 2)

    def test_single_frame_all_contours_have_frame_0(self):
        """Test that single frame contours all have frame index 0."""
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image[0, 30:70, 30:70, 0] = 255

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # All contours should have frame=0
        for contour in overlay.contours:
            self.assertEqual(contour.frame, 0)

    def test_overlay_with_no_contours(self):
        """Test that empty overlay is still valid Overlay type."""
        image = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        # All-black image produces no contours

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Should be Overlay instance
        self.assertIsInstance(overlay, Overlay)
        # Should be empty
        self.assertEqual(len(overlay), 0)
        self.assertEqual(len(overlay.contours), 0)


class TestCannyIntegration(unittest.TestCase):
    """Integration tests with other components."""

    def test_realistic_cell_image(self):
        """Test with realistic synthetic cell-like image."""
        # Create a synthetic cell-like image
        image = np.zeros((1, 256, 256, 1), dtype=np.uint8)

        # Draw several circular cells
        for i in range(3):
            for j in range(3):
                x = 50 + i * 80
                y = 50 + j * 80
                cv2.circle(image[0, :, :, 0], (x, y), 30, 200, -1)

        processor = CannySegmentationProcessor(canny_low=50, canny_high=150)
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Should detect multiple cells
        self.assertGreater(len(overlay), 1)

    def test_time_lapse_sequence(self):
        """Test with time-lapse sequence."""
        image = np.zeros((5, 200, 200, 1), dtype=np.uint8)
        for t in range(5):
            # Cells move over time
            offset = t * 10
            cv2.circle(image[t, :, :, 0], (100 + offset, 100), 40, 255, -1)
            cv2.circle(image[t, :, :, 0], (100 - offset, 100), 40, 255, -1)

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Should have contours from all frames
        frames = {c.frame for c in overlay.contours}
        self.assertEqual(len(frames), 5)

    def test_processor_reusability(self):
        """Test that processor can be reused on multiple sequences."""
        processor = CannySegmentationProcessor()

        # First sequence
        image1 = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image1[0, 30:70, 30:70, 0] = 255
        source1 = THWCSequenceSource(image1)
        overlay1 = processor(source1)

        # Second sequence with different image
        image2 = np.zeros((1, 100, 100, 1), dtype=np.uint8)
        image2[0, 20:80, 20:80, 0] = 255
        source2 = THWCSequenceSource(image2)
        overlay2 = processor(source2)

        # Both should produce results
        self.assertGreater(len(overlay1), 0)
        self.assertGreater(len(overlay2), 0)

    def test_large_image_sequence(self):
        """Test with larger image sequence."""
        image = np.zeros((10, 512, 512, 1), dtype=np.uint8)
        for t in range(10):
            # Random circles
            np.random.seed(t)  # For reproducibility
            for _ in range(5):
                x = np.random.randint(50, 462)
                y = np.random.randint(50, 462)
                radius = np.random.randint(20, 50)
                cv2.circle(image[t, :, :, 0], (x, y), radius, 255, -1)

        processor = CannySegmentationProcessor()
        source = THWCSequenceSource(image)
        overlay = processor(source)

        # Should process all frames
        frames = {c.frame for c in overlay.contours}
        self.assertEqual(len(frames), 10)
        self.assertGreater(len(overlay), 10)  # Multiple shapes per frame


if __name__ == "__main__":
    unittest.main()
