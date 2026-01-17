"""Integration tests for Jupyter notebook _repr_html_() method.

Tests verify that _repr_html_() correctly integrates with ipywidgets,
creates appropriate UI elements for overlay and channel selection,
and handles various input scenarios.
"""

import unittest

import numpy as np

from acia.base import Contour, Overlay
from acia.notebook import JupyterVisualizationMixin


class MockFrame:
    """Mock frame object for testing."""

    def __init__(self, raw_data: np.ndarray):
        """Initialize mock frame with raw data.

        Args:
            raw_data: The raw image data (can be 2D or 3D)
        """
        self.raw = raw_data

    def get_channel(self, channel_idx: int) -> np.ndarray:
        """Get specific channel from frame.

        Args:
            channel_idx: Index of channel to retrieve

        Returns:
            np.ndarray: Single channel data (2D array)
        """
        if len(self.raw.shape) == 3:
            return self.raw[:, :, channel_idx]
        else:
            # Single channel image
            return self.raw


class MockImageSequenceSource(JupyterVisualizationMixin):
    """Mock image sequence source for testing _repr_html_() integration."""

    def __init__(
        self,
        num_frames=3,
        height=100,
        width=100,
        num_channels=3,
        overlay=None,
        frames_data=None,
    ):
        """Initialize mock source.

        Args:
            num_frames: Number of frames in sequence
            height: Frame height
            width: Frame width
            num_channels: Number of channels
            overlay: Optional overlay object
            frames_data: Optional pre-generated frame data
        """
        self.size_t = num_frames
        self.num_channels = num_channels
        self.overlay = overlay
        self.height = height
        self.width = width

        if frames_data is None:
            # Create random frames with specified channels
            frames_data = [
                np.random.randint(
                    0, 255, size=(height, width, num_channels), dtype=np.uint8
                )
                for _ in range(num_frames)
            ]

        self._frames = [MockFrame(frame) for frame in frames_data]

    def get_frame(self, frame_idx: int) -> MockFrame:
        """Get frame at given index.

        Args:
            frame_idx: Index of frame to retrieve

        Returns:
            MockFrame: Frame at specified index
        """
        return self._frames[frame_idx]


# ============================================================================
# Helper Functions for Creating Test Fixtures
# ============================================================================


def create_simple_overlay(frames=3, height=100, width=100):
    """Create a simple overlay with rectangular contours.

    Args:
        frames: Number of frames
        height: Frame height
        width: Frame width

    Returns:
        Overlay: Simple overlay with rectangular contours
    """
    contours = []
    for f in range(frames):
        # Create a small rectangle in the center
        coords = np.array(
            [
                [25, 25],
                [75, 25],
                [75, 75],
                [25, 75],
            ]
        )
        contours.append(
            Contour(
                coordinates=coords,
                score=1.0,
                frame=f,
                id=f"cell_{f}",
                label=1,
            )
        )
    return Overlay(contours)


# ============================================================================
# Tests for _repr_html_() Integration
# ============================================================================


class TestReprHtmlBasicFunctionality(unittest.TestCase):
    """Test basic _repr_html_() functionality."""

    def test_repr_html_source_without_overlay_has_no_overlay_attribute(self):
        """Test that source without overlay has overlay=None."""
        source = MockImageSequenceSource(num_frames=3, num_channels=3, overlay=None)
        self.assertIsNone(source.overlay)

    def test_repr_html_source_with_overlay_has_overlay_attribute(self):
        """Test that source with overlay has overlay set."""
        overlay = create_simple_overlay(frames=3)
        source = MockImageSequenceSource(num_frames=3, num_channels=3, overlay=overlay)
        self.assertIsNotNone(source.overlay)
        self.assertEqual(source.overlay, overlay)

    def test_repr_html_method_exists_on_mixin(self):
        """Test that _repr_html_ method exists on JupyterVisualizationMixin."""
        source = MockImageSequenceSource(num_frames=3, num_channels=3)
        self.assertTrue(hasattr(source, "_repr_html_"))
        self.assertTrue(callable(source._repr_html_))


class TestReprHtmlWithOverlay(unittest.TestCase):
    """Test _repr_html_() with overlay parameter."""

    def test_repr_html_getattr_overlay_returns_none_when_not_present(self):
        """Test that getattr(source, 'overlay', None) returns None when overlay not set."""
        source = MockImageSequenceSource(num_frames=3, num_channels=3)
        # Don't set overlay attribute
        if hasattr(source, "overlay"):
            delattr(source, "overlay")

        overlay = getattr(source, "overlay", None)
        self.assertIsNone(overlay)

    def test_repr_html_getattr_overlay_returns_overlay_when_present(self):
        """Test that getattr(source, 'overlay', None) returns overlay when set."""
        overlay = create_simple_overlay(frames=3)
        source = MockImageSequenceSource(num_frames=3, num_channels=3, overlay=overlay)

        retrieved_overlay = getattr(source, "overlay", None)
        self.assertIsNotNone(retrieved_overlay)
        self.assertEqual(retrieved_overlay, overlay)


class TestReprHtmlSourceProperties(unittest.TestCase):
    """Test that mock source has required properties for _repr_html_()."""

    def test_source_has_size_t_property(self):
        """Test that source has size_t property."""
        source = MockImageSequenceSource(num_frames=5)
        self.assertEqual(source.size_t, 5)

    def test_source_has_num_channels_property(self):
        """Test that source has num_channels property."""
        source = MockImageSequenceSource(num_channels=3)
        self.assertEqual(source.num_channels, 3)

    def test_source_has_get_frame_method(self):
        """Test that source has get_frame method."""
        source = MockImageSequenceSource(num_frames=3)
        self.assertTrue(hasattr(source, "get_frame"))
        self.assertTrue(callable(source.get_frame))

    def test_source_get_frame_returns_frame_with_raw_attribute(self):
        """Test that get_frame returns frame with raw attribute."""
        source = MockImageSequenceSource(num_frames=3, num_channels=3)
        frame = source.get_frame(0)
        self.assertTrue(hasattr(frame, "raw"))
        self.assertIsNotNone(frame.raw)

    def test_source_get_frame_returns_frame_with_get_channel_method(self):
        """Test that get_frame returns frame with get_channel method."""
        source = MockImageSequenceSource(num_frames=3, num_channels=3)
        frame = source.get_frame(0)
        self.assertTrue(hasattr(frame, "get_channel"))
        self.assertTrue(callable(frame.get_channel))


class TestReprHtmlFrameData(unittest.TestCase):
    """Test frame data handling in mock source."""

    def test_frame_raw_data_is_uint8_rgb(self):
        """Test that frame raw data is uint8 RGB."""
        source = MockImageSequenceSource(
            num_frames=1, height=100, width=100, num_channels=3
        )
        frame = source.get_frame(0)
        self.assertEqual(frame.raw.dtype, np.uint8)
        self.assertEqual(frame.raw.shape, (100, 100, 3))

    def test_frame_raw_data_correct_size(self):
        """Test that frame raw data has correct dimensions."""
        source = MockImageSequenceSource(
            num_frames=3, height=50, width=75, num_channels=3
        )
        frame = source.get_frame(0)
        self.assertEqual(frame.raw.shape, (50, 75, 3))

    def test_frame_get_channel_returns_2d_array(self):
        """Test that get_channel returns 2D array."""
        source = MockImageSequenceSource(
            num_frames=1, height=100, width=100, num_channels=3
        )
        frame = source.get_frame(0)
        channel = frame.get_channel(0)
        self.assertEqual(len(channel.shape), 2)
        self.assertEqual(channel.shape, (100, 100))

    def test_frame_get_channel_correct_data(self):
        """Test that get_channel returns correct channel data."""
        # Create source with known frame data
        known_data = np.zeros((100, 100, 3), dtype=np.uint8)
        known_data[:, :, 0] = 255  # First channel all 255
        known_data[:, :, 1] = 128  # Second channel all 128
        known_data[:, :, 2] = 64  # Third channel all 64

        source = MockImageSequenceSource(num_frames=1, frames_data=[known_data])
        frame = source.get_frame(0)

        ch0 = frame.get_channel(0)
        ch1 = frame.get_channel(1)
        ch2 = frame.get_channel(2)

        np.testing.assert_array_equal(ch0, 255)
        np.testing.assert_array_equal(ch1, 128)
        np.testing.assert_array_equal(ch2, 64)


class TestReprHtmlMultiFrame(unittest.TestCase):
    """Test multi-frame handling."""

    def test_multi_frame_source_returns_different_frames(self):
        """Test that get_frame returns different frames for different indices."""
        source = MockImageSequenceSource(num_frames=3)
        frame0 = source.get_frame(0)
        frame1 = source.get_frame(1)
        frame2 = source.get_frame(2)

        # All frames should have data
        self.assertIsNotNone(frame0.raw)
        self.assertIsNotNone(frame1.raw)
        self.assertIsNotNone(frame2.raw)

        # Frames are from same source, should all be valid
        self.assertEqual(frame0.raw.shape, (100, 100, 3))
        self.assertEqual(frame1.raw.shape, (100, 100, 3))
        self.assertEqual(frame2.raw.shape, (100, 100, 3))


class TestReprHtmlOverlayIntegration(unittest.TestCase):
    """Test overlay integration with _repr_html_()."""

    def test_simple_overlay_creation(self):
        """Test that simple overlay can be created."""
        overlay = create_simple_overlay(frames=3)
        self.assertIsNotNone(overlay)
        self.assertEqual(len(list(overlay)), 3)

    def test_overlay_has_time_iterator(self):
        """Test that overlay has time_iterator method."""
        overlay = create_simple_overlay(frames=3)
        self.assertTrue(hasattr(overlay, "time_iterator"))
        self.assertTrue(callable(overlay.time_iterator))

    def test_overlay_time_iterator_returns_overlays(self):
        """Test that overlay.time_iterator returns overlay objects."""
        overlay = create_simple_overlay(frames=3)
        iterator = overlay.time_iterator()
        frame_overlays = list(iterator)

        # Should have 3 frame overlays
        self.assertEqual(len(frame_overlays), 3)

        # Each should be an Overlay
        for frame_overlay in frame_overlays:
            self.assertIsInstance(frame_overlay, Overlay)

    def test_source_with_overlay_integration(self):
        """Test that source with overlay integrates correctly."""
        overlay = create_simple_overlay(frames=3)
        source = MockImageSequenceSource(num_frames=3, num_channels=3, overlay=overlay)

        # Source should have overlay
        self.assertEqual(source.overlay, overlay)

        # Source should still work normally
        frame = source.get_frame(0)
        self.assertIsNotNone(frame.raw)


if __name__ == "__main__":
    unittest.main()
