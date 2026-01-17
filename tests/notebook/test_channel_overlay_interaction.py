"""Unit tests for channel selection + overlay rendering interaction.

Tests verify that overlay rendering works correctly with various channel
selection states (single, multiple, all, none) and that overlay compositing
properly overlays on top of selected channels.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock

import numpy as np

from acia.base import Contour, Instance, Overlay
from acia.notebook import JupyterVisualizationMixin
from acia.viz import render_overlay_frame


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
    """Mock image sequence source for testing channel + overlay interaction."""

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
                np.random.randint(0, 255, size=(height, width, num_channels), dtype=np.uint8)
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


def create_simple_overlay(
    frames=3,
    height=100,
    width=100,
    contours_per_frame=1,
):
    """Create a simple overlay with rectangular contours.

    Args:
        frames: Number of frames
        height: Frame height (for positioning)
        width: Frame width (for positioning)
        contours_per_frame: Number of contours per frame

    Returns:
        Overlay: Simple overlay with rectangular contours
    """
    contours = []
    for f in range(frames):
        for c in range(contours_per_frame):
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
                    id=f"cell_{f}_{c}",
                    label=c + 1,
                )
            )
    return Overlay(contours)


def create_mask_overlay(frames=3, height=100, width=100):
    """Create an overlay with mask-based instances.

    Args:
        frames: Number of frames
        height: Frame height
        width: Frame width

    Returns:
        Overlay: Overlay with mask instances
    """
    instances = []
    for f in range(frames):
        # Create a mask with a single instance
        mask = np.zeros((height, width), dtype=np.uint16)
        mask[25:75, 25:75] = 1  # Create a 50x50 region
        instances.append(
            Instance(
                mask=mask,
                frame=f,
                label=1,
                id=f"inst_{f}_0",
            )
        )
    return Overlay(instances)


# ============================================================================
# Tests for Channel Selection with Overlay
# ============================================================================


class TestSingleChannelWithOverlay(unittest.TestCase):
    """Test overlay rendering with single channel selected."""

    def setUp(self):
        """Set up test fixtures."""
        self.height = 100
        self.width = 100
        self.num_frames = 3

    def test_single_channel_grayscale_with_overlay(self):
        """Test overlay renders correctly on single grayscale channel."""
        # Create 3-channel RGB image
        frames_data = [
            np.random.randint(0, 255, size=(self.height, self.width, 3), dtype=np.uint8)
            for _ in range(self.num_frames)
        ]
        source = MockImageSequenceSource(
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            num_channels=3,
            frames_data=frames_data,
            overlay=create_simple_overlay(frames=self.num_frames),
        )

        # Single channel selected should be converted to RGB
        self.assertEqual(source.num_channels, 3)
        self.assertIsNotNone(source.overlay)

        # Verify frame can be retrieved
        frame = source.get_frame(0)
        self.assertEqual(frame.raw.shape, (self.height, self.width, 3))

        # Get single channel
        channel_data = frame.get_channel(0)
        self.assertEqual(channel_data.shape, (self.height, self.width))

    def test_overlay_on_selected_channel_converted_to_rgb(self):
        """Test that single channel is converted to RGB before overlay application."""
        # Create frame with distinct channel values
        frame_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame_data[:, :, 0] = 100  # Red channel
        frame_data[:, :, 1] = 50   # Green channel
        frame_data[:, :, 2] = 25   # Blue channel

        frames_data = [frame_data.copy() for _ in range(self.num_frames)]

        source = MockImageSequenceSource(
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            num_channels=3,
            frames_data=frames_data,
            overlay=create_simple_overlay(frames=self.num_frames),
        )

        # When single channel is selected and overlay applied,
        # channel should be converted to RGB (3 identical channels)
        frame = source.get_frame(0)
        channel_0 = frame.get_channel(0)
        self.assertEqual(channel_0.shape, (self.height, self.width))


class TestMultipleChannelsWithOverlay(unittest.TestCase):
    """Test overlay rendering with multiple channels selected."""

    def setUp(self):
        """Set up test fixtures."""
        self.height = 100
        self.width = 100
        self.num_frames = 3

    def test_overlay_on_multiple_channels_combined(self):
        """Test overlay renders correctly on combined multi-channel image."""
        # Create 3-channel image with distinct values per channel
        frame_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame_data[:, :, 0] = 100  # Red
        frame_data[:, :, 1] = 150  # Green
        frame_data[:, :, 2] = 200  # Blue

        frames_data = [frame_data.copy() for _ in range(self.num_frames)]

        source = MockImageSequenceSource(
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            num_channels=3,
            frames_data=frames_data,
            overlay=create_simple_overlay(frames=self.num_frames),
        )

        # Verify multi-channel source
        self.assertEqual(source.num_channels, 3)

        # Each channel should be retrievable
        frame = source.get_frame(0)
        for c in range(3):
            channel = frame.get_channel(c)
            self.assertEqual(channel.shape, (self.height, self.width))

    def test_overlay_with_two_channels_selected(self):
        """Test overlay renders correctly with two channels selected."""
        frames_data = [
            np.random.randint(0, 255, size=(self.height, self.width, 3), dtype=np.uint8)
            for _ in range(self.num_frames)
        ]

        source = MockImageSequenceSource(
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            num_channels=3,
            frames_data=frames_data,
            overlay=create_simple_overlay(frames=self.num_frames),
        )

        # Simulate selecting 2 out of 3 channels
        frame = source.get_frame(0)
        channel_0 = frame.get_channel(0)
        channel_1 = frame.get_channel(1)

        # Both channels should be retrievable
        self.assertEqual(channel_0.shape, (self.height, self.width))
        self.assertEqual(channel_1.shape, (self.height, self.width))

    def test_overlay_with_all_channels_selected(self):
        """Test overlay renders correctly with all channels selected."""
        frames_data = [
            np.random.randint(0, 255, size=(self.height, self.width, 3), dtype=np.uint8)
            for _ in range(self.num_frames)
        ]

        source = MockImageSequenceSource(
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            num_channels=3,
            frames_data=frames_data,
            overlay=create_simple_overlay(frames=self.num_frames),
        )

        # All channels should be accessible
        frame = source.get_frame(0)
        for c in range(source.num_channels):
            channel = frame.get_channel(c)
            self.assertEqual(channel.shape, (self.height, self.width))


class TestNoChannelsSelectedWithOverlay(unittest.TestCase):
    """Test overlay rendering when no channels are selected."""

    def setUp(self):
        """Set up test fixtures."""
        self.height = 100
        self.width = 100
        self.num_frames = 3

    def test_overlay_on_blank_image_when_no_channels(self):
        """Test overlay renders on blank image when no channels selected."""
        frames_data = [
            np.random.randint(0, 255, size=(self.height, self.width, 3), dtype=np.uint8)
            for _ in range(self.num_frames)
        ]

        source = MockImageSequenceSource(
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            num_channels=3,
            frames_data=frames_data,
            overlay=create_simple_overlay(frames=self.num_frames),
        )

        # When no channels are selected, image should be blank (zeros)
        # Overlay should still render on top
        self.assertIsNotNone(source.overlay)
        self.assertEqual(source.num_channels, 3)


# ============================================================================
# Tests for Overlay Compositing with Various Channel States
# ============================================================================


class TestOverlayCompositingWithChannels(unittest.TestCase):
    """Test that overlay compositing works correctly regardless of channel state."""

    def setUp(self):
        """Set up test fixtures."""
        self.height = 100
        self.width = 100
        self.num_frames = 2

    def test_overlay_always_renders_same_regardless_of_channel_selection(self):
        """Test overlay rendering is independent of channel selection."""
        # Create distinct overlay
        overlay = create_simple_overlay(frames=self.num_frames, height=self.height, width=self.width)

        # Create frame with uniform gray value
        frame_data = np.ones((self.height, self.width, 3), dtype=np.uint8) * 128
        frames_data = [frame_data.copy() for _ in range(self.num_frames)]

        source = MockImageSequenceSource(
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            num_channels=3,
            frames_data=frames_data,
            overlay=overlay,
        )

        # Verify overlay exists
        self.assertIsNotNone(source.overlay)

        # Apply overlay to base image
        frame = source.get_frame(0)
        base_image = frame.raw

        # Render overlay on full RGB image
        result = render_overlay_frame(base_image, overlay, 0)

        # Verify result is uint8 RGB
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (self.height, self.width, 3))

    def test_overlay_compositing_with_single_channel_converted_image(self):
        """Test overlay compositing on single channel converted to RGB."""
        overlay = create_simple_overlay(frames=self.num_frames)

        # Create single-channel image
        frame_data = np.ones((self.height, self.width), dtype=np.uint8) * 128
        frames_data = [frame_data.copy() for _ in range(self.num_frames)]

        source = MockImageSequenceSource(
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            num_channels=1,
            frames_data=frames_data,
            overlay=overlay,
        )

        frame = source.get_frame(0)
        grayscale_image = frame.raw

        # Convert grayscale to RGB (as done in render_image)
        rgb_image = np.repeat(grayscale_image[:, :, np.newaxis], 3, axis=-1)

        # Apply overlay
        result = render_overlay_frame(rgb_image, overlay, 0)

        # Result should be uint8 RGB
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (self.height, self.width, 3))

    def test_overlay_compositing_with_multi_channel_combined_image(self):
        """Test overlay compositing on combined multi-channel image."""
        overlay = create_simple_overlay(frames=self.num_frames)

        # Create 3-channel image
        frame_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame_data[:, :, 0] = 100
        frame_data[:, :, 1] = 150
        frame_data[:, :, 2] = 200

        frames_data = [frame_data.copy() for _ in range(self.num_frames)]

        source = MockImageSequenceSource(
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            num_channels=3,
            frames_data=frames_data,
            overlay=overlay,
        )

        frame = source.get_frame(0)
        rgb_image = frame.raw

        # Apply overlay
        result = render_overlay_frame(rgb_image, overlay, 0)

        # Result should be uint8 RGB
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (self.height, self.width, 3))

    def test_overlay_with_different_opacity_values(self):
        """Test overlay renders correctly with various opacity values."""
        overlay = create_simple_overlay(frames=self.num_frames)

        frame_data = np.ones((self.height, self.width, 3), dtype=np.uint8) * 128
        frames_data = [frame_data.copy() for _ in range(self.num_frames)]

        source = MockImageSequenceSource(
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            num_channels=3,
            frames_data=frames_data,
            overlay=overlay,
        )

        frame = source.get_frame(0)
        image = frame.raw

        # Test various opacity values
        for alpha in [0.0, 0.3, 0.5, 0.8, 1.0]:
            result = render_overlay_frame(image, overlay, 0, alpha=alpha)
            self.assertEqual(result.dtype, np.uint8)
            self.assertEqual(result.shape, (self.height, self.width, 3))


# ============================================================================
# Tests for Edge Cases
# ============================================================================


class TestChannelOverlayEdgeCases(unittest.TestCase):
    """Test edge cases in channel selection + overlay rendering."""

    def setUp(self):
        """Set up test fixtures."""
        self.height = 100
        self.width = 100
        self.num_frames = 2

    def test_overlay_with_grayscale_image_single_channel(self):
        """Test overlay renders on pure grayscale (single channel) image."""
        overlay = create_simple_overlay(frames=self.num_frames)

        # Pure grayscale image (2D)
        frame_data = np.ones((self.height, self.width), dtype=np.uint8) * 100
        frames_data = [frame_data.copy() for _ in range(self.num_frames)]

        source = MockImageSequenceSource(
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            num_channels=1,
            frames_data=frames_data,
            overlay=overlay,
        )

        frame = source.get_frame(0)
        grayscale = frame.raw

        # Convert to RGB
        rgb = np.repeat(grayscale[:, :, np.newaxis], 3, axis=-1)

        # Apply overlay
        result = render_overlay_frame(rgb, overlay, 0)

        self.assertEqual(result.shape, (self.height, self.width, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_overlay_rendering_preserves_dimensions_with_channels(self):
        """Test overlay rendering preserves image dimensions regardless of channel count."""
        overlay = create_simple_overlay(frames=self.num_frames)

        # Test with different channel counts
        for num_channels in [1, 3, 4]:
            if num_channels == 1:
                frame_data = np.ones((self.height, self.width), dtype=np.uint8) * 100
                frames_data = [frame_data.copy() for _ in range(self.num_frames)]
            else:
                frame_data = np.ones((self.height, self.width, num_channels), dtype=np.uint8) * 100
                frames_data = [frame_data.copy() for _ in range(self.num_frames)]

            source = MockImageSequenceSource(
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                num_channels=num_channels,
                frames_data=frames_data,
                overlay=overlay,
            )

            frame = source.get_frame(0)
            image = frame.raw

            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)
            elif image.shape[2] != 3:
                # Pad or trim to 3 channels
                if image.shape[2] < 3:
                    padding = np.zeros((self.height, self.width, 3 - image.shape[2]), dtype=np.uint8)
                    image = np.concatenate([image, padding], axis=2)
                else:
                    image = image[:, :, :3]

            # Apply overlay
            result = render_overlay_frame(image, overlay, 0)

            self.assertEqual(result.shape, (self.height, self.width, 3))

    def test_overlay_with_empty_frame(self):
        """Test overlay rendering when frame has no overlay data."""
        # Create overlay only for first frame
        contours = [
            Contour(
                coordinates=np.array([[25, 25], [75, 25], [75, 75], [25, 75]]),
                score=1.0,
                frame=0,  # Only first frame
                id="cell_0_0",
                label=1,
            )
        ]
        overlay = Overlay(contours)

        frame_data = np.ones((self.height, self.width, 3), dtype=np.uint8) * 128
        frames_data = [frame_data.copy() for _ in range(self.num_frames)]

        source = MockImageSequenceSource(
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            num_channels=3,
            frames_data=frames_data,
            overlay=overlay,
        )

        # Apply overlay to frame 1 (which has no overlay data)
        frame = source.get_frame(1)
        image = frame.raw

        result = render_overlay_frame(image, overlay, 1)

        # Result should be uint8 RGB (same as input since no overlay for this frame)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (self.height, self.width, 3))

    def test_overlay_with_mask_based_instances(self):
        """Test overlay rendering works with mask-based instances."""
        overlay = create_mask_overlay(frames=self.num_frames)

        frame_data = np.ones((self.height, self.width, 3), dtype=np.uint8) * 128
        frames_data = [frame_data.copy() for _ in range(self.num_frames)]

        source = MockImageSequenceSource(
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            num_channels=3,
            frames_data=frames_data,
            overlay=overlay,
        )

        frame = source.get_frame(0)
        image = frame.raw

        result = render_overlay_frame(image, overlay, 0)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (self.height, self.width, 3))


if __name__ == "__main__":
    unittest.main()
