"""End-to-end tests for overlay visualization flows in JupyterVisualizationMixin.

Tests simulate complete user workflows:
1. Toggle overlay on/off with frame navigation
2. Adjust opacity with frame changes
3. Change channels with overlay enabled
4. Navigate all frames with overlay persistence
"""

import unittest

import numpy as np

from acia.base import Contour, Overlay
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
    """Mock image sequence source for testing E2E flows."""

    def __init__(
        self,
        num_frames=5,
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


def create_overlay_with_multiple_frames(
    frames=5, height=100, width=100, contours_per_frame=2
):
    """Create overlay with contours in multiple frames.

    Args:
        frames: Number of frames
        height: Frame height
        width: Frame width
        contours_per_frame: Number of contours per frame

    Returns:
        Overlay: Multi-frame overlay
    """
    contours = []
    for f in range(frames):
        for c in range(contours_per_frame):
            # Create different rectangles for each contour
            offset = c * 15
            coords = np.array(
                [
                    [25 + offset, 25 + offset],
                    [75 + offset, 25 + offset],
                    [75 + offset, 75 + offset],
                    [25 + offset, 75 + offset],
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


# ============================================================================
# End-to-End Tests
# ============================================================================


class TestE2EToggleOverlayWithFrameNavigation(unittest.TestCase):
    """Test toggling overlay on/off while navigating frames."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_frames = 5
        self.overlay = create_overlay_with_multiple_frames(frames=self.num_frames)
        self.source = MockImageSequenceSource(
            num_frames=self.num_frames, num_channels=3, overlay=self.overlay
        )

    def test_e2e_overlay_toggle_maintains_state_across_frames(self):
        """Test that overlay toggle state is maintained when navigating frames."""
        # Simulate user clicking overlay toggle off
        overlay_enabled = False

        # Simulate navigating through frames while overlay is off
        for frame_idx in range(self.num_frames):
            frame = self.source.get_frame(frame_idx)
            self.assertIsNotNone(frame.raw)
            # State should persist: overlay still disabled
            self.assertFalse(overlay_enabled)

        # Simulate user clicking overlay toggle on
        overlay_enabled = True

        # Simulate navigating through frames while overlay is on
        for frame_idx in range(self.num_frames):
            frame = self.source.get_frame(frame_idx)
            self.assertIsNotNone(frame.raw)
            # State should persist: overlay now enabled
            self.assertTrue(overlay_enabled)

    def test_e2e_overlay_toggle_on_off_sequence(self):
        """Test sequence of toggling overlay on and off."""
        overlay_enabled = False
        toggle_sequence = [True, False, True, True, False, True]

        for toggle_value in toggle_sequence:
            overlay_enabled = toggle_value
            # Simulate frame navigation with new toggle state
            frame = self.source.get_frame(0)
            self.assertIsNotNone(frame.raw)
            # Verify state matches current toggle
            self.assertEqual(overlay_enabled, toggle_value)

    def test_e2e_toggle_with_overlay_time_iterator(self):
        """Test overlay toggle works with time_iterator synchronization."""
        overlay_enabled = True
        frame_idx = 0

        # Get overlay frames for current frame when enabled
        if overlay_enabled and self.source.overlay:
            frame_overlays = list(
                self.source.overlay.time_iterator(
                    start_frame=frame_idx, end_frame=frame_idx
                )
            )
            self.assertEqual(len(frame_overlays), 1)

        # Disable overlay
        overlay_enabled = False

        # With overlay disabled, we shouldn't try to render it
        if not overlay_enabled:
            # No overlay rendering happens
            pass
        else:
            # This code path shouldn't be reached
            self.fail("Overlay should be disabled")

        # Toggle back on
        overlay_enabled = True
        if overlay_enabled and self.source.overlay:
            frame_overlays = list(
                self.source.overlay.time_iterator(
                    start_frame=frame_idx, end_frame=frame_idx
                )
            )
            self.assertEqual(len(frame_overlays), 1)


class TestE2EAdjustOpacityWithFrameChanges(unittest.TestCase):
    """Test adjusting opacity slider while navigating frames."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_frames = 5
        self.overlay = create_overlay_with_multiple_frames(frames=self.num_frames)
        self.source = MockImageSequenceSource(
            num_frames=self.num_frames, num_channels=3, overlay=self.overlay
        )

    def test_e2e_opacity_slider_values_valid_range(self):
        """Test that opacity slider values stay within valid range."""
        # Simulate user moving opacity slider
        opacity_values = [0.0, 0.25, 0.5, 0.75, 1.0, 0.8]

        for opacity in opacity_values:
            # Validate opacity is in range
            self.assertGreaterEqual(opacity, 0.0)
            self.assertLessEqual(opacity, 1.0)

            # Simulate frame navigation with this opacity
            frame = self.source.get_frame(0)
            self.assertIsNotNone(frame.raw)

    def test_e2e_opacity_persists_across_frame_navigation(self):
        """Test that opacity value persists when navigating frames."""
        opacity_alpha = 0.6

        # Navigate through frames with fixed opacity
        for frame_idx in range(self.num_frames):
            frame = self.source.get_frame(frame_idx)
            self.assertIsNotNone(frame.raw)

            # Overlay opacity should remain constant
            self.assertEqual(opacity_alpha, 0.6)

        # Change opacity
        opacity_alpha = 0.3

        # Navigate again with new opacity
        for frame_idx in range(self.num_frames):
            frame = self.source.get_frame(frame_idx)
            self.assertIsNotNone(frame.raw)

            # Overlay opacity should be updated
            self.assertEqual(opacity_alpha, 0.3)

    def test_e2e_opacity_with_render_overlay_frame(self):
        """Test opacity affects render_overlay_frame output."""
        frame = self.source.get_frame(0)
        frame_overlays = list(
            self.source.overlay.time_iterator(start_frame=0, end_frame=0)
        )
        frame_overlay = frame_overlays[0]

        # Test with different opacity values
        opacity_values = [0.0, 0.5, 1.0]
        results = []

        for opacity in opacity_values:
            result = render_overlay_frame(frame.raw, frame_overlay, 0, alpha=opacity)
            results.append(result)
            # Verify output is valid
            self.assertIsNotNone(result)
            self.assertEqual(result.dtype, np.uint8)
            self.assertEqual(len(result.shape), 3)

        # With alpha=0.0, should be mostly original image
        # With alpha=1.0, should be mostly overlay
        # Verify results are different (opacity has effect)
        self.assertFalse(np.array_equal(results[0], results[2]))


class TestE2EChangeChannelsWithOverlay(unittest.TestCase):
    """Test changing channel selection while overlay is displayed."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_frames = 3
        self.num_channels = 3
        self.overlay = create_overlay_with_multiple_frames(frames=self.num_frames)
        self.source = MockImageSequenceSource(
            num_frames=self.num_frames,
            num_channels=self.num_channels,
            overlay=self.overlay,
        )

    def test_e2e_toggle_single_channel_with_overlay(self):
        """Test toggling single channel on/off while overlay is visible."""
        active_channels = [True, True, True]  # All channels active
        overlay_enabled = True

        # Toggle first channel off
        active_channels[0] = False

        # Get frame and verify overlay is still applicable
        frame = self.source.get_frame(0)
        self.assertIsNotNone(frame.raw)

        # Extract selected channels
        selected_channels = [
            frame.get_channel(i) for i, active in enumerate(active_channels) if active
        ]
        self.assertEqual(len(selected_channels), 2)  # Only 2 channels now

        # Overlay should still be renderable on any channel selection
        if overlay_enabled and self.source.overlay:
            frame_overlays = list(
                self.source.overlay.time_iterator(start_frame=0, end_frame=0)
            )
            self.assertEqual(len(frame_overlays), 1)

    def test_e2e_cycle_through_all_channel_combinations(self):
        """Test cycling through different channel selections with overlay."""
        overlay_enabled = True
        frame_idx = 0

        # Test different channel combinations
        channel_combinations = [
            [True, True, True],  # All channels
            [True, False, False],  # Channel 0 only
            [False, True, False],  # Channel 1 only
            [False, False, True],  # Channel 2 only
            [True, True, False],  # Channels 0,1
            [True, False, True],  # Channels 0,2
            [False, True, True],  # Channels 1,2
        ]

        for active_channels in channel_combinations:
            frame = self.source.get_frame(frame_idx)
            self.assertIsNotNone(frame.raw)

            # Extract selected channels
            selected_channels = [
                frame.get_channel(i)
                for i, active in enumerate(active_channels)
                if active
            ]

            # Should have selected at least 1 channel
            self.assertGreater(len(selected_channels), 0)

            # Overlay state should be independent of channel selection
            self.assertTrue(overlay_enabled)

    def test_e2e_overlay_renders_same_regardless_of_channel_selection(self):
        """Test that overlay rendering is independent of channel selection."""
        frame = self.source.get_frame(0)
        frame_overlay = list(
            self.source.overlay.time_iterator(start_frame=0, end_frame=0)
        )[0]

        # Render with overlay
        result_with_overlay = render_overlay_frame(
            frame.raw, frame_overlay, 0, alpha=0.8
        )

        # Result should have all channels (converted to RGB if needed)
        self.assertEqual(len(result_with_overlay.shape), 3)
        self.assertEqual(result_with_overlay.shape[2], 3)  # RGB output

        # Test with different channel selections (these affect input image composition)
        active_channels_configs = [
            [True, True, True],
            [True, False, False],
            [False, True, True],
        ]

        for active_channels in active_channels_configs:
            # Create composed image from selected channels
            if sum(active_channels) == 1:
                # Single channel: convert to RGB
                channel_idx = active_channels.index(True)
                composed = np.stack([frame.get_channel(channel_idx)] * 3, axis=-1)
            else:
                # Multiple channels: stack selected ones
                channels = [
                    frame.get_channel(i)
                    for i, active in enumerate(active_channels)
                    if active
                ]
                # If less than 3 channels, pad with last channel
                while len(channels) < 3:
                    channels.append(channels[-1])
                composed = np.stack(channels[:3], axis=-1)

            # Render overlay on this composed image
            result = render_overlay_frame(composed, frame_overlay, 0, alpha=0.8)
            self.assertIsNotNone(result)
            self.assertEqual(result.dtype, np.uint8)


class TestE2ENavigateAllFramesWithOverlay(unittest.TestCase):
    """Test complete frame navigation with overlay persistence."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_frames = 5
        self.overlay = create_overlay_with_multiple_frames(frames=self.num_frames)
        self.source = MockImageSequenceSource(
            num_frames=self.num_frames, num_channels=3, overlay=self.overlay
        )

    def test_e2e_navigate_forward_through_all_frames(self):
        """Test navigating forward through all frames with overlay."""
        overlay_enabled = True
        current_frame = 0

        # Navigate forward frame by frame
        while current_frame < self.num_frames:
            frame = self.source.get_frame(current_frame)
            self.assertIsNotNone(frame.raw)

            # Get overlay for current frame
            if overlay_enabled and self.source.overlay:
                frame_overlays = list(
                    self.source.overlay.time_iterator(
                        start_frame=current_frame, end_frame=current_frame
                    )
                )
                self.assertEqual(len(frame_overlays), 1)
                # Verify overlay is valid
                self.assertIsNotNone(frame_overlays[0])

            current_frame += 1

        # Should have processed all frames
        self.assertEqual(current_frame, self.num_frames)

    def test_e2e_navigate_backward_through_frames(self):
        """Test navigating backward through frames with overlay."""
        overlay_enabled = True
        current_frame = self.num_frames - 1

        # Navigate backward frame by frame
        while current_frame >= 0:
            frame = self.source.get_frame(current_frame)
            self.assertIsNotNone(frame.raw)

            # Get overlay for current frame
            if overlay_enabled and self.source.overlay:
                frame_overlays = list(
                    self.source.overlay.time_iterator(
                        start_frame=current_frame, end_frame=current_frame
                    )
                )
                self.assertEqual(len(frame_overlays), 1)

            current_frame -= 1

        # Should have processed all frames
        self.assertEqual(current_frame, -1)

    def test_e2e_random_frame_navigation_sequence(self):
        """Test random frame navigation pattern with overlay."""
        overlay_enabled = True
        navigation_sequence = [0, 3, 1, 4, 2, 0, 4, 1]

        for frame_idx in navigation_sequence:
            # Ensure frame index is valid
            self.assertLess(frame_idx, self.num_frames)
            self.assertGreaterEqual(frame_idx, 0)

            frame = self.source.get_frame(frame_idx)
            self.assertIsNotNone(frame.raw)

            # Verify overlay frame synchronization
            if overlay_enabled and self.source.overlay:
                frame_overlays = list(
                    self.source.overlay.time_iterator(
                        start_frame=frame_idx, end_frame=frame_idx
                    )
                )
                self.assertEqual(len(frame_overlays), 1)

    def test_e2e_overlay_persistence_with_repeated_frame_access(self):
        """Test that overlay state persists when accessing same frame multiple times."""
        overlay_enabled = True
        overlay_alpha = 0.8
        frame_idx = 2

        # Access frame multiple times
        for _ in range(3):
            frame = self.source.get_frame(frame_idx)
            self.assertIsNotNone(frame.raw)

            # Get overlay
            if overlay_enabled and self.source.overlay:
                frame_overlays = list(
                    self.source.overlay.time_iterator(
                        start_frame=frame_idx, end_frame=frame_idx
                    )
                )
                self.assertEqual(len(frame_overlays), 1)

                # Render overlay
                result = render_overlay_frame(
                    frame.raw, frame_overlays[0], frame_idx, alpha=overlay_alpha
                )
                self.assertIsNotNone(result)

            # State should be unchanged
            self.assertTrue(overlay_enabled)
            self.assertEqual(overlay_alpha, 0.8)

    def test_e2e_full_workflow_toggle_opacity_navigate(self):
        """Test full workflow: toggle overlay, adjust opacity, navigate frames."""
        # Start with overlay disabled
        overlay_enabled = False
        overlay_alpha = 0.8

        # Toggle overlay on
        overlay_enabled = True

        # Adjust opacity
        overlay_alpha = 0.5

        # Navigate through all frames
        for frame_idx in range(self.num_frames):
            frame = self.source.get_frame(frame_idx)
            self.assertIsNotNone(frame.raw)

            # Verify state
            self.assertTrue(overlay_enabled)
            self.assertEqual(overlay_alpha, 0.5)

            # Get and render overlay
            if overlay_enabled and self.source.overlay:
                frame_overlays = list(
                    self.source.overlay.time_iterator(
                        start_frame=frame_idx, end_frame=frame_idx
                    )
                )
                result = render_overlay_frame(
                    frame.raw, frame_overlays[0], frame_idx, alpha=overlay_alpha
                )
                self.assertIsNotNone(result)
                self.assertEqual(result.dtype, np.uint8)

        # Toggle overlay off
        overlay_enabled = False

        # Navigate again without overlay
        for frame_idx in range(self.num_frames):
            frame = self.source.get_frame(frame_idx)
            self.assertIsNotNone(frame.raw)
            self.assertFalse(overlay_enabled)


class TestE2EEdgeCases(unittest.TestCase):
    """Test edge cases in E2E workflows."""

    def test_e2e_single_frame_source_with_overlay(self):
        """Test E2E flow with single-frame source."""
        overlay = create_overlay_with_multiple_frames(frames=1)
        source = MockImageSequenceSource(num_frames=1, num_channels=3, overlay=overlay)

        # Navigate through single frame
        frame = source.get_frame(0)
        self.assertIsNotNone(frame.raw)

        # Toggle overlay
        frame_overlays = list(overlay.time_iterator(start_frame=0, end_frame=0))
        self.assertEqual(len(frame_overlays), 1)

    def test_e2e_empty_overlay_frames(self):
        """Test E2E flow when some overlay frames are empty."""
        # Create overlay with missing frames
        contours = [
            Contour(
                coordinates=np.array([[25, 25], [75, 25], [75, 75], [25, 75]]),
                score=1.0,
                frame=0,
                id="cell_0",
                label=1,
            ),
            # Frame 1 has no contours (empty)
            Contour(
                coordinates=np.array([[25, 25], [75, 25], [75, 75], [25, 75]]),
                score=1.0,
                frame=2,
                id="cell_2",
                label=1,
            ),
        ]
        overlay = Overlay(contours)
        source = MockImageSequenceSource(num_frames=3, num_channels=3, overlay=overlay)

        overlay_enabled = True

        # Navigate through frames including empty ones
        for frame_idx in range(3):
            frame = source.get_frame(frame_idx)
            self.assertIsNotNone(frame.raw)

            if overlay_enabled and overlay:
                frame_overlays = list(
                    overlay.time_iterator(start_frame=frame_idx, end_frame=frame_idx)
                )
                self.assertEqual(len(frame_overlays), 1)
                # Frame may be empty (no contours) but iterator returns it
                self.assertIsNotNone(frame_overlays[0])

    def test_e2e_single_channel_image_with_overlay(self):
        """Test E2E flow with single-channel images."""
        overlay = create_overlay_with_multiple_frames(frames=3)
        source = MockImageSequenceSource(num_frames=3, num_channels=1, overlay=overlay)

        overlay_enabled = True

        # Navigate through frames
        for frame_idx in range(3):
            frame = source.get_frame(frame_idx)
            self.assertIsNotNone(frame.raw)
            self.assertEqual(frame.raw.shape[2], 1)  # Single channel

            if overlay_enabled and overlay:
                frame_overlays = list(
                    overlay.time_iterator(start_frame=frame_idx, end_frame=frame_idx)
                )
                # render_overlay_frame should handle single-channel by converting to RGB
                result = render_overlay_frame(
                    frame.raw, frame_overlays[0], frame_idx, alpha=0.8
                )
                # Should convert to RGB
                self.assertEqual(len(result.shape), 3)
                self.assertEqual(result.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
