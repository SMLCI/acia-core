"""Unit tests for overlay frame synchronization in JupyterVisualizationMixin."""

import unittest
from unittest.mock import Mock

import numpy as np

from acia.base import Contour, Overlay


class MockImageSequenceSource:
    """Mock image sequence source for testing frame synchronization."""

    def __init__(self, num_frames=5, num_channels=3):
        """Initialize mock image source.

        Args:
            num_frames: Number of time frames
            num_channels: Number of channels
        """
        self.size_t = num_frames
        self.num_channels = num_channels
        self._frames = []
        for _t in range(num_frames):
            # Create mock frame with raw data
            frame = Mock()
            frame.raw = np.zeros((100, 100, num_channels), dtype=np.uint8)
            frame.get_channel = Mock(return_value=np.zeros((100, 100), dtype=np.uint8))
            self._frames.append(frame)

    def get_frame(self, frame_idx):
        """Get frame at given index."""
        return self._frames[frame_idx]


class TestOverlayFrameSynchronization(unittest.TestCase):
    """Test overlay frame synchronization with image frames."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_frames = 5
        self.num_channels = 3
        self.image_source = MockImageSequenceSource(
            num_frames=self.num_frames, num_channels=self.num_channels
        )

    def test_overlay_time_iterator_single_frame(self):
        """Test that time_iterator returns single frame overlay for specific frame."""
        # Create contours for frame 0
        contours = [
            Contour(
                coordinates=np.array([[10, 10], [20, 10], [20, 20]]),
                score=1.0,
                frame=0,
                id=1,
            )
        ]
        overlay = Overlay(contours)

        # Get overlay for frame 0
        frame_overlays = list(overlay.time_iterator(start_frame=0, end_frame=0))

        self.assertEqual(len(frame_overlays), 1)
        self.assertEqual(len(frame_overlays[0]), 1)
        self.assertEqual(frame_overlays[0].contours[0].frame, 0)

    def test_overlay_time_iterator_multiple_frames(self):
        """Test that time_iterator returns overlays for all requested frames."""
        # Create contours for frames 0, 1, 2
        contours = [
            Contour(
                coordinates=np.array([[10, 10], [20, 10], [20, 20]]),
                score=1.0,
                frame=0,
                id=1,
            ),
            Contour(
                coordinates=np.array([[30, 30], [40, 30], [40, 40]]),
                score=1.0,
                frame=1,
                id=2,
            ),
            Contour(
                coordinates=np.array([[50, 50], [60, 50], [60, 60]]),
                score=1.0,
                frame=2,
                id=3,
            ),
        ]
        overlay = Overlay(contours)

        # Get overlays for frames 0-2
        frame_overlays = list(overlay.time_iterator(start_frame=0, end_frame=2))

        self.assertEqual(len(frame_overlays), 3)
        self.assertEqual(len(frame_overlays[0]), 1)  # Frame 0 has 1 contour
        self.assertEqual(len(frame_overlays[1]), 1)  # Frame 1 has 1 contour
        self.assertEqual(len(frame_overlays[2]), 1)  # Frame 2 has 1 contour

    def test_overlay_time_iterator_empty_frame(self):
        """Test that time_iterator handles frames with no contours."""
        # Create contours only for frames 0 and 2 (skip frame 1)
        contours = [
            Contour(
                coordinates=np.array([[10, 10], [20, 10], [20, 20]]),
                score=1.0,
                frame=0,
                id=1,
            ),
            Contour(
                coordinates=np.array([[30, 30], [40, 30], [40, 40]]),
                score=1.0,
                frame=2,
                id=2,
            ),
        ]
        overlay = Overlay(contours)

        # Get overlays for frames 0-2
        frame_overlays = list(overlay.time_iterator(start_frame=0, end_frame=2))

        self.assertEqual(len(frame_overlays), 3)
        self.assertEqual(len(frame_overlays[0]), 1)  # Frame 0 has 1 contour
        self.assertEqual(len(frame_overlays[1]), 0)  # Frame 1 has 0 contours (empty)
        self.assertEqual(len(frame_overlays[2]), 1)  # Frame 2 has 1 contour

    def test_overlay_frame_indices_match_iteration_order(self):
        """Test that frame indices in contours match the iteration order."""
        # Create contours for multiple frames
        contours = [
            Contour(
                coordinates=np.array([[10, 10], [20, 10], [20, 20]]),
                score=1.0,
                frame=1,
                id=1,
            ),
            Contour(
                coordinates=np.array([[30, 30], [40, 30], [40, 40]]),
                score=1.0,
                frame=1,
                id=2,
            ),
            Contour(
                coordinates=np.array([[50, 50], [60, 50], [60, 60]]),
                score=1.0,
                frame=3,
                id=3,
            ),
            Contour(
                coordinates=np.array([[70, 70], [80, 70], [80, 80]]),
                score=1.0,
                frame=3,
                id=4,
            ),
        ]
        overlay = Overlay(contours)

        # Get overlays for frames 1-3
        frame_overlays = list(overlay.time_iterator(start_frame=1, end_frame=3))

        # Frame 1 should have 2 contours
        self.assertEqual(len(frame_overlays[0]), 2)
        for contour in frame_overlays[0].contours:
            self.assertEqual(contour.frame, 1)

        # Frame 2 should have 0 contours
        self.assertEqual(len(frame_overlays[1]), 0)

        # Frame 3 should have 2 contours
        self.assertEqual(len(frame_overlays[2]), 2)
        for contour in frame_overlays[2].contours:
            self.assertEqual(contour.frame, 3)

    def test_overlay_single_frame_extraction(self):
        """Test extracting single frame overlay during navigation."""
        # Simulate image sequence with overlays
        contours = [
            Contour(
                coordinates=np.array([[10, 10], [20, 10], [20, 20]]),
                score=1.0,
                frame=i,
                id=i + 1,
            )
            for i in range(self.num_frames)
        ]
        overlay = Overlay(contours)

        # Simulate navigating through each frame
        for frame_idx in range(self.num_frames):
            frame_overlay = next(
                overlay.time_iterator(start_frame=frame_idx, end_frame=frame_idx)
            )

            # Each frame should have exactly 1 contour
            self.assertEqual(len(frame_overlay), 1)
            # Contour should match the frame index
            self.assertEqual(frame_overlay.contours[0].frame, frame_idx)
            self.assertEqual(frame_overlay.contours[0].id, frame_idx + 1)

    def test_overlay_frame_sync_with_slider_navigation(self):
        """Test frame synchronization when navigating with time slider."""
        # Create contours with varying number per frame
        contours = [
            Contour(
                coordinates=np.array([[10, 10], [20, 10], [20, 20]]),
                score=1.0,
                frame=0,
                id=1,
            ),
            Contour(
                coordinates=np.array([[30, 30], [40, 30], [40, 40]]),
                score=1.0,
                frame=1,
                id=2,
            ),
            Contour(
                coordinates=np.array([[50, 50], [60, 50], [60, 60]]),
                score=1.0,
                frame=1,
                id=3,
            ),
            Contour(
                coordinates=np.array([[70, 70], [80, 70], [80, 80]]),
                score=1.0,
                frame=2,
                id=4,
            ),
        ]
        overlay = Overlay(contours)

        # Simulate slider movements
        test_frames = [0, 1, 2, 1, 0]
        expected_contour_counts = [1, 2, 1, 2, 1]

        for frame_idx, expected_count in zip(
            test_frames, expected_contour_counts, strict=False
        ):
            frame_overlay = next(
                overlay.time_iterator(start_frame=frame_idx, end_frame=frame_idx)
            )
            self.assertEqual(
                len(frame_overlay),
                expected_count,
                f"Frame {frame_idx} should have {expected_count} contours",
            )

    def test_overlay_empty_overlay(self):
        """Test handling of empty overlay (no contours)."""
        # Create empty overlay
        overlay = Overlay([])

        # Get overlay for any frame
        frame_overlay = next(overlay.time_iterator(start_frame=0, end_frame=0))

        # Should return empty overlay
        self.assertEqual(len(frame_overlay), 0)

    def test_overlay_sparse_frames(self):
        """Test handling overlay with sparse frames (gaps between frames)."""
        # Create contours for frames 0, 3, 5 (with gaps)
        contours = [
            Contour(
                coordinates=np.array([[10, 10], [20, 10], [20, 20]]),
                score=1.0,
                frame=0,
                id=1,
            ),
            Contour(
                coordinates=np.array([[30, 30], [40, 30], [40, 40]]),
                score=1.0,
                frame=3,
                id=2,
            ),
            Contour(
                coordinates=np.array([[50, 50], [60, 50], [60, 60]]),
                score=1.0,
                frame=5,
                id=3,
            ),
        ]
        overlay = Overlay(contours)

        # Get overlays across the range
        frame_overlays = list(overlay.time_iterator(start_frame=0, end_frame=5))

        # Should have 6 frames (0-5)
        self.assertEqual(len(frame_overlays), 6)

        # Check specific frames
        self.assertEqual(len(frame_overlays[0]), 1)  # Frame 0
        self.assertEqual(len(frame_overlays[1]), 0)  # Frame 1 (empty)
        self.assertEqual(len(frame_overlays[2]), 0)  # Frame 2 (empty)
        self.assertEqual(len(frame_overlays[3]), 1)  # Frame 3
        self.assertEqual(len(frame_overlays[4]), 0)  # Frame 4 (empty)
        self.assertEqual(len(frame_overlays[5]), 1)  # Frame 5

    def test_overlay_frame_boundaries(self):
        """Test frame iteration at boundaries (first and last frames)."""
        # Create contours at boundaries
        contours = [
            Contour(
                coordinates=np.array([[10, 10], [20, 10], [20, 20]]),
                score=1.0,
                frame=0,
                id=1,
            ),
            Contour(
                coordinates=np.array([[30, 30], [40, 30], [40, 40]]),
                score=1.0,
                frame=4,
                id=2,
            ),
        ]
        overlay = Overlay(contours)

        # Get first frame
        first_frame_overlay = next(overlay.time_iterator(start_frame=0, end_frame=0))
        self.assertEqual(len(first_frame_overlay), 1)
        self.assertEqual(first_frame_overlay.contours[0].frame, 0)

        # Get last frame
        last_frame_overlay = next(overlay.time_iterator(start_frame=4, end_frame=4))
        self.assertEqual(len(last_frame_overlay), 1)
        self.assertEqual(last_frame_overlay.contours[0].frame, 4)

    def test_overlay_multiple_contours_per_frame(self):
        """Test that multiple contours in same frame are all retrieved."""
        # Create multiple contours for same frame
        contours = [
            Contour(
                coordinates=np.array([[10, 10], [20, 10], [20, 20]]),
                score=1.0,
                frame=2,
                id=1,
            ),
            Contour(
                coordinates=np.array([[30, 30], [40, 30], [40, 40]]),
                score=1.0,
                frame=2,
                id=2,
            ),
            Contour(
                coordinates=np.array([[50, 50], [60, 50], [60, 60]]),
                score=1.0,
                frame=2,
                id=3,
            ),
        ]
        overlay = Overlay(contours)

        # Get frame 2
        frame_overlay = next(overlay.time_iterator(start_frame=2, end_frame=2))

        # Should have all 3 contours
        self.assertEqual(len(frame_overlay), 3)
        ids = {c.id for c in frame_overlay.contours}
        self.assertEqual(ids, {1, 2, 3})

    def test_overlay_frame_preservation_through_iteration(self):
        """Test that frame values are preserved through iteration."""
        # Create contours with distinct frame values
        contours = [
            Contour(
                coordinates=np.array([[10, 10], [20, 10], [20, 20]]),
                score=1.0,
                frame=i * 2,
                id=i,
            )
            for i in range(1, 4)
        ]
        overlay = Overlay(contours)

        # Iterate and verify frame values
        frame_overlays = list(overlay.time_iterator(start_frame=2, end_frame=6))

        # Collect all contours from iteration
        all_contours = []
        for frame_overlay in frame_overlays:
            all_contours.extend(frame_overlay.contours)

        # Verify frame values are preserved
        frame_values = {c.frame for c in all_contours}
        self.assertEqual(frame_values, {2, 4, 6})


class TestFrameSyncWithRenderCallback(unittest.TestCase):
    """Test frame synchronization in the context of render callback."""

    def test_render_callback_frame_overlay_retrieval(self):
        """Test that render callback correctly retrieves overlay for specific frame."""
        # Create overlays for frames 0-2
        contours = [
            Contour(
                coordinates=np.array([[10, 10], [20, 10], [20, 20]]),
                score=1.0,
                frame=0,
                id=1,
            ),
            Contour(
                coordinates=np.array([[30, 30], [40, 30], [40, 40]]),
                score=1.0,
                frame=1,
                id=2,
            ),
            Contour(
                coordinates=np.array([[50, 50], [60, 50], [60, 60]]),
                score=1.0,
                frame=2,
                id=3,
            ),
        ]
        overlay = Overlay(contours)

        # Simulate render callback getting overlay for each frame
        for frame_idx in range(3):
            # This mimics the code in render_image callback
            frame_overlay = next(
                overlay.time_iterator(start_frame=frame_idx, end_frame=frame_idx)
            )

            # Verify we get the correct overlay
            if len(frame_overlay) > 0:
                self.assertEqual(frame_overlay.contours[0].frame, frame_idx)

    def test_render_callback_handles_missing_overlay_frame(self):
        """Test that render callback handles frames with no overlay data."""
        # Create overlay only for frames 0 and 2
        contours = [
            Contour(
                coordinates=np.array([[10, 10], [20, 10], [20, 20]]),
                score=1.0,
                frame=0,
                id=1,
            ),
            Contour(
                coordinates=np.array([[30, 30], [40, 30], [40, 40]]),
                score=1.0,
                frame=2,
                id=2,
            ),
        ]
        overlay = Overlay(contours)

        # Try to get overlay for frame 1 (no data)
        frame_overlay = next(overlay.time_iterator(start_frame=1, end_frame=1))

        # Should return empty overlay
        self.assertEqual(len(frame_overlay), 0)

    def test_frame_sync_rapid_navigation(self):
        """Test frame synchronization with rapid frame navigation."""
        # Create dense overlay with many contours
        contours = []
        for frame_idx in range(10):
            for contour_idx in range(frame_idx + 1):  # Vary number per frame
                contours.append(
                    Contour(
                        coordinates=np.array([[10, 10], [20, 10], [20, 20]]),
                        score=1.0,
                        frame=frame_idx,
                        id=frame_idx * 100 + contour_idx,
                    )
                )
        overlay = Overlay(contours)

        # Simulate rapid frame navigation
        navigation_sequence = [0, 5, 2, 9, 1, 8, 3, 7, 4, 6]
        expected_counts = [1, 6, 3, 10, 2, 9, 4, 8, 5, 7]

        for frame_idx, expected_count in zip(
            navigation_sequence, expected_counts, strict=False
        ):
            frame_overlay = next(
                overlay.time_iterator(start_frame=frame_idx, end_frame=frame_idx)
            )
            self.assertEqual(len(frame_overlay), expected_count)

    def test_overlay_frame_consistency_across_iterations(self):
        """Test that overlay frames are consistent across multiple iterations."""
        # Create overlay
        contours = [
            Contour(
                coordinates=np.array([[10, 10], [20, 10], [20, 20]]),
                score=1.0,
                frame=i % 3,
                id=i,
            )
            for i in range(9)
        ]
        overlay = Overlay(contours)

        # First iteration
        first_iteration_counts = []
        for frame_idx in range(3):
            frame_overlay = next(
                overlay.time_iterator(start_frame=frame_idx, end_frame=frame_idx)
            )
            first_iteration_counts.append(len(frame_overlay))

        # Second iteration - should be identical
        second_iteration_counts = []
        for frame_idx in range(3):
            frame_overlay = next(
                overlay.time_iterator(start_frame=frame_idx, end_frame=frame_idx)
            )
            second_iteration_counts.append(len(frame_overlay))

        # Verify consistency
        self.assertEqual(first_iteration_counts, second_iteration_counts)


if __name__ == "__main__":
    unittest.main()
