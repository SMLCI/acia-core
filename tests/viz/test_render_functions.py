"""Module for testing visualization render functions

Tests for:
- render_segmentation
- render_tracking
- render_scalebar
- render_time
- render_segmentation_mask
- render_tracking_mask
"""

import unittest
from datetime import timedelta

import networkx as nx
import numpy as np

from acia import ureg
from acia.base import Contour, Instance, Overlay
from acia.segm.local import InMemorySequenceSource, THWCSequenceSource
from acia.viz import (
    render_scalebar,
    render_segmentation,
    render_segmentation_mask,
    render_time,
    render_tracking,
    render_tracking_mask,
)


# ============================================================================
# Helper functions for creating test fixtures
# ============================================================================


def create_test_image_source(frames=3, height=100, width=100):
    """Create a minimal image source for testing.

    Args:
        frames (int): Number of frames in the sequence
        height (int): Height of images in pixels
        width (int): Width of images in pixels

    Returns:
        InMemorySequenceSource: Image source with random RGB images
    """
    images = np.random.randint(0, 255, size=(frames, height, width, 3), dtype=np.uint8)
    return InMemorySequenceSource(images)


def create_test_overlay_with_contours(frames=3, contours_per_frame=2, height=100, width=100):
    """Create a test overlay with simple rectangular contours.

    Args:
        frames (int): Number of frames
        contours_per_frame (int): Number of contours per frame
        height (int): Image height (for positioning)
        width (int): Image width (for positioning)

    Returns:
        Overlay: Overlay containing rectangular contours
    """
    contours = []
    for f in range(frames):
        for c in range(contours_per_frame):
            offset = c * 20
            coords = np.array([
                [10 + offset, 10],
                [30 + offset, 10],
                [30 + offset, 30],
                [10 + offset, 30],
            ])
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


def create_test_overlay_with_instances(frames=3, instances_per_frame=2, height=100, width=100):
    """Create a test overlay with mask-based instances.

    Args:
        frames (int): Number of frames
        instances_per_frame (int): Number of instances per frame
        height (int): Image height
        width (int): Image width

    Returns:
        Overlay: Overlay containing mask-based instances
    """
    instances = []
    for f in range(frames):
        for i in range(instances_per_frame):
            label = i + 1
            # Create a mask with the instance labeled
            mask = np.zeros((height, width), dtype=np.uint16)
            y_start, x_start = 10 + i * 25, 10 + i * 25
            mask[y_start : y_start + 15, x_start : x_start + 15] = label
            instances.append(
                Instance(
                    mask=mask,
                    frame=f,
                    label=label,
                    id=f"inst_{f}_{i}",
                )
            )
    return Overlay(instances)


def create_simple_tracking_graph(overlay):
    """Create a simple tracking graph connecting consecutive frames.

    Creates edges between cells with the same label across consecutive frames,
    simulating basic cell tracking.

    Args:
        overlay (Overlay): Overlay containing contours/instances

    Returns:
        nx.DiGraph: Directed graph with nodes for each cell and edges for tracking
    """
    G = nx.DiGraph()

    # Add all contours as nodes
    contours = list(overlay)
    for cont in contours:
        G.add_node(cont.id, frame=cont.frame, label=cont.label)

    # Group contours by frame
    frames_dict = {}
    for cont in contours:
        if cont.frame not in frames_dict:
            frames_dict[cont.frame] = []
        frames_dict[cont.frame].append(cont)

    # Connect contours with same label across consecutive frames
    sorted_frames = sorted(frames_dict.keys())
    for i in range(len(sorted_frames) - 1):
        current_frame = sorted_frames[i]
        next_frame = sorted_frames[i + 1]

        for cont in frames_dict[current_frame]:
            # Find matching contour in next frame (by label)
            for next_cont in frames_dict[next_frame]:
                if cont.label == next_cont.label:
                    G.add_edge(cont.id, next_cont.id)

    return G


# ============================================================================
# Tests for render_segmentation
# ============================================================================


class TestRenderSegmentation(unittest.TestCase):
    """Tests for the render_segmentation function"""

    def test_basic_rendering_with_contours(self):
        """Render segmentation on images with contour overlay"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_contours(frames=3, contours_per_frame=2)

        result = render_segmentation(image_source, overlay)

        # Check that result is an InMemorySequenceSource
        self.assertIsInstance(result, InMemorySequenceSource)

        # Verify output has correct number of frames and shape
        self.assertEqual(len(result), 3)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_rendering_with_none_overlay(self):
        """Render segmentation without overlay (None)"""
        image_source = create_test_image_source(frames=3, height=100, width=100)

        result = render_segmentation(image_source, None)

        # Should still return valid result
        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_custom_cell_color(self):
        """Render segmentation with custom cell color"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_contours(frames=3, contours_per_frame=2)

        result = render_segmentation(image_source, overlay, cell_color=(255, 0, 0))

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_grayscale_images(self):
        """Render segmentation on grayscale images (converts to RGB)"""
        # Create grayscale images (2D)
        images = np.random.randint(0, 255, size=(3, 100, 100), dtype=np.uint8)
        image_source = InMemorySequenceSource(images)
        overlay = create_test_overlay_with_contours(frames=3, contours_per_frame=2)

        result = render_segmentation(image_source, overlay)

        # Output should be RGB (3 channels)
        self.assertIsInstance(result, InMemorySequenceSource)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_rendering_single_frame(self):
        """Render segmentation on a single frame"""
        image_source = create_test_image_source(frames=1, height=100, width=100)
        overlay = create_test_overlay_with_contours(frames=1, contours_per_frame=2)

        result = render_segmentation(image_source, overlay)

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 1)
        frame = result.get_frame(0)
        np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_rendering_with_empty_overlay(self):
        """Render segmentation with overlay containing no contours"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = Overlay([])

        result = render_segmentation(image_source, overlay)

        # Should handle empty overlay gracefully
        self.assertIsInstance(result, InMemorySequenceSource)

    def test_output_dtype_is_uint8(self):
        """Verify output images are uint8 dtype"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_contours(frames=3, contours_per_frame=2)

        result = render_segmentation(image_source, overlay)

        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            self.assertEqual(frame.raw.dtype, np.uint8)


# ============================================================================
# Tests for render_tracking
# ============================================================================


class TestRenderTracking(unittest.TestCase):
    """Tests for the render_tracking function"""

    def test_basic_rendering_with_tracking_graph(self):
        """Render tracking on images with overlay and tracking graph"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_contours(frames=3, contours_per_frame=2)
        tracking_graph = create_simple_tracking_graph(overlay)

        result = render_tracking(image_source, overlay, tracking_graph)

        # Check that result is an InMemorySequenceSource
        self.assertIsInstance(result, InMemorySequenceSource)

        # Verify output has correct number of frames and shape
        self.assertEqual(len(result), 3)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_rendering_with_grayscale_images(self):
        """Render tracking on grayscale images (converts to RGB)"""
        # Create grayscale images (2D)
        images = np.random.randint(0, 255, size=(3, 100, 100), dtype=np.uint8)
        image_source = InMemorySequenceSource(images)
        overlay = create_test_overlay_with_contours(frames=3, contours_per_frame=2)
        tracking_graph = create_simple_tracking_graph(overlay)

        result = render_tracking(image_source, overlay, tracking_graph)

        # Output should be RGB (3 channels)
        self.assertIsInstance(result, InMemorySequenceSource)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_rendering_single_frame(self):
        """Render tracking on a single frame"""
        image_source = create_test_image_source(frames=1, height=100, width=100)
        overlay = create_test_overlay_with_contours(frames=1, contours_per_frame=2)
        tracking_graph = create_simple_tracking_graph(overlay)

        result = render_tracking(image_source, overlay, tracking_graph)

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 1)
        frame = result.get_frame(0)
        np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_rendering_with_empty_tracking_graph(self):
        """Render tracking with empty tracking graph"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_contours(frames=3, contours_per_frame=2)
        tracking_graph = nx.DiGraph()  # Empty graph

        result = render_tracking(image_source, overlay, tracking_graph)

        # Should handle empty graph gracefully
        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_output_dtype_is_uint8(self):
        """Verify output images are uint8 dtype"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_contours(frames=3, contours_per_frame=2)
        tracking_graph = create_simple_tracking_graph(overlay)

        result = render_tracking(image_source, overlay, tracking_graph)

        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            self.assertEqual(frame.raw.dtype, np.uint8)

    def test_rendering_with_branching_tracks(self):
        """Render tracking with cell divisions (branching)"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_contours(frames=3, contours_per_frame=3)

        # Create a tracking graph with branching (cell division)
        G = nx.DiGraph()
        for cont in overlay:
            G.add_node(cont.id, frame=cont.frame, label=cont.label)

        # Cell in frame 0 divides into two cells in frame 1
        G.add_edge("cell_0_0", "cell_1_0")
        G.add_edge("cell_0_0", "cell_1_1")  # Division event
        G.add_edge("cell_1_0", "cell_2_0")
        G.add_edge("cell_1_1", "cell_2_1")

        result = render_tracking(image_source, overlay, G)

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
