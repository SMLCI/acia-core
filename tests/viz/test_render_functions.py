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

    def test_rendering_with_none_overlay_raises_error(self):
        """Render segmentation with None overlay raises AttributeError"""
        image_source = create_test_image_source(frames=3, height=100, width=100)

        # None overlay is not supported - requires valid Overlay object
        with self.assertRaises(AttributeError):
            render_segmentation(image_source, None)

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

    def test_rendering_with_empty_overlay_raises_error(self):
        """Render segmentation with empty overlay raises ValueError"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = Overlay([])

        # Empty overlay is not supported - raises ValueError in numpy min operation
        with self.assertRaises(ValueError):
            render_segmentation(image_source, overlay)

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


# ============================================================================
# Tests for render_scalebar
# ============================================================================


class TestRenderScalebar(unittest.TestCase):
    """Tests for the render_scalebar function"""

    def test_basic_rendering_with_integer_position(self):
        """Render scalebar with integer xy position"""
        image_source = create_test_image_source(frames=3, height=100, width=100)

        result = render_scalebar(
            image_source,
            xy_position=(10, 90),
            size_of_pixel="0.07 micrometer/pixel",
            bar_width="5 micrometer",
            bar_height="0.25 micrometer",
        )

        # Check that result is an InMemorySequenceSource
        self.assertIsInstance(result, InMemorySequenceSource)

        # Verify output has correct number of frames and shape
        self.assertEqual(len(result), 3)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_rendering_with_relative_float_position(self):
        """Render scalebar with relative float xy position (0-1 range)"""
        image_source = create_test_image_source(frames=3, height=100, width=100)

        result = render_scalebar(
            image_source,
            xy_position=(0.1, 0.9),
            size_of_pixel="0.07 micrometer/pixel",
            bar_width="5 micrometer",
            bar_height="0.25 micrometer",
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_rendering_with_background_color(self):
        """Render scalebar with background color"""
        image_source = create_test_image_source(frames=3, height=100, width=100)

        result = render_scalebar(
            image_source,
            xy_position=(10, 90),
            size_of_pixel="0.07 micrometer/pixel",
            bar_width="5 micrometer",
            bar_height="0.25 micrometer",
            background_color=(0, 0, 0),
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_without_text(self):
        """Render scalebar with show_text=False"""
        image_source = create_test_image_source(frames=3, height=100, width=100)

        result = render_scalebar(
            image_source,
            xy_position=(10, 90),
            size_of_pixel="0.07 micrometer/pixel",
            bar_width="5 micrometer",
            bar_height="0.25 micrometer",
            show_text=False,
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_custom_color(self):
        """Render scalebar with custom color"""
        image_source = create_test_image_source(frames=3, height=100, width=100)

        result = render_scalebar(
            image_source,
            xy_position=(10, 90),
            size_of_pixel="0.07 micrometer/pixel",
            bar_width="5 micrometer",
            bar_height="0.25 micrometer",
            color=(255, 0, 0),
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_custom_font_size(self):
        """Render scalebar with custom font size"""
        image_source = create_test_image_source(frames=3, height=100, width=100)

        result = render_scalebar(
            image_source,
            xy_position=(10, 90),
            size_of_pixel="0.07 micrometer/pixel",
            bar_width="5 micrometer",
            bar_height="0.25 micrometer",
            font_size=12,
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_custom_background_margin(self):
        """Render scalebar with custom background margin"""
        image_source = create_test_image_source(frames=3, height=100, width=100)

        result = render_scalebar(
            image_source,
            xy_position=(10, 90),
            size_of_pixel="0.07 micrometer/pixel",
            bar_width="5 micrometer",
            bar_height="0.25 micrometer",
            background_color=(0, 0, 0),
            background_margin_pixel=10,
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_single_frame(self):
        """Render scalebar on a single frame"""
        image_source = create_test_image_source(frames=1, height=100, width=100)

        result = render_scalebar(
            image_source,
            xy_position=(10, 90),
            size_of_pixel="0.07 micrometer/pixel",
            bar_width="5 micrometer",
            bar_height="0.25 micrometer",
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 1)
        frame = result.get_frame(0)
        np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_output_dtype_is_uint8(self):
        """Verify output images are uint8 dtype"""
        image_source = create_test_image_source(frames=3, height=100, width=100)

        result = render_scalebar(
            image_source,
            xy_position=(10, 90),
            size_of_pixel="0.07 micrometer/pixel",
            bar_width="5 micrometer",
            bar_height="0.25 micrometer",
        )

        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            self.assertEqual(frame.raw.dtype, np.uint8)

    def test_invalid_float_x_position_raises_error(self):
        """Float x position > 1.0 should raise ValueError"""
        image_source = create_test_image_source(frames=3, height=100, width=100)

        with self.assertRaises(ValueError):
            render_scalebar(
                image_source,
                xy_position=(1.5, 0.9),
                size_of_pixel="0.07 micrometer/pixel",
                bar_width="5 micrometer",
                bar_height="0.25 micrometer",
            )

    def test_invalid_float_y_position_raises_error(self):
        """Float y position > 1.0 should raise ValueError"""
        image_source = create_test_image_source(frames=3, height=100, width=100)

        with self.assertRaises(ValueError):
            render_scalebar(
                image_source,
                xy_position=(0.1, 1.5),
                size_of_pixel="0.07 micrometer/pixel",
                bar_width="5 micrometer",
                bar_height="0.25 micrometer",
            )

    def test_rendering_with_pint_quantities(self):
        """Render scalebar using pint Quantity objects directly"""
        image_source = create_test_image_source(frames=3, height=100, width=100)

        result = render_scalebar(
            image_source,
            xy_position=(10, 90),
            size_of_pixel=0.07 * ureg.micrometer / ureg.pixel,
            bar_width=5 * ureg.micrometer,
            bar_height=0.25 * ureg.micrometer,
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_larger_image(self):
        """Render scalebar on larger images"""
        image_source = create_test_image_source(frames=3, height=500, width=500)

        result = render_scalebar(
            image_source,
            xy_position=(50, 450),
            size_of_pixel="0.07 micrometer/pixel",
            bar_width="10 micrometer",
            bar_height="0.5 micrometer",
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (500, 500, 3))

    def test_rendering_with_all_options(self):
        """Render scalebar with all optional parameters specified"""
        image_source = create_test_image_source(frames=3, height=200, width=200)

        result = render_scalebar(
            image_source,
            xy_position=(20, 180),
            size_of_pixel="0.07 micrometer/pixel",
            bar_width="5 micrometer",
            bar_height="0.25 micrometer",
            color=(255, 255, 0),
            font_size=18,
            background_color=(50, 50, 50),
            background_margin_pixel=5,
            show_text=True,
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (200, 200, 3))


# ============================================================================
# Tests for render_time
# ============================================================================


class TestRenderTime(unittest.TestCase):
    """Tests for the render_time function"""

    def test_basic_rendering_with_pint_quantities(self):
        """Render time with pint Quantity timepoints"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        timepoints = [0 * ureg.minute, 5 * ureg.minute, 10 * ureg.minute]

        result = render_time(
            image_source,
            xy_position=(10, 10),
            timepoints=timepoints,
        )

        # Check that result is an InMemorySequenceSource
        self.assertIsInstance(result, InMemorySequenceSource)

        # Verify output has correct number of frames and shape
        self.assertEqual(len(result), 3)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_basic_rendering_with_timedelta(self):
        """Render time with timedelta timepoints"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        timepoints = [
            timedelta(minutes=0),
            timedelta(minutes=5),
            timedelta(minutes=10),
        ]

        result = render_time(
            image_source,
            xy_position=(10, 10),
            timepoints=timepoints,
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_rendering_with_mixed_timepoints(self):
        """Render time with mixed pint and timedelta timepoints"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        timepoints = [
            0 * ureg.minute,
            timedelta(minutes=5),
            10 * ureg.minute,
        ]

        result = render_time(
            image_source,
            xy_position=(10, 10),
            timepoints=timepoints,
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_relative_float_position(self):
        """Render time with relative float xy position (0-1 range)"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        timepoints = [0 * ureg.minute, 5 * ureg.minute, 10 * ureg.minute]

        result = render_time(
            image_source,
            xy_position=(0.1, 0.1),
            timepoints=timepoints,
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_rendering_with_background_color(self):
        """Render time with background color"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        timepoints = [0 * ureg.minute, 5 * ureg.minute, 10 * ureg.minute]

        result = render_time(
            image_source,
            xy_position=(10, 10),
            timepoints=timepoints,
            background_color=(0, 0, 0),
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_custom_time_format(self):
        """Render time with custom time format"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        timepoints = [0 * ureg.minute, 5 * ureg.minute, 10 * ureg.minute]

        result = render_time(
            image_source,
            xy_position=(10, 10),
            timepoints=timepoints,
            time_format="{M:02}:{S:02}",
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_custom_color(self):
        """Render time with custom color"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        timepoints = [0 * ureg.minute, 5 * ureg.minute, 10 * ureg.minute]

        result = render_time(
            image_source,
            xy_position=(10, 10),
            timepoints=timepoints,
            color=(255, 0, 0),
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_custom_font_size(self):
        """Render time with custom font size"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        timepoints = [0 * ureg.minute, 5 * ureg.minute, 10 * ureg.minute]

        result = render_time(
            image_source,
            xy_position=(10, 10),
            timepoints=timepoints,
            font_size=12,
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_custom_background_margin(self):
        """Render time with custom background margin"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        timepoints = [0 * ureg.minute, 5 * ureg.minute, 10 * ureg.minute]

        result = render_time(
            image_source,
            xy_position=(10, 10),
            timepoints=timepoints,
            background_color=(0, 0, 0),
            background_margin_pixel=10,
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_single_frame(self):
        """Render time on a single frame"""
        image_source = create_test_image_source(frames=1, height=100, width=100)
        timepoints = [0 * ureg.minute]

        result = render_time(
            image_source,
            xy_position=(10, 10),
            timepoints=timepoints,
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 1)
        frame = result.get_frame(0)
        np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_output_dtype_is_uint8(self):
        """Verify output images are uint8 dtype"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        timepoints = [0 * ureg.minute, 5 * ureg.minute, 10 * ureg.minute]

        result = render_time(
            image_source,
            xy_position=(10, 10),
            timepoints=timepoints,
        )

        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            self.assertEqual(frame.raw.dtype, np.uint8)

    def test_invalid_float_x_position_raises_error(self):
        """Float x position > 1.0 should raise ValueError"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        timepoints = [0 * ureg.minute, 5 * ureg.minute, 10 * ureg.minute]

        with self.assertRaises(ValueError):
            render_time(
                image_source,
                xy_position=(1.5, 0.1),
                timepoints=timepoints,
            )

    def test_invalid_float_y_position_raises_error(self):
        """Float y position > 1.0 should raise ValueError"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        timepoints = [0 * ureg.minute, 5 * ureg.minute, 10 * ureg.minute]

        with self.assertRaises(ValueError):
            render_time(
                image_source,
                xy_position=(0.1, 1.5),
                timepoints=timepoints,
            )

    def test_rendering_with_larger_image(self):
        """Render time on larger images"""
        image_source = create_test_image_source(frames=3, height=500, width=500)
        timepoints = [0 * ureg.minute, 5 * ureg.minute, 10 * ureg.minute]

        result = render_time(
            image_source,
            xy_position=(50, 50),
            timepoints=timepoints,
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (500, 500, 3))

    def test_rendering_with_hours_format(self):
        """Render time with hours-based pint quantities"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        timepoints = [0 * ureg.hour, 1 * ureg.hour, 2 * ureg.hour]

        result = render_time(
            image_source,
            xy_position=(10, 10),
            timepoints=timepoints,
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_seconds_format(self):
        """Render time with seconds-based pint quantities"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        timepoints = [0 * ureg.second, 30 * ureg.second, 60 * ureg.second]

        result = render_time(
            image_source,
            xy_position=(10, 10),
            timepoints=timepoints,
            time_format="{M:02}m {S:02}s",
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_timedelta_hours(self):
        """Render time with timedelta containing hours"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        timepoints = [
            timedelta(hours=0),
            timedelta(hours=1),
            timedelta(hours=2),
        ]

        result = render_time(
            image_source,
            xy_position=(10, 10),
            timepoints=timepoints,
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_all_options(self):
        """Render time with all optional parameters specified"""
        image_source = create_test_image_source(frames=3, height=200, width=200)
        timepoints = [0 * ureg.minute, 15 * ureg.minute, 30 * ureg.minute]

        result = render_time(
            image_source,
            xy_position=(20, 20),
            timepoints=timepoints,
            time_format="{H:02}:{M:02}",
            color=(255, 255, 0),
            font_size=18,
            background_color=(50, 50, 50),
            background_margin_pixel=5,
        )

        self.assertIsInstance(result, InMemorySequenceSource)
        self.assertEqual(len(result), 3)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (200, 200, 3))


# ============================================================================
# Tests for render_segmentation_mask
# ============================================================================


class TestRenderSegmentationMask(unittest.TestCase):
    """Tests for the render_segmentation_mask function"""

    def test_basic_rendering_with_mask_instances(self):
        """Render segmentation mask on images with mask-based instances"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=2)

        result = render_segmentation_mask(image_source, overlay)

        # Check that result is a THWCSequenceSource
        self.assertIsInstance(result, THWCSequenceSource)

        # Verify output has correct number of frames and shape
        self.assertEqual(len(result), 3)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_rendering_with_custom_alpha(self):
        """Render segmentation mask with custom alpha value"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=2)

        result = render_segmentation_mask(image_source, overlay, alpha=0.5)

        self.assertIsInstance(result, THWCSequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_high_alpha(self):
        """Render segmentation mask with high alpha (more original image)"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=2)

        result = render_segmentation_mask(image_source, overlay, alpha=0.95)

        self.assertIsInstance(result, THWCSequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_low_alpha(self):
        """Render segmentation mask with low alpha (more mask visible)"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=2)

        result = render_segmentation_mask(image_source, overlay, alpha=0.2)

        self.assertIsInstance(result, THWCSequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_single_frame(self):
        """Render segmentation mask on a single frame"""
        image_source = create_test_image_source(frames=1, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=1, instances_per_frame=2)

        result = render_segmentation_mask(image_source, overlay)

        self.assertIsInstance(result, THWCSequenceSource)
        self.assertEqual(len(result), 1)
        frame = result.get_frame(0)
        np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_rendering_with_empty_overlay_raises_error(self):
        """Render segmentation mask with empty overlay raises ValueError"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = Overlay([])

        # Empty overlay is not supported - raises ValueError in numpy min operation
        with self.assertRaises(ValueError):
            render_segmentation_mask(image_source, overlay)

    def test_output_dtype_is_uint8(self):
        """Verify output images are uint8 dtype"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=2)

        result = render_segmentation_mask(image_source, overlay)

        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            self.assertEqual(frame.raw.dtype, np.uint8)

    def test_rendering_with_larger_image(self):
        """Render segmentation mask on larger images"""
        image_source = create_test_image_source(frames=3, height=500, width=500)
        overlay = create_test_overlay_with_instances(
            frames=3, instances_per_frame=3, height=500, width=500
        )

        result = render_segmentation_mask(image_source, overlay)

        self.assertIsInstance(result, THWCSequenceSource)
        self.assertEqual(len(result), 3)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (500, 500, 3))

    def test_rendering_multiple_instances_per_frame(self):
        """Render segmentation mask with multiple instances per frame"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=5)

        result = render_segmentation_mask(image_source, overlay)

        self.assertIsInstance(result, THWCSequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_single_instance_per_frame(self):
        """Render segmentation mask with a single instance per frame"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=1)

        result = render_segmentation_mask(image_source, overlay)

        self.assertIsInstance(result, THWCSequenceSource)
        self.assertEqual(len(result), 3)

    def test_output_preserves_image_dimensions(self):
        """Verify output images preserve the input image dimensions"""
        heights = [50, 100, 200]
        widths = [75, 150, 300]

        for height, width in zip(heights, widths, strict=False):
            image_source = create_test_image_source(frames=2, height=height, width=width)
            overlay = create_test_overlay_with_instances(
                frames=2, instances_per_frame=1, height=height, width=width
            )

            result = render_segmentation_mask(image_source, overlay)

            frame = result.get_frame(0)
            self.assertEqual(frame.raw.shape[0], height)
            self.assertEqual(frame.raw.shape[1], width)
            self.assertEqual(frame.raw.shape[2], 3)

    def test_output_is_rgb(self):
        """Verify output images have 3 channels (RGB)"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=2)

        result = render_segmentation_mask(image_source, overlay)

        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            self.assertEqual(len(frame.raw.shape), 3)
            self.assertEqual(frame.raw.shape[2], 3)


# ============================================================================
# Tests for render_tracking_mask
# ============================================================================


class TestRenderTrackingMask(unittest.TestCase):
    """Tests for the render_tracking_mask function"""

    def test_basic_rendering_with_mask_instances(self):
        """Render tracking mask on images with mask-based instances"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=2)

        result = render_tracking_mask(image_source, overlay)

        # Check that result is a THWCSequenceSource
        self.assertIsInstance(result, THWCSequenceSource)

        # Verify output has correct number of frames and shape
        self.assertEqual(len(result), 3)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_consistent_label_colors_across_frames(self):
        """Verify that same label gets same color across different frames"""
        # Create a black image source for easier color detection
        images = np.zeros((3, 100, 100, 3), dtype=np.uint8)
        image_source = InMemorySequenceSource(images)

        # Create instances where same label appears in multiple frames
        instances = []
        for f in range(3):
            # Instance with label 1 in all frames
            mask1 = np.zeros((100, 100), dtype=np.uint16)
            mask1[10:25, 10:25] = 1
            instances.append(Instance(mask=mask1, frame=f, label=1, id=f"inst_{f}_0"))

            # Instance with label 2 in all frames
            mask2 = np.zeros((100, 100), dtype=np.uint16)
            mask2[50:65, 50:65] = 2
            instances.append(Instance(mask=mask2, frame=f, label=2, id=f"inst_{f}_1"))

        overlay = Overlay(instances)

        result = render_tracking_mask(image_source, overlay, alpha=0.0)  # alpha=0 to see only mask colors

        # Extract colors from label 1 region across all frames
        label1_colors = []
        for frame_idx in range(3):
            frame = result.get_frame(frame_idx).raw
            # Get color from center of label 1 region
            color = tuple(frame[17, 17, :])
            label1_colors.append(color)

        # All frames should have the same color for label 1
        self.assertEqual(label1_colors[0], label1_colors[1])
        self.assertEqual(label1_colors[1], label1_colors[2])

        # Extract colors from label 2 region across all frames
        label2_colors = []
        for frame_idx in range(3):
            frame = result.get_frame(frame_idx).raw
            # Get color from center of label 2 region
            color = tuple(frame[57, 57, :])
            label2_colors.append(color)

        # All frames should have the same color for label 2
        self.assertEqual(label2_colors[0], label2_colors[1])
        self.assertEqual(label2_colors[1], label2_colors[2])

        # Label 1 and label 2 should have different colors
        self.assertNotEqual(label1_colors[0], label2_colors[0])

    def test_seed_produces_reproducible_colors(self):
        """Verify that the same seed produces the same colors"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=2)

        result1 = render_tracking_mask(image_source, overlay, seed=42)
        result2 = render_tracking_mask(image_source, overlay, seed=42)

        # Results should be identical with same seed
        for frame_idx in range(len(result1)):
            frame1 = result1.get_frame(frame_idx).raw
            frame2 = result2.get_frame(frame_idx).raw
            np.testing.assert_array_equal(frame1, frame2)

    def test_different_seeds_produce_different_colors(self):
        """Verify that different seeds produce different colors"""
        # Use black background for easier comparison
        images = np.zeros((3, 100, 100, 3), dtype=np.uint8)
        image_source = InMemorySequenceSource(images)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=2)

        result1 = render_tracking_mask(image_source, overlay, seed=42, alpha=0.0)
        result2 = render_tracking_mask(image_source, overlay, seed=123, alpha=0.0)

        # At least one frame should differ with different seeds
        any_difference = False
        for frame_idx in range(len(result1)):
            frame1 = result1.get_frame(frame_idx).raw
            frame2 = result2.get_frame(frame_idx).raw
            if not np.array_equal(frame1, frame2):
                any_difference = True
                break

        self.assertTrue(any_difference)

    def test_rendering_with_custom_alpha(self):
        """Render tracking mask with custom alpha value"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=2)

        result = render_tracking_mask(image_source, overlay, alpha=0.5)

        self.assertIsInstance(result, THWCSequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_high_alpha(self):
        """Render tracking mask with high alpha (more original image)"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=2)

        result = render_tracking_mask(image_source, overlay, alpha=0.95)

        self.assertIsInstance(result, THWCSequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_low_alpha(self):
        """Render tracking mask with low alpha (more mask visible)"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=2)

        result = render_tracking_mask(image_source, overlay, alpha=0.2)

        self.assertIsInstance(result, THWCSequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_with_show_label_numbers(self):
        """Render tracking mask with label numbers displayed"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=2)

        result = render_tracking_mask(image_source, overlay, show_label_numbers=True)

        self.assertIsInstance(result, THWCSequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_single_frame(self):
        """Render tracking mask on a single frame"""
        image_source = create_test_image_source(frames=1, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=1, instances_per_frame=2)

        result = render_tracking_mask(image_source, overlay)

        self.assertIsInstance(result, THWCSequenceSource)
        self.assertEqual(len(result), 1)
        frame = result.get_frame(0)
        np.testing.assert_array_equal(frame.raw.shape, (100, 100, 3))

    def test_rendering_with_empty_overlay_raises_error(self):
        """Render tracking mask with empty overlay raises ValueError"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = Overlay([])

        # Empty overlay is not supported - raises ValueError in numpy min operation
        with self.assertRaises(ValueError):
            render_tracking_mask(image_source, overlay)

    def test_output_dtype_is_uint8(self):
        """Verify output images are uint8 dtype"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=2)

        result = render_tracking_mask(image_source, overlay)

        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            self.assertEqual(frame.raw.dtype, np.uint8)

    def test_rendering_with_larger_image(self):
        """Render tracking mask on larger images"""
        image_source = create_test_image_source(frames=3, height=500, width=500)
        overlay = create_test_overlay_with_instances(
            frames=3, instances_per_frame=3, height=500, width=500
        )

        result = render_tracking_mask(image_source, overlay)

        self.assertIsInstance(result, THWCSequenceSource)
        self.assertEqual(len(result), 3)
        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            np.testing.assert_array_equal(frame.raw.shape, (500, 500, 3))

    def test_rendering_multiple_instances_per_frame(self):
        """Render tracking mask with multiple instances per frame"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=5)

        result = render_tracking_mask(image_source, overlay)

        self.assertIsInstance(result, THWCSequenceSource)
        self.assertEqual(len(result), 3)

    def test_rendering_single_instance_per_frame(self):
        """Render tracking mask with a single instance per frame"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=1)

        result = render_tracking_mask(image_source, overlay)

        self.assertIsInstance(result, THWCSequenceSource)
        self.assertEqual(len(result), 3)

    def test_output_preserves_image_dimensions(self):
        """Verify output images preserve the input image dimensions"""
        heights = [50, 100, 200]
        widths = [75, 150, 300]

        for height, width in zip(heights, widths, strict=False):
            image_source = create_test_image_source(frames=2, height=height, width=width)
            overlay = create_test_overlay_with_instances(
                frames=2, instances_per_frame=1, height=height, width=width
            )

            result = render_tracking_mask(image_source, overlay)

            frame = result.get_frame(0)
            self.assertEqual(frame.raw.shape[0], height)
            self.assertEqual(frame.raw.shape[1], width)
            self.assertEqual(frame.raw.shape[2], 3)

    def test_output_is_rgb(self):
        """Verify output images have 3 channels (RGB)"""
        image_source = create_test_image_source(frames=3, height=100, width=100)
        overlay = create_test_overlay_with_instances(frames=3, instances_per_frame=2)

        result = render_tracking_mask(image_source, overlay)

        for frame_idx in range(len(result)):
            frame = result.get_frame(frame_idx)
            self.assertEqual(len(frame.raw.shape), 3)
            self.assertEqual(frame.raw.shape[2], 3)

    def test_background_remains_unchanged(self):
        """Verify background pixels (label 0) remain unchanged"""
        # Create a known background color
        background_color = np.array([100, 150, 200], dtype=np.uint8)
        images = np.full((3, 100, 100, 3), background_color, dtype=np.uint8)
        image_source = InMemorySequenceSource(images)

        # Create a small instance in the corner
        instances = []
        for f in range(3):
            mask = np.zeros((100, 100), dtype=np.uint16)
            mask[10:20, 10:20] = 1
            instances.append(Instance(mask=mask, frame=f, label=1, id=f"inst_{f}_0"))

        overlay = Overlay(instances)

        result = render_tracking_mask(image_source, overlay)

        # Check that background pixels (far from instance) remain unchanged
        for frame_idx in range(3):
            frame = result.get_frame(frame_idx).raw
            # Check a pixel far from the instance
            np.testing.assert_array_equal(frame[80, 80, :], background_color)

    def test_label_colors_differ_for_different_labels(self):
        """Verify that different labels get different colors"""
        # Use black background for easier color detection
        images = np.zeros((3, 100, 100, 3), dtype=np.uint8)
        image_source = InMemorySequenceSource(images)

        # Create instances with different labels
        instances = []
        for label in range(1, 6):  # 5 different labels
            mask = np.zeros((100, 100), dtype=np.uint16)
            y_start = label * 15
            mask[y_start : y_start + 10, 10:20] = label
            instances.append(
                Instance(mask=mask, frame=0, label=label, id=f"inst_0_{label}")
            )

        overlay = Overlay(instances)

        result = render_tracking_mask(image_source, overlay, alpha=0.0)

        # Collect colors for each label
        frame = result.get_frame(0).raw
        colors = set()
        for label in range(1, 6):
            y_center = label * 15 + 5
            color = tuple(frame[y_center, 15, :])
            colors.add(color)

        # All labels should have different colors (at least most of them)
        # With random colors, there's a tiny chance of collision, so we check for at least 4 unique
        self.assertGreaterEqual(len(colors), 4)


if __name__ == "__main__":
    unittest.main()
