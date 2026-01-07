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


if __name__ == "__main__":
    unittest.main()
