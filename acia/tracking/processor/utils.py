"""Utility functions for the tracking processors"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from acia.base import Overlay
from acia.segm.rasterize import contour_labels, frame_label_mask


def overlay_to_masks(segmentation: Overlay, height: int, width: int) -> np.ndarray:
    """Rasterize an overlay into the ``(T, height, width)`` label stack trackers want.

    Every tracking backend takes its segmentation this way (Trackastra, LapTrack,
    PyUAT, Ultrack), so this is on the critical path of each of them.

    Frame ``t`` of the returned stack holds the detections of frame ``t`` of the
    overlay -- **indexed absolutely**. The previous implementation built the
    stack from ``Overlay.timeIterator()``, which starts at the overlay's *first
    populated* frame, so an overlay whose earliest detection sat on frame 3 put
    that frame's cells into ``masks[0]`` and handed the tracker a segmentation
    silently shifted against its images.

    Args:
        segmentation (Overlay): the detections to rasterize.
        height (int): frame height, in pixels.
        width (int): frame width, in pixels.

    Returns:
        np.ndarray: ``(T, height, width)`` label stack, 0 = background.
            ``uint16``, widened to ``uint32`` if any label needs it.
    """
    # one pass over the contours instead of the nested timeIterator() the old
    # implementation ran (once here, then again inside Overlay.toMasks for every
    # single-frame sub-overlay, rebuilding an object array and a lookup dict
    # each time)
    by_frame: dict[int, list] = defaultdict(list)
    for cont in segmentation:
        by_frame[cont.frame].append(cont)

    frames = segmentation.frames()
    num_frames = int(np.max(frames)) + 1 if len(frames) else 0
    if by_frame:
        # a detection outside the overlay's declared frame list still has to fit
        num_frames = max(num_frames, int(max(by_frame)) + 1)

    # resolve the labels up front so the buffer's dtype is decided before it is
    # allocated: labels past 65535 must widen the stack rather than wrap in it
    labels = {
        frame: contour_labels(conts, enumerate_fallback=True)
        for frame, conts in by_frame.items()
    }
    max_label = max((max(ls) for ls in labels.values() if ls), default=0)
    dtype = np.uint16 if max_label < np.iinfo(np.uint16).max else np.uint32

    # preallocated rather than np.stack()ed from a list, which would hold the
    # per-frame masks and their stacked copy at the same time
    masks = np.zeros((num_frames, height, width), dtype=dtype)
    for frame, conts in by_frame.items():
        masks[frame] = frame_label_mask(
            conts,
            height=height,
            width=width,
            labels=labels[frame],
            # trackers associate on cell geometry, so the polygons have to
            # rasterize exactly as they always did -- cv2.fillPoly would dilate
            # every cell by a pixel
            exact_polygons=True,
        )

    return masks
