"""Utils for segmentation data handling"""

from itertools import count
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from tqdm.auto import tqdm

from acia.base import Contour, Overlay
from acia.segm.local import THWCSequenceSource
from acia.utils import pairwise_distances


def compute_indices(frame: int, size_t: int, size_z: int) -> tuple[int, int]:
    """Compute t and z values from a linearized frame number

    Args:
        frame (int): the linearized frame index
        size_t (int): the total size of the t dimension
        size_z (int): the total size of the z dimension

    Returns:
        (Tuple[int, int]): tuple of (t,z) indices
    """

    if size_t > 1 and size_z > 1:
        t = int(np.floor(frame / size_t))
        z = frame % size_t
    elif size_t > 1:
        t = frame
        z = 0
    elif size_z >= 1:
        t = 0
        z = frame
    elif size_t == 1 and size_z == 1:
        t = 0
        z = 0
    else:
        raise ValueError("This state should not be reachable!")

    return t, z


def length_and_area(contour: Contour) -> tuple[float, float]:
    """Compute length and area of a contour object (in pixel coordinates)

    Args:
        contour (Contour): contour object

    Returns:
        tuple[float, float]: length and area of the contour
    """

    polygon = Polygon(contour.coordinates)

    length = np.max(
        pairwise_distances(np.array(polygon.minimum_rotated_rectangle.exterior.coords))
    )
    return length, polygon.area


def merge_cells_to_colonies(overlay: Overlay, expand=10) -> Overlay:
    """Computing colony blobs from single-cell overlay

    Args:
        overlay (Overlay): Single-cell overlay containing the individual cell objects
        expand (int, optional): The number of pixels to expand single-cell objects in order to form blobs. Defaults to 10.

    Returns:
        Overlay: Overlay of colony blobs
    """

    merged_contours: list[Contour] = []
    next_id = count()

    # iterate over frames and all the cell instances
    for frame_overlay in tqdm(
        overlay.timeIterator(), desc="Merging cells to colonies..."
    ):
        if len(frame_overlay) == 0:
            # a frame without detections yields no colony blob
            continue

        # the real frame number, taken from the detections themselves -- the
        # position in the iteration is *not* it (timeIterator starts at
        # min(frames()), so a sliced overlay would get silently renumbered)
        frame = int(frame_overlay.contours[0].frame)

        # get all polygons
        cont_polys = [cont.polygon for cont in frame_overlay]

        # increase their size (like a dilation)
        oversized_polys = [poly.buffer(expand) for poly in cont_polys]

        # merge all polys
        intersection = unary_union(oversized_polys)

        # erose the merged polygon
        i = intersection.buffer(-expand)

        # make it a contour
        polygons = [i]
        if isinstance(i, MultiPolygon):
            polygons = list(i.geoms)

        # ids must be unique across the whole overlay, not per frame: a frame can
        # yield several blobs (separate colonies), and property extraction joins
        # extractor results on `id` -- duplicates silently multiply the rows and
        # blow up any per-frame sum (e.g. total colony area)
        contours = [
            Contour(
                np.array(list(zip(p.exterior.xy))).T.squeeze(),
                -1,
                frame,
                next(next_id),
            )
            for p in polygons
        ]

        contours = list(filter(lambda c: len(c.coordinates) >= 3, contours))

        # add merged contour to results
        merged_contours += contours

    # return new overlay with merged contours, carrying the input's time model so
    # the colony overlay stays calibrated on its own
    return Overlay(merged_contours, timepoints=overlay.timepoints)


def _bbox_from_mask(
    mask: np.ndarray,
    margin: int,
) -> tuple[slice, slice] | None:
    """Compute a clipped bounding box with margin from a binary mask.

    Args:
        mask: Binary mask array (2D) where True/non-zero values indicate the object.
        margin: Margin in pixels to add around the bounding box.

    Returns:
        Tuple of (y_slice, x_slice) for numpy array indexing,
        or None if the mask is empty.
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None  # Empty mask

    y_min, y_max = rows.min(), rows.max() + 1
    x_min, x_max = cols.min(), cols.max() + 1

    # Apply margin and clip to bounds
    height, width = mask.shape
    y_start = max(0, y_min - margin)
    y_end = min(height, y_max + margin)
    x_start = max(0, x_min - margin)
    x_end = min(width, x_max + margin)

    return slice(y_start, y_end), slice(x_start, x_end)


def extract_segmentation_stacks(
    source: THWCSequenceSource,
    overlay: Overlay,
    margin: int = 10,
    frame: int | None = 0,
) -> dict[Any, THWCSequenceSource]:
    """Extract individual image stacks for each segmentation in an overlay.

    For each contour/instance in the overlay (optionally filtered by frame),
    extracts a cropped image stack centered on the segmentation's bounding box
    with an optional margin. Uses toMask() to compute bounding boxes.
    The bounding boxes are clipped to image bounds (no padding is applied).

    Args:
        source: The source image stack with shape [T, H, W, C].
        overlay: Overlay containing Contour or Instance objects.
        margin: Margin in pixels to add around each bounding box. Defaults to 10.
        frame: If specified, only extract segmentations from this frame.
            If None, extract all segmentations regardless of frame. Defaults to 0.

    Returns:
        Dictionary mapping contour/instance IDs to their corresponding cropped
        image stacks. Each cropped stack maintains the full time dimension but
        has reduced H and W dimensions.

    Raises:
        ValueError: If margin is negative.

    Example:
        >>> source = THWCSequenceSource(np.zeros((10, 100, 100, 3)))
        >>> contours = [Contour([[10, 10], [20, 10], [20, 30], [10, 30]], -1, 0, id=1)]
        >>> overlay = Overlay(contours)
        >>> stacks = extract_segmentation_stacks(source, overlay, margin=5, frame=0)
        >>> stacks[1].image_stack.shape
        (10, 25, 15, 3)
    """
    if margin < 0:
        raise ValueError(f"Margin must be non-negative, got {margin}")

    # Get image dimensions
    height = source.size_h
    width = source.size_w

    # Filter contours by frame if specified
    if frame is not None:
        contours = [c for c in overlay if c.frame == frame]
    else:
        contours = list(overlay)

    # Handle empty overlay case
    if len(contours) == 0:
        return {}

    # Extract stacks for each contour
    result = {}
    for contour in contours:
        # Use toMask to get binary mask, then compute bounding box
        mask = contour.toMask(height, width)
        bbox = _bbox_from_mask(mask, margin)

        # Skip empty masks
        if bbox is None:
            continue

        y_slice, x_slice = bbox

        # Extract cropped stack: source.image_stack is [T, H, W, C]
        # We want to crop H and W dimensions but keep T and C intact
        cropped_array = source.image_stack[:, y_slice, x_slice, :]

        # Create new THWCSequenceSource from cropped array
        result[contour.id] = THWCSequenceSource(cropped_array)

    return result
