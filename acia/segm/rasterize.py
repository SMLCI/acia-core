"""Fast rasterisation of overlay detections into label masks and pixel gathers.

The naive way to turn a frame's detections into an instance label mask is one
full-image ``mask == label`` (or one full-image polygon rasterisation) plus an
``np.maximum`` per cell, which costs O(n_cells * height * width) even though the
cells themselves usually cover well under 1% of the frame. Every routine here is
O(height * width) per frame, or O(cell) per detection.

This module is the single home for that work. It backs the mask renderers
(:mod:`acia.viz`), the tracking processors' input conversion
(:func:`acia.tracking.processor.utils.overlay_to_masks`), the CTC exporters, and
the fluorescence extractor -- all of which used to carry their own copy of the
slow loop.

Two rasterisers for polygons appear below, and choosing between them matters:

* ``rasterio`` follows the pixel-centre rule, which is what
  :func:`acia.utils.polygon_to_mask` -- the full-frame rasterisation everything
  here replaces -- has always used. :func:`contour_pixels` always uses it, and
  :func:`frame_label_mask` uses it under ``exact_polygons=True``.
* ``cv2.fillPoly`` is cheaper but fills a closed polygon *inclusively*: a 10 px
  square covers 11 px, dilating every cell by a pixel (~+11% area on
  bacterium-sized cells). :func:`frame_label_mask` uses it by default, which
  suits the renderers.

Anything whose output is measured, persisted, or associated on -- fluorescence,
the CTC exporters, the trackers' input -- must take the rasterio path, so that
the numbers do not move when the code merely gets faster.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import rasterio.features
import rasterio.transform

from acia.base import Instance

__all__ = [
    "contour_labels",
    "contour_pixels",
    "frame_label_mask",
    "instance_window",
]


def contour_labels(contours: Sequence[Any], enumerate_fallback: bool) -> list[int]:
    """Resolve the integer label to rasterize for each contour.

    Args:
        contours (Sequence): contours/instances of a single frame.
        enumerate_fallback (bool): if True, contours whose ``label`` is None or
            not convertible to int fall back to their 1-based position. If
            False, such contours are skipped (label 0).

    Returns:
        list[int]: one label per contour.
    """
    labels = []
    for i, cont in enumerate(contours):
        label = i + 1 if enumerate_fallback else 0
        if cont.label is not None:
            # could not convert label to integer -> keep the fallback label
            with contextlib.suppress(ValueError, TypeError):
                label = int(cont.label)
        labels.append(label)
    return labels


def instance_window(
    cont: Instance,
) -> tuple[slice, slice, np.ndarray] | None:
    """``(rows, cols, crop)`` bounding-box window of one mask-backed instance.

    ``crop`` is the boolean mask of the instance *within* the window, so
    ``frame[rows, cols][crop]`` addresses exactly the instance's pixels. Reads
    :attr:`~acia.base.Instance._bounds` / ``_cropped_mask``, both cached, rather
    than :attr:`~acia.base.Instance.binary_mask`, which is the one uncached
    property and rebuilds a full-frame comparison on every access.

    Args:
        cont (Instance): the instance to locate.

    Returns:
        tuple[slice, slice, np.ndarray] | None: the window, or None when the
            instance's label is absent from its mask (empty instance).
    """
    bounds = cont._bounds  # noqa: SLF001 -- cached box, this module owns the fast path
    crop = cont._cropped_mask  # noqa: SLF001
    if bounds is None or crop is None:
        return None
    y0, y1, x0, x1 = bounds
    return slice(y0, y1), slice(x0, x1), crop


def _polygon_window(
    cont: Any, height: int, width: int
) -> tuple[slice, slice, np.ndarray] | None:
    """``(rows, cols, crop)`` window of one polygon-backed contour.

    Rasterised with ``rasterio`` inside the contour's bounding box and offset
    back into frame coordinates by an affine translation. That is bit-identical
    to rasterising over the whole frame and cropping -- rasterio's pixel-centre
    rule is unaffected by the translation -- but costs O(cell) rather than
    O(frame).
    """
    coordinates = np.asarray(cont.coordinates, dtype=np.float64).reshape(-1, 2)
    if len(coordinates) == 0:
        return None

    # clip the box to the frame: a contour may stick out past the image edge,
    # and a window has to stay addressable
    x0 = int(np.clip(np.floor(coordinates[:, 0].min()), 0, width))
    x1 = int(np.clip(np.ceil(coordinates[:, 0].max()) + 1, 0, width))
    y0 = int(np.clip(np.floor(coordinates[:, 1].min()), 0, height))
    y1 = int(np.clip(np.ceil(coordinates[:, 1].max()) + 1, 0, height))
    if x1 <= x0 or y1 <= y0:
        return None

    crop = rasterio.features.rasterize(
        [cont.polygon],
        out_shape=(y1 - y0, x1 - x0),
        # maps window pixel -> frame coordinate, so rasterio sees the polygon
        # in its own coordinates while writing into the small buffer
        transform=rasterio.transform.Affine.translation(x0, y0),
    ).astype(bool)

    return slice(y0, y1), slice(x0, x1), crop


def contour_pixels(cont: Any, image: np.ndarray) -> np.ndarray:
    """The image values inside one detection, as a flat array.

    Replaces the full-frame ``np.ma.masked_array(image, mask=~cont.toMask(...))``
    ``.compressed()`` idiom, which allocated two frame-sized temporaries per
    detection. The returned values are in the same order that idiom produced
    (row-major within the frame), so even an order-sensitive summarising
    operator sees an unchanged sequence.

    Args:
        cont (Contour | Instance): the detection to read.
        image (np.ndarray): the frame to sample, ``(height, width[, ...])``.

    Returns:
        np.ndarray: the detection's pixel values; empty when it covers nothing.
    """
    height, width = image.shape[:2]

    window = (
        instance_window(cont)
        if isinstance(cont, Instance)
        else _polygon_window(cont, height, width)
    )

    if window is None:
        return np.empty((0, *image.shape[2:]), dtype=image.dtype)

    rows, cols, crop = window
    return np.asarray(image[rows, cols][crop])


def frame_label_mask(
    contours: Sequence[Any],
    height: int,
    width: int,
    enumerate_fallback: bool = False,
    labels: Sequence[int] | None = None,
    exact_polygons: bool = False,
) -> np.ndarray:
    """Rasterize one frame's contours into a single instance label mask.

    This is the hot path of every mask-based renderer and of every tracking
    backend's input conversion. The naive formulation (one full-image
    ``mask == label`` plus ``np.maximum`` per cell) costs
    O(n_cells * height * width); every branch below is O(height * width).

    On overlapping pixels the higher label wins, matching the ``np.maximum``
    semantics of the original implementation.

    Args:
        contours (Sequence): contours/instances of a single frame.
        height (int): frame height.
        width (int): frame width.
        enumerate_fallback (bool): see :func:`contour_labels`. Ignored when
            ``labels`` is given.
        labels (Sequence[int] | None): explicit label to burn per contour, for
            callers that number detections by something other than
            ``cont.label`` (the CTC exporter burns a life-cycle id). ``None``
            resolves labels via :func:`contour_labels`.
        exact_polygons (bool): which rasterizer polygon-backed contours get.
            False (default) uses ``cv2.fillPoly``, which is fastest but fills a
            closed polygon inclusively -- a 10 px square covers 11 px, dilating
            every cell by a pixel. True uses ``rasterio`` windowed to the
            contour's bounding box: same O(cell) cost, one more allocation, and
            bit-identical to rasterizing over the whole frame. Callers whose
            output is measured or tracked want True; renderers do not care.

    Returns:
        np.ndarray: HxW uint32 label mask (0 = background).
    """
    contours = list(contours)
    if not contours:
        return np.zeros((height, width), dtype=np.uint32)

    if labels is None:
        labels = contour_labels(contours, enumerate_fallback)
    else:
        labels = list(labels)
        if len(labels) != len(contours):
            raise ValueError(
                f"labels has {len(labels)} entries but there are "
                f"{len(contours)} contours; they must correspond one-to-one."
            )

    first = contours[0]
    if isinstance(first, Instance) and all(
        isinstance(c, Instance) and c.mask is first.mask for c in contours
    ):
        # Fast path: acia.segm.formats.overlay_from_masks hands every instance
        # of a frame a reference to the same full-frame label mask, so the mask
        # we want already exists -- one LUT remap keeps the requested labels and
        # drops everything else, instead of one pass per cell.
        src = first.mask
        # the LUT remap returns src.shape, so it can only stand in for a
        # frame-sized rasterisation when the mask really is frame-sized;
        # otherwise fall through and write into a correctly shaped buffer
        if src.shape[:2] == (height, width):
            src_labels = np.asarray([c.label for c in contours])
            lut_size = int(max(int(src.max()), int(src_labels.max()), max(labels))) + 1
            lut = np.zeros(lut_size, dtype=np.uint32)
            lut[src_labels] = np.asarray(labels, dtype=np.uint32)
            return np.asarray(lut[src])

    # Batched exact path: a frame of nothing but polygons -- what
    # load_segmentation returns -- burns in a single rasterio pass. Ascending
    # label order makes rasterio's last-wins merge reproduce "higher label
    # wins", and one C call avoids the per-shape overhead that dominates when
    # rasterizing cells one at a time.
    if exact_polygons and all(
        not isinstance(c, Instance) and getattr(c, "coordinates", None) is not None
        for c in contours
    ):
        local_mask = np.zeros((height, width), dtype=np.int32)
        order = np.argsort(np.asarray(labels, dtype=np.int64), kind="stable")
        shapes = [
            (contours[i].polygon, int(labels[i]))
            for i in order
            # a polygon that failed to build has nothing to burn, and a
            # non-positive label is background -- burning it would erase
            # whatever a lower-labelled cell already wrote
            if contours[i].polygon is not None and labels[i] > 0
        ]
        if shapes:
            rasterio.features.rasterize(shapes, out=local_mask)
        return local_mask.astype(np.uint32)

    # Slow path: write each contour into the shared buffer. Ascending label
    # order reproduces "higher label wins" without a per-cell np.maximum.
    # int32 rather than uint32 because cv2.fillPoly has no uint32 overload.
    local_mask = np.zeros((height, width), dtype=np.int32)

    for i in np.argsort(np.asarray(labels, dtype=np.int64), kind="stable"):
        cont = contours[i]

        if isinstance(cont, Instance):
            # window write rather than a full-frame putmask on binary_mask:
            # _bounds/_cropped_mask are cached, binary_mask is not
            window = instance_window(cont)
            if window is not None:
                rows, cols, crop = window
                np.putmask(local_mask[rows, cols], crop, np.int32(labels[i]))
            continue

        # Instance must be handled above: its `coordinates` property derives a
        # shapely polygon from the mask and raises when the mask is empty.
        coordinates = getattr(cont, "coordinates", None)

        if coordinates is None:
            # anything exposing only the toMask() protocol
            np.putmask(
                local_mask, cont.toMask(height=height, width=width), np.int32(labels[i])
            )
        elif exact_polygons:
            # windowed rasterio: same pixel-centre rule as a full-frame
            # rasterize, so the result is unchanged, but O(cell) not O(frame)
            window = _polygon_window(cont, height, width)
            if window is not None:
                rows, cols, crop = window
                np.putmask(local_mask[rows, cols], crop, np.int32(labels[i]))
        else:
            # cv2.fillPoly only touches the polygon bounding box, whereas
            # Contour.toMask rasterizes over the whole frame per contour.
            points = np.asarray(coordinates, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(local_mask, [points], int(labels[i]))

    return local_mask.astype(np.uint32)
