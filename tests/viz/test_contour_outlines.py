"""Cell outlines drawn on top of the colored masks (``contour_color``)."""

import numpy as np

from acia.segm.formats import overlay_from_masks
from acia.segm.local import THWCSequenceSource
from acia.viz import (
    render_overlay_frame,
    render_segmentation_mask,
    render_tracking_mask,
)

RED = (255, 0, 0)


def _touching_cells():
    """Two cells sharing the border between columns 9 and 10."""
    masks = np.zeros((1, 20, 20), dtype=np.int32)
    masks[0, 5:15, 2:10] = 1
    masks[0, 5:15, 10:18] = 2
    src = THWCSequenceSource(np.full((1, 20, 20, 3), 128, dtype=np.uint8))
    return overlay_from_masks(masks), src


def _is(pixel, color):
    return tuple(int(v) for v in pixel) == color


def test_outline_separates_touching_cells():
    ov, src = _touching_cells()
    frame = next(iter(render_segmentation_mask(src, ov, contour_color=RED))).raw

    # both sides of the shared border are outline
    assert _is(frame[10, 9], RED) and _is(frame[10, 10], RED)
    # outer border of the cell, but not its interior or the background
    assert _is(frame[5, 5], RED)
    assert not _is(frame[10, 5], RED)
    assert _is(frame[0, 0], (128, 128, 128))


def test_no_contour_color_is_unchanged_output():
    ov, src = _touching_cells()
    plain = next(iter(render_segmentation_mask(src, ov))).raw
    none = next(iter(render_segmentation_mask(src, ov, contour_color=None))).raw
    np.testing.assert_array_equal(plain, none)
    assert not any(_is(p, (0, 0, 0)) for p in plain.reshape(-1, 3))


def test_black_is_a_real_color():
    ov, src = _touching_cells()
    frame = next(iter(render_segmentation_mask(src, ov, contour_color=(0, 0, 0)))).raw
    assert _is(frame[10, 9], (0, 0, 0))


def test_thickness_grows_inwards_only():
    ov, src = _touching_cells()
    thin = next(iter(render_segmentation_mask(src, ov, contour_color=RED))).raw
    thick = next(
        iter(render_segmentation_mask(src, ov, contour_color=RED, contour_thickness=2))
    ).raw
    assert not _is(thin[10, 8], RED) and _is(thick[10, 8], RED)
    assert _is(thick[4, 5], (128, 128, 128))  # background untouched


def test_tracking_mask_and_single_frame_support_outlines():
    ov, src = _touching_cells()
    tracked = next(iter(render_tracking_mask(src, ov, contour_color=RED))).raw
    assert _is(tracked[10, 9], RED)

    single = render_overlay_frame(
        np.full((20, 20), 128, np.uint8), ov, 0, contour_color=RED
    )
    assert _is(single[10, 10], RED)
