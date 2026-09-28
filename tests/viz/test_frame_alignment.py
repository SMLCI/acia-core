"""Overlays stay frame-aligned with their images through filtering and rendering.

Regression: ``apply_cell_filters`` used to drop the overlay's frame set, so when
every cell of the first frame(s) was filtered out, the renderers (which paired
images and overlay frames by position) drew every mask one frame too early.
"""

import logging

import numpy as np

from acia import Q_
from acia.analysis import AreaEx, ExtractorExecutor
from acia.segm.filter import AreaFilter, apply_cell_filters
from acia.segm.formats import overlay_from_masks
from acia.segm.local import THWCSequenceSource
from acia.segm.utils import merge_cells_to_colonies
from acia.viz import render_segmentation_mask, render_tracking_mask


def _movie():
    """3 frames: frame 0 holds only a tiny cell, frames 1-2 one big cell each."""
    masks = np.zeros((3, 40, 40), dtype=np.int32)
    masks[0, 2:4, 2:4] = 1  # tiny -> filtered out
    masks[1, 20:32, 20:32] = 2  # big, bottom-right
    masks[2, 5:17, 5:17] = 3  # big, top-left
    ov = overlay_from_masks(masks)
    src = THWCSequenceSource(
        np.zeros((3, 40, 40, 1), dtype=np.uint8), pixel_size="1 micrometer"
    )
    return ov, src


def _drop_small(ov, src):
    table = ExtractorExecutor().execute(ov, src, [AreaEx()])
    return apply_cell_filters(
        ov, [AreaFilter(Q_(20, "micrometer**2"), None)], properties=table
    )


def test_filtered_overlay_keeps_frame_set():
    ov, src = _movie()
    filtered = _drop_small(ov, src)

    assert len(filtered) == 2
    assert list(filtered.frames()) == [0, 1, 2]
    per_frame = [len(f) for f in filtered.timeIterator()]
    assert per_frame == [0, 1, 1]


def test_render_segmentation_mask_after_filter_is_frame_aligned():
    ov, src = _movie()
    filtered = _drop_small(ov, src)

    frames = np.stack(
        [im.raw for im in render_segmentation_mask(src, filtered, alpha=0.0)]
    )

    assert frames.shape[0] == 3
    assert frames[0].max() == 0  # nothing left on frame 0
    assert frames[1, 25, 25].any() and not frames[1, 10, 10].any()
    assert frames[2, 10, 10].any() and not frames[2, 25, 25].any()


def test_render_tracking_mask_aligns_by_frame_number():
    ov, src = _movie()
    filtered = _drop_small(ov, src)  # labels are the mask values 2 and 3

    frames = np.stack([im.raw for im in render_tracking_mask(src, filtered, alpha=0.0)])

    assert frames[0].max() == 0
    assert frames[1, 25, 25].any()
    assert frames[2, 10, 10].any()


def test_overlay_beyond_source_warns(caplog):
    ov, _ = _movie()
    short = THWCSequenceSource(np.zeros((2, 40, 40, 1), dtype=np.uint8))

    with caplog.at_level(logging.WARNING):
        out = render_segmentation_mask(short, ov)

    assert len(np.stack([im.raw for im in out])) == 2
    assert "beyond" in caplog.text


def test_colony_overlay_keeps_frame_set():
    ov, src = _movie()
    colonies = merge_cells_to_colonies(_drop_small(ov, src), expand=1)

    assert list(colonies.frames()) == [0, 1, 2]
    assert [len(f) for f in colonies.timeIterator()] == [0, 1, 1]
