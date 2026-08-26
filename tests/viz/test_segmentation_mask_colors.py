"""render_segmentation_mask: coloring cells from a per-cell value table."""

import numpy as np
import pandas as pd

from acia.segm.formats import overlay_from_masks
from acia.segm.local import THWCSequenceSource
from acia.viz import render_segmentation_mask


def _overlay_and_source():
    masks = np.zeros((2, 30, 30), dtype=np.int32)
    masks[:, 4:12, 4:12] = 1
    masks[:, 16:24, 16:24] = 2
    ov = overlay_from_masks(masks)
    src = THWCSequenceSource(np.zeros((2, 30, 30, 3), dtype=np.uint8))
    return ov, src


def _frames(seq):
    return np.stack([im.raw for im in seq])


def test_default_still_random_colors():
    ov, src = _overlay_and_source()
    seq = render_segmentation_mask(src, ov)  # no colors -> original behaviour
    assert _frames(seq).shape == (2, 30, 30, 3)


def test_categorical_table_colors_cells():
    ov, src = _overlay_and_source()
    ids = [c.id for c in ov]
    colors = pd.Series({i: ("area" if i % 2 else "kept") for i in ids})
    seq = render_segmentation_mask(src, ov, colors=colors)
    assert _frames(seq).sum() > 0  # cells got colored over black background


def test_numeric_table_uses_colormap():
    ov, src = _overlay_and_source()
    ids = [c.id for c in ov]
    seq = render_segmentation_mask(
        src, ov, colors=pd.Series({i: float(i) for i in ids})
    )
    assert _frames(seq).shape == (2, 30, 30, 3)


def test_single_column_dataframe_accepted():
    ov, src = _overlay_and_source()
    ids = [c.id for c in ov]
    df = pd.DataFrame({"reason": {i: "kept" for i in ids}})
    seq = render_segmentation_mask(src, ov, colors=df)
    assert _frames(seq).shape == (2, 30, 30, 3)


def test_explicit_rgb_and_palette():
    ov, src = _overlay_and_source()
    ids = [c.id for c in ov]
    # explicit red for every cell; alpha=0.0 -> solid mask color (alpha is the
    # weight of the original image, so 0.0 shows pure color)
    seq = render_segmentation_mask(
        src, ov, alpha=0.0, colors=pd.Series({i: (255, 0, 0) for i in ids})
    )
    frames = _frames(seq)
    assert (frames[..., 0] > 200).any()
    assert frames[..., 1].max() < 60
