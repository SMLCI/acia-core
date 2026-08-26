"""Tests for numpy-style (T, H, W, C) slicing of image sequence sources."""

import numpy as np
import pytest

from acia.base import BaseImage, ImageSequenceSource
from acia.segm.local import InMemorySequenceSource, THWCSequenceSource


def _src(t=6, h=8, w=10, c=3):
    stack = np.arange(t * h * w * c, dtype=np.uint8).reshape(t, h, w, c)
    return THWCSequenceSource(stack), stack


def test_int_index_returns_frame():
    src, stack = _src()
    frame = src[5]
    assert isinstance(frame, BaseImage)
    assert frame.raw.shape == (8, 10, 3)
    np.testing.assert_array_equal(frame.raw, stack[5])


def test_negative_int_index():
    src, stack = _src()
    np.testing.assert_array_equal(src[-1].raw, stack[-1])


def test_int_out_of_range_raises():
    src, _ = _src()
    with pytest.raises(IndexError):
        _ = src[99]


def test_slice_returns_view_sequence():
    src, stack = _src()
    sub = src[::2]
    assert isinstance(sub, ImageSequenceSource)
    assert sub.size_t == 3
    np.testing.assert_array_equal(sub.get_frame(1).raw, stack[2])


def test_range_slice():
    src, _ = _src()
    assert src[3:7].size_t == 3  # frames 3,4,5 (t=6)


def test_fancy_index():
    src, stack = _src()
    sub = src[[0, 2, 5]]
    assert sub.size_t == 3
    np.testing.assert_array_equal(sub.get_frame(2).raw, stack[5])


def test_spatial_crop_and_channel():
    src, stack = _src()
    cropped = src[:, 1:3, 2:5, 0]
    assert cropped.size_t == 6
    assert cropped.size_h == 2
    assert cropped.size_w == 3
    assert cropped.size_c == 1
    np.testing.assert_array_equal(cropped.get_frame(0).raw, stack[0, 1:3, 2:5, 0])


def test_int_t_with_spatial_returns_cropped_frame():
    src, stack = _src()
    img = src[5, 1:3]
    assert isinstance(img, BaseImage)
    np.testing.assert_array_equal(img.raw, stack[5, 1:3])


def test_ellipsis_channel_select():
    src, stack = _src()
    sub = src[..., 0]
    assert sub.get_frame(0).raw.shape == (8, 10)
    np.testing.assert_array_equal(sub.get_frame(0).raw, stack[0, :, :, 0])


def test_to_channel_matches_ellipsis_select():
    # InMemorySequenceSource doesn't override to_channel, so this exercises the
    # generic ImageSequenceSource.to_channel -> self[..., c] base implementation
    # (THWCSequenceSource has its own to_channel override with different, keep-
    # axis semantics).
    stack = np.arange(6 * 8 * 10 * 3, dtype=np.uint8).reshape(6, 8, 10, 3)
    src = InMemorySequenceSource(stack)
    sub = src.to_channel(1)
    assert sub.get_frame(0).raw.shape == (8, 10)
    np.testing.assert_array_equal(sub.get_frame(0).raw, stack[0, :, :, 1])


def test_chained_slicing_composes():
    src, stack = _src()
    sub = src[::2][1:]  # frames 0,2,4 then drop first -> 2,4
    assert sub.size_t == 2
    np.testing.assert_array_equal(sub.get_frame(0).raw, stack[2])
    np.testing.assert_array_equal(sub.get_frame(1).raw, stack[4])


def test_iteration_yields_frames():
    src, stack = _src()
    frames = list(src[1:4])
    assert len(frames) == 3
    np.testing.assert_array_equal(frames[0].raw, stack[1])
