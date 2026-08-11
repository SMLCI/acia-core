"""Tests for the lazy `to_rgb()` view (`RGBSequenceSource`, `acia/base.py`)."""

import numpy as np
import pytest
import tifffile

from acia.base import ArrayImage, BaseImage, ImageSequenceSource, RGBSequenceSource
from acia.notebook import normalize_to_uint8
from acia.segm.local import LocalSequenceSource, THWCSequenceSource
from tests.segm.test_czi_source import _CZITestBase


def _src(t=4, h=5, w=6, c=3, dtype=np.uint8):
    if dtype == np.uint8:
        stack = np.arange(t * h * w * c, dtype=np.uint32).reshape(t, h, w, c) % 256
        stack = stack.astype(np.uint8)
    else:
        stack = np.arange(t * h * w * c, dtype=dtype).reshape(t, h, w, c)
    return THWCSequenceSource(stack), stack


class _GraySource(ImageSequenceSource):
    """Minimal source yielding 2D grayscale (H, W) frames."""

    def __init__(self, stack: np.ndarray):
        self.stack = stack  # (T, H, W)

    def get_frame(self, frame: int) -> BaseImage:
        return ArrayImage(self.stack[frame], frame=frame)

    @property
    def size_t(self) -> int:
        return self.stack.shape[0]

    @property
    def size_h(self) -> int:
        return self.stack.shape[1]

    @property
    def size_w(self) -> int:
        return self.stack.shape[2]

    @property
    def size_c(self) -> int:
        return 1

    @property
    def num_channels(self) -> int:
        return 1


class _CountingSource(ImageSequenceSource):
    """Wraps a source, counting how many times get_frame() is actually called."""

    def __init__(self, parent: ImageSequenceSource):
        self.parent = parent
        self.calls = 0

    def get_frame(self, frame: int) -> BaseImage:
        self.calls += 1
        return self.parent.get_frame(frame)

    @property
    def size_t(self) -> int:
        return self.parent.size_t

    @property
    def size_h(self) -> int:
        return self.parent.size_h

    @property
    def size_w(self) -> int:
        return self.parent.size_w

    @property
    def size_c(self) -> int:
        return self.parent.size_c

    @property
    def num_channels(self) -> int:
        return self.parent.num_channels


# --- grayscale mode ------------------------------------------------------


def test_grayscale_default_channel():
    src, stack = _src()
    view = src.to_rgb()

    assert isinstance(view, RGBSequenceSource)
    assert view.size_t == src.size_t
    assert view.size_h == src.size_h
    assert view.size_w == src.size_w
    assert view.size_c == 3
    assert view.num_channels == 3

    frame = view.get_frame(0).raw
    assert frame.shape == (5, 6, 3)
    assert frame.dtype == np.uint8

    expected_gray = normalize_to_uint8(stack[0, ..., 0])
    expected = np.stack((expected_gray,) * 3, axis=-1)
    np.testing.assert_array_equal(frame, expected)


def test_grayscale_explicit_channel():
    src, stack = _src()
    view = src.to_rgb(channel=1)

    frame = view.get_frame(2).raw
    expected_gray = normalize_to_uint8(stack[2, ..., 1])
    expected = np.stack((expected_gray,) * 3, axis=-1)
    np.testing.assert_array_equal(frame, expected)


def test_grayscale_on_genuinely_2d_frames():
    stack = np.arange(3 * 5 * 6, dtype=np.uint8).reshape(3, 5, 6)
    src = _GraySource(stack)
    view = src.to_rgb()

    frame = view.get_frame(1).raw
    assert frame.shape == (5, 6, 3)
    expected_gray = normalize_to_uint8(stack[1])
    expected = np.stack((expected_gray,) * 3, axis=-1)
    np.testing.assert_array_equal(frame, expected)


# --- composite mode --------------------------------------------------------


def test_composite_blend_hand_computed():
    t, h, w, c = 2, 4, 5, 2
    stack = np.zeros((t, h, w, c), dtype=np.uint8)
    # channel 0: a simple ramp; channel 1: a different ramp
    stack[..., 0] = np.arange(h * w, dtype=np.uint8).reshape(h, w)
    stack[..., 1] = (
        (np.arange(h * w, dtype=np.uint16).reshape(h, w) * 2) % 256
    ).astype(np.uint8)
    src = THWCSequenceSource(stack)

    colors = {0: "#00FF00", 1: "DAPI"}  # green + blue
    view = src.to_rgb(colors=colors)

    frame = view.get_frame(0).raw
    assert frame.shape == (h, w, 3)
    assert frame.dtype == np.uint8

    # hand-computed reference, mirroring the spec's blend formula exactly
    acc = np.zeros((h, w, 3), dtype=np.float32)
    ch0_gray = normalize_to_uint8(stack[0, ..., 0]).astype(np.float32)
    ch1_gray = normalize_to_uint8(stack[0, ..., 1]).astype(np.float32)
    acc += ch0_gray[..., None] * np.array([0.0, 1.0, 0.0], dtype=np.float32)
    acc += ch1_gray[..., None] * np.array([0.0, 0.0, 1.0], dtype=np.float32)
    expected = np.clip(acc, 0, 255).astype(np.uint8)

    np.testing.assert_array_equal(frame, expected)


def test_composite_channels_not_in_colors_are_excluded():
    t, h, w, c = 1, 4, 5, 3
    stack = np.zeros((t, h, w, c), dtype=np.uint8)
    stack[..., 0] = 200  # excluded channel: should have zero contribution
    stack[..., 1] = np.arange(h * w, dtype=np.uint8).reshape(h, w)
    src = THWCSequenceSource(stack)

    view = src.to_rgb(colors={1: "#FF0000"})
    frame = view.get_frame(0).raw

    ch1_gray = normalize_to_uint8(stack[0, ..., 1]).astype(np.float32)
    expected = np.clip(
        ch1_gray[..., None] * np.array([1.0, 0.0, 0.0], dtype=np.float32), 0, 255
    ).astype(np.uint8)
    np.testing.assert_array_equal(frame, expected)


# --- errors ------------------------------------------------------------------


def test_unknown_color_name_raises_value_error():
    src, _ = _src()
    with pytest.raises(ValueError, match="NOTACOLOR"):
        src.to_rgb(colors={0: "NOTACOLOR"})


# --- dtype handling ------------------------------------------------------


def test_uint16_input_scaled_to_uint8():
    src, stack = _src(dtype=np.uint16)
    view = src.to_rgb()

    frame = view.get_frame(0).raw
    assert frame.dtype == np.uint8
    assert frame.shape == (5, 6, 3)

    expected_gray = normalize_to_uint8(stack[0, ..., 0])
    expected = np.stack((expected_gray,) * 3, axis=-1)
    np.testing.assert_array_equal(frame, expected)


# --- laziness --------------------------------------------------------------


def test_laziness_no_frame_read_until_get_frame():
    src, _ = _src()
    counting = _CountingSource(src)

    view = counting.to_rgb()
    assert counting.calls == 0

    view.get_frame(0)
    assert counting.calls == 1

    view.get_frame(2)
    assert counting.calls == 2


def test_laziness_composite_mode():
    src, _ = _src()
    counting = _CountingSource(src)

    view = counting.to_rgb(colors={0: "DAPI", 1: "#00FF00"})
    assert counting.calls == 0

    view.get_frame(0)
    assert counting.calls == 1


# --- calibration delegation --------------------------------------------------


def test_calibration_delegates_to_parent():
    from acia import ureg

    stack = np.arange(2 * 4 * 5 * 1, dtype=np.uint8).reshape(2, 4, 5, 1)
    src = THWCSequenceSource(
        stack,
        frame_interval=2 * ureg.minute,
        pixel_size=0.5 * ureg.micrometer,
    )
    view = src.to_rgb()

    assert view.pixel_size == src.pixel_size
    np.testing.assert_array_equal(view.timepoints.magnitude, src.timepoints.magnitude)
    assert view.timepoints.units == src.timepoints.units


# --- legacy THWCSequenceSource.to_rgb() call ---------------------------------


def test_legacy_thwc_to_rgb_call_still_works():
    t, h, w = 3, 4, 5
    stack = np.arange(t * h * w, dtype=np.uint16).reshape(t, h, w, 1)
    src = THWCSequenceSource(stack)

    view = src.to_rgb()
    assert isinstance(view, RGBSequenceSource)
    assert view.size_t == t

    frame = view.get_frame(0).raw
    assert frame.shape == (h, w, 3)
    assert frame.dtype == np.uint8

    expected_gray = normalize_to_uint8(stack[0, ..., 0])
    expected = np.stack((expected_gray,) * 3, axis=-1)
    np.testing.assert_array_equal(frame, expected)


# --- channel bounds validation -----------------------------------------------


def test_out_of_range_channel_raises_value_error():
    src, _ = _src(c=2)
    view = src.to_rgb(channel=5)
    with pytest.raises(ValueError, match="channel index 5"):
        view.get_frame(0)


def test_negative_channel_raises_value_error():
    src, _ = _src(c=2)
    view = src.to_rgb(channel=-1)
    with pytest.raises(ValueError, match="channel index -1"):
        view.get_frame(0)


def test_out_of_range_colors_key_raises_value_error():
    src, _ = _src(c=2)
    view = src.to_rgb(colors={5: "DAPI"})
    with pytest.raises(ValueError, match="channel index 5"):
        view.get_frame(0)


def test_empty_colors_dict_raises_value_error():
    src, _ = _src()
    with pytest.raises(ValueError, match="colors must not be empty"):
        src.to_rgb(colors={})


# --- real LocalSequenceSource (the reported bug's exact class) --------------


def test_local_sequence_source_to_rgb_reproduces_bug_report(tmp_path):
    """Reproduces the originally reported scenario end-to-end: a real TIFF
    opened via LocalSequenceSource, `.to_rgb()`, fed into `render_tracking_mask`
    (the consumer from the bug report's traceback)."""
    from acia.base import Overlay
    from acia.viz import render_tracking_mask

    path = tmp_path / "seq.tif"
    stack = np.arange(4 * 8 * 10, dtype=np.uint16).reshape(4, 8, 10)
    tifffile.imwrite(str(path), stack)

    source = LocalSequenceSource(str(path), normalize_image=False)
    assert source.size_t == 4
    assert source.num_channels == 1

    source_rgb = source.to_rgb()
    assert isinstance(source_rgb, RGBSequenceSource)

    frame0 = source_rgb.get_frame(0).raw
    assert frame0.shape == (8, 10, 3)
    assert frame0.dtype == np.uint8

    expected_gray = normalize_to_uint8(stack[0])
    expected = np.stack((expected_gray,) * 3, axis=-1)
    np.testing.assert_array_equal(frame0, expected)

    # one dummy mask-based instance so Overlay.timeIterator() has frames to
    # iterate (an entirely empty Overlay is a separate, pre-existing
    # limitation unrelated to to_rgb -- see deferred-work.md)
    from acia.base import Instance

    mask = np.zeros((8, 10), dtype=np.uint16)
    mask[2:4, 2:4] = 1
    overlay = Overlay([Instance(mask=mask, frame=0, label=1, id=1)])
    rendered = render_tracking_mask(source_rgb, overlay)
    assert rendered.size_t >= 1
    assert rendered.get_frame(0).raw.shape == (8, 10, 3)


# --- real CZISequenceSource (cross-source acceptance criterion) -------------


class TestToRgbOnCZISequenceSource(_CZITestBase):
    """Confirms `to_rgb()` works identically on a lazy CZISequenceSource, with
    no subclass-specific code path (spec-to-rgb.md Acceptance Criteria)."""

    dims = "STCYX"
    sizes = {"S": 1, "T": 3, "C": 2, "Y": 6, "X": 7}

    def test_grayscale(self):
        src = self.make_source(position=0)
        view = src.to_rgb(channel=1)
        frame = view.get_frame(0).raw
        assert frame.shape == (6, 7, 3)
        assert frame.dtype == np.uint8

    def test_composite(self):
        src = self.make_source(position=0)
        view = src.to_rgb(colors={0: "DAPI", 1: "#00FF00"})
        frame = view.get_frame(0).raw
        assert frame.shape == (6, 7, 3)
        assert frame.dtype == np.uint8
