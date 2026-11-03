"""Tests for the rotated-rectangle crop transform and the materialize escape hatch."""

import numpy as np
import pytest

from acia import ureg
from acia.base import (
    ArrayImage,
    BaseImage,
    ImageSequenceSource,
    RotatedCropSpec,
)
from acia.segm.local import THWCSequenceSource


def _src(t=6, h=20, w=24, c=3):
    stack = np.arange(t * h * w * c, dtype=np.uint8).reshape(t, h, w, c)
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


# --- RotatedCropSpec ---------------------------------------------------------


def test_spec_roundtrip_dict():
    spec = RotatedCropSpec(center=(5.0, 6.0), size=(8, 10), angle=30.0)
    data = spec.to_dict()
    assert data == {"center": [5.0, 6.0], "size": [8, 10], "angle": 30.0}
    assert RotatedCropSpec.from_dict(data) == spec


@pytest.mark.parametrize(
    "size", [(0, 5), (5, 0), (-1, 5), (5, -2), (5.0, 5), (5, 5.5), (True, 5)]
)
def test_spec_bad_size_raises(size):
    with pytest.raises(ValueError):
        RotatedCropSpec(center=(1.0, 1.0), size=size, angle=0.0)


# --- crop_rotated: shape / channels ------------------------------------------


def test_happy_path_shape_and_channels():
    src, _ = _src()
    spec = RotatedCropSpec(center=(12.0, 10.0), size=(8, 6), angle=30.0)
    crop = src.crop_rotated(spec)

    assert isinstance(crop, ImageSequenceSource)
    assert crop.size_t == src.size_t
    assert crop.size_h == 6
    assert crop.size_w == 8
    assert crop.num_channels == src.num_channels
    assert crop.size_c == src.size_c

    frame = crop.get_frame(0)
    assert isinstance(frame, BaseImage)
    assert frame.raw.shape == (6, 8, 3)


def test_identity_centered_full_size():
    # smooth gradient image so half-pixel interpolation stays within tolerance
    t, h, w, c = 2, 20, 24, 3
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (yy + xx)[None, ..., None] + np.zeros((t, h, w, c), dtype=np.float32)
    src = THWCSequenceSource(base)
    # center at (w/2, h/2) maps output(u,v) -> input(u,v): an exact identity at angle 0
    spec = RotatedCropSpec(center=(w / 2.0, h / 2.0), size=(w, h), angle=0.0)
    crop = src.crop_rotated(spec)
    # crop ~= original within interpolation tolerance (exact equality won't hold)
    np.testing.assert_allclose(
        crop.get_frame(0).raw.astype(float), base[0].astype(float), atol=1.0
    )


def test_grayscale_input():
    stack = np.arange(3 * 20 * 24, dtype=np.uint8).reshape(3, 20, 24)
    src = _GraySource(stack)
    spec = RotatedCropSpec(center=(12.0, 10.0), size=(8, 6), angle=15.0)
    crop = src.crop_rotated(spec)
    frame = crop.get_frame(0).raw
    assert frame.shape == (6, 8)
    assert crop.num_channels == 1


def test_many_channels():
    t, h, w, c = 2, 20, 24, 7
    stack = np.arange(t * h * w * c, dtype=np.uint8).reshape(t, h, w, c)
    src = THWCSequenceSource(stack)
    spec = RotatedCropSpec(center=(12.0, 10.0), size=(8, 6), angle=20.0)
    crop = src.crop_rotated(spec)
    frame = crop.get_frame(0).raw
    assert frame.shape == (6, 8, 7)
    assert crop.num_channels == 7


def test_angle0_integer_translation_is_exact_crop():
    # angle 0 with an integer center offset is a pure pixel shift, so the crop
    # equals an exact numpy slice (no interpolation) -- a strong value check.
    src, stack = _src()  # (6, 20, 24, 3)
    spec = RotatedCropSpec(center=(12.0, 10.0), size=(8, 6), angle=0.0)
    crop = src.crop_rotated(spec)
    # output[v, u] = input[v + (cy - h/2), u + (cx - w/2)] = input[v + 7, u + 8]
    np.testing.assert_array_equal(crop.get_frame(0).raw, stack[0, 7:13, 8:16, :])


def test_single_channel_with_axis_preserved():
    # a (H, W, 1) parent must keep its channel axis after cropping (regression:
    # cv2.warpAffine collapses a trailing singleton channel)
    stack = np.arange(2 * 20 * 24 * 1, dtype=np.uint8).reshape(2, 20, 24, 1)
    src = THWCSequenceSource(stack)
    spec = RotatedCropSpec(center=(12.0, 10.0), size=(8, 6), angle=15.0)
    crop = src.crop_rotated(spec)
    assert crop.get_frame(0).raw.shape == (6, 8, 1)
    assert crop.size_c == 1
    assert crop.materialize().get_frame(0).raw.shape == (6, 8, 1)


def test_region_partly_off_frame_zero_border():
    # +1 offset so no input pixel is 0; any 0 in the output is genuine border fill
    src = THWCSequenceSource(
        np.arange(1, 6 * 20 * 24 * 3 + 1, dtype=np.uint16).reshape(6, 20, 24, 3)
    )
    spec = RotatedCropSpec(center=(1.0, 1.0), size=(10, 10), angle=45.0)
    crop = src.crop_rotated(spec)
    frame = crop.get_frame(0).raw
    assert frame.shape == (10, 10, 3)
    assert (frame == 0).any()  # zeros can only come from out-of-bounds border fill


def test_materialize_empty_source_raises():
    # np.stack on an empty list would give an opaque error; we raise a clear one
    src, _ = _src()
    empty = src[0:0]
    assert empty.size_t == 0
    with pytest.raises(ValueError):
        empty.materialize()


# --- calibration -------------------------------------------------------------


def test_calibration_preserved():
    t, h, w, c = 4, 20, 24, 3
    stack = np.arange(t * h * w * c, dtype=np.uint8).reshape(t, h, w, c)
    src = THWCSequenceSource(
        stack,
        frame_interval=2 * ureg.minute,
        pixel_size=0.5 * ureg.micrometer,
    )
    spec = RotatedCropSpec(center=(12.0, 10.0), size=(8, 6), angle=10.0)
    crop = src.crop_rotated(spec)

    assert crop.pixel_size == src.pixel_size
    np.testing.assert_array_equal(crop.timepoints.magnitude, src.timepoints.magnitude)
    assert crop.timepoints.units == src.timepoints.units


# --- composition -------------------------------------------------------------


def test_reslicing_crop_composes():
    src, _ = _src()
    spec = RotatedCropSpec(center=(12.0, 10.0), size=(8, 6), angle=30.0)
    crop = src.crop_rotated(spec)
    sub = crop[::2]
    assert isinstance(sub, ImageSequenceSource)
    assert sub.size_t == 3
    np.testing.assert_array_equal(sub.get_frame(1).raw, crop.get_frame(2).raw)


# --- materialize -------------------------------------------------------------


def test_materialize_data_and_calibration():
    t, h, w, c = 4, 20, 24, 3
    stack = np.arange(t * h * w * c, dtype=np.uint8).reshape(t, h, w, c)
    src = THWCSequenceSource(
        stack,
        frame_interval=3 * ureg.second,
        pixel_size=0.25 * ureg.micrometer,
    )
    spec = RotatedCropSpec(center=(12.0, 10.0), size=(8, 6), angle=22.0)
    crop = src.crop_rotated(spec)

    mat = crop.materialize()
    assert isinstance(mat, THWCSequenceSource)
    assert mat is not crop
    assert not hasattr(mat, "parent")

    assert mat.size_t == crop.size_t
    assert mat.size_h == crop.size_h
    assert mat.size_w == crop.size_w
    assert mat.size_c == crop.size_c

    for i in range(crop.size_t):
        np.testing.assert_array_equal(mat.get_frame(i).raw, crop.get_frame(i).raw)

    assert mat.pixel_size == crop.pixel_size
    np.testing.assert_array_equal(mat.timepoints.magnitude, crop.timepoints.magnitude)
    assert mat.timepoints.units == crop.timepoints.units


def test_materialize_grayscale_normalizes_channel_axis():
    stack = np.arange(3 * 20 * 24, dtype=np.uint8).reshape(3, 20, 24)
    src = _GraySource(stack)
    spec = RotatedCropSpec(center=(12.0, 10.0), size=(8, 6), angle=0.0)
    crop = src.crop_rotated(spec)

    mat = crop.materialize()
    assert isinstance(mat, THWCSequenceSource)
    assert mat.get_frame(0).raw.shape == (6, 8, 1)
    np.testing.assert_array_equal(mat.get_frame(0).raw[..., 0], crop.get_frame(0).raw)


@pytest.mark.parametrize("channels", [None, 1, 3, 6])
def test_warp_rotated_crop_matches_the_lazy_view(channels):
    """The frame-level helper and the lazy view must warp identically.

    ``export_rois`` crops one already-read frame for several ROIs through
    :func:`acia.base.warp_rotated_crop`; the lazy view's own ``get_frame`` is
    what every other caller uses. One implementation, so they cannot drift --
    including over the >4-channel path, which OpenCV cannot warp in one call.
    """
    from acia.base import warp_rotated_crop

    rng = np.random.default_rng(3)
    shape = (20, 24) if channels is None else (20, 24, channels)
    frame = rng.integers(0, 255, shape, dtype=np.uint16)
    spec = RotatedCropSpec(center=(11.0, 9.0), size=(8, 6), angle=25.0)

    stack = frame[None] if channels is not None else frame[None, ..., None]
    source = THWCSequenceSource(stack)
    expected = source.crop_rotated(spec).get_frame(0).raw

    got = warp_rotated_crop(frame, spec)
    assert got.shape[:2] == (6, 8)
    np.testing.assert_array_equal(
        got if got.ndim == expected.ndim else got[..., None], expected
    )


# --------------------------------------------------------------------------- #
# cropping a drift-corrected view: one warp, not two
# --------------------------------------------------------------------------- #
def _smooth_stack(t=4, h=200, w=200):
    """Smooth, correlated content -- what a microscopy frame actually looks like."""
    yy, xx = np.mgrid[0:h, 0:w]
    frames = [
        (
            30000 + 25000 * np.sin((xx + 3 * i) / 17.0) * np.cos((yy - 2 * i) / 23.0)
        ).astype(np.uint16)
        for i in range(t)
    ]
    return np.stack(frames)[..., None]


def _registered(stack, transform, *, on_missing="warn"):
    from acia.registration import FrameTransform

    src = THWCSequenceSource(stack)
    transforms = {i: transform for i in range(stack.shape[0])}
    assert isinstance(transform, FrameTransform)
    return src, src.register(transforms, on_missing=on_missing)


def _two_step(raw_source, registered, spec, frame):
    """What the previous pipeline did: correct the whole frame, then crop it."""
    from acia.base import warp_rotated_crop
    from acia.registration import apply_correction

    raw = np.asarray(raw_source.get_frame(frame).raw)
    corrected = apply_correction(raw, registered.transforms[frame])
    return warp_rotated_crop(corrected, spec)


def test_registered_crop_is_a_single_warp():
    from acia.base import RegisteredRotatedCropSource
    from acia.registration import FrameTransform

    stack = _smooth_stack()
    _, registered = _registered(stack, FrameTransform(dx=6.0, dy=-3.5, theta=0.8))
    spec = RotatedCropSpec(center=(110.0, 90.0), size=(40, 32), angle=15.0)
    crop = registered.crop_rotated(spec)

    assert isinstance(crop, RegisteredRotatedCropSource)
    # Shapes and calibration are unchanged from an ordinary crop view.
    assert crop.size_t == stack.shape[0]
    assert crop.size_h == 32
    assert crop.size_w == 40
    assert crop.get_frame(0).raw.shape == (32, 40, 1)


def test_registered_crop_matches_correcting_then_cropping():
    """The composition is exact, so the crop is the same region -- just sharper."""
    from acia.registration import FrameTransform

    stack = _smooth_stack()
    raw_source, registered = _registered(
        stack, FrameTransform(dx=6.0, dy=-3.5, theta=0.8)
    )
    spec = RotatedCropSpec(center=(110.0, 90.0), size=(40, 32), angle=15.0)

    for frame in range(stack.shape[0]):
        one = np.asarray(registered.crop_rotated(spec).get_frame(frame).raw)[..., 0]
        two = _two_step(raw_source, registered, spec, frame)[..., 0]
        # Differences are the second interpolation, not a different region:
        # well under a percent of the data range on smooth content.
        spread = float(stack.max() - stack.min())
        assert np.abs(one.astype(float) - two.astype(float)).max() < 0.02 * spread


def test_registered_crop_keeps_the_region_to_sub_pixel_accuracy():
    """A blob's centroid lands in the same place through either path."""
    from acia.registration import FrameTransform

    h = w = 200
    yy, xx = np.mgrid[0:h, 0:w]
    blob = (
        60000 * np.exp(-((xx - 110.0) ** 2 + (yy - 90.0) ** 2) / (2 * 6.0**2))
    ).astype(np.uint16)
    stack = blob[None, ..., None]
    raw_source, registered = _registered(
        stack, FrameTransform(dx=6.0, dy=-3.5, theta=0.8)
    )
    spec = RotatedCropSpec(center=(110.0, 90.0), size=(60, 60), angle=15.0)

    def centroid(img):
        a = img.astype(np.float64)
        gy, gx = np.mgrid[0 : a.shape[0], 0 : a.shape[1]]
        return (gx * a).sum() / a.sum(), (gy * a).sum() / a.sum()

    one = np.asarray(registered.crop_rotated(spec).get_frame(0).raw)[..., 0]
    two = _two_step(raw_source, registered, spec, 0)[..., 0]
    cx1, cy1 = centroid(one)
    cx2, cy2 = centroid(two)
    assert abs(cx1 - cx2) < 0.05
    assert abs(cy1 - cy2) < 0.05
    # One resample keeps more of the peak than two do -- that is the point.
    assert one.max() >= two.max()


def test_an_identity_transform_crops_exactly_like_an_uncorrected_source():
    from acia.registration import FrameTransform

    stack = _smooth_stack()
    raw_source, registered = _registered(
        stack, FrameTransform(dx=0.0, dy=0.0, theta=0.0)
    )
    spec = RotatedCropSpec(center=(110.0, 90.0), size=(40, 32), angle=0.0)

    np.testing.assert_allclose(
        np.asarray(registered.crop_rotated(spec).get_frame(1).raw).astype(float),
        np.asarray(raw_source.crop_rotated(spec).get_frame(1).raw).astype(float),
        atol=1,
    )


def test_spec_for_round_trips_through_the_transform():
    from acia.registration import FrameTransform, apply_correction_to_spec

    stack = _smooth_stack()
    transform = FrameTransform(dx=6.0, dy=-3.5, theta=0.8)
    _, registered = _registered(stack, transform)
    spec = RotatedCropSpec(center=(110.0, 90.0), size=(40, 32), angle=15.0)

    in_frame = registered.spec_for(2, spec)
    back = apply_correction_to_spec(
        in_frame, transform, shape=(registered.size_h, registered.size_w)
    )
    assert abs(back.center[0] - spec.center[0]) < 1e-6
    assert abs(back.center[1] - spec.center[1]) < 1e-6
    assert abs(back.angle - spec.angle) < 1e-9


def test_a_missing_transform_is_still_recorded_when_cropping():
    """The composed crop must not lose the missing-frame bookkeeping."""
    import pytest

    from acia.registration import FrameTransform

    stack = _smooth_stack(t=4)
    src = THWCSequenceSource(stack)
    transform = FrameTransform(dx=6.0, dy=-3.5, theta=0.8)
    registered = src.register(
        {0: transform, 1: transform, 3: transform}, on_missing="nearest"
    )
    spec = RotatedCropSpec(center=(110.0, 90.0), size=(40, 32), angle=15.0)
    crop = registered.crop_rotated(spec)

    with pytest.warns(UserWarning, match="no stored correction"):
        crop.get_frame(2)
    assert registered.missing_frames == {2}

    # ... and the policy itself still applies.
    strict = src.register({0: transform}, on_missing="error")
    with pytest.raises(KeyError):
        strict.crop_rotated(spec).get_frame(1)
