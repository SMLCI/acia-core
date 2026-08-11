"""Tests for the pint calibration model (time + pixel size) and its propagation
through slicing, overlays and extractors."""

import numpy as np
import pytest

from acia import Q_, ureg
from acia.analysis import AreaEx, ExtractorExecutor, FrameEx, LengthEx, TimeEx
from acia.base import Contour, Overlay
from acia.segm.local import THWCSequenceSource


def _src(t=6, h=8, w=10, c=1, **cal):
    stack = np.zeros((t, h, w, c), dtype=np.uint8)
    return THWCSequenceSource(stack, **cal)


# --- source time calibration -------------------------------------------------


def test_frame_interval_timepoints():
    src = _src(frame_interval=15 * ureg.minute)
    np.testing.assert_array_equal(src.timepoints[:3].magnitude, [0, 15, 30])
    assert str(src.timepoints.units) == "minute"


def test_temporal_slice_scales_interval():
    src = _src(frame_interval=15 * ureg.minute)
    np.testing.assert_array_equal(src[::2].timepoints.magnitude, [0, 30, 60])


def test_explicit_timepoints_roundtrip_and_slice():
    tp = Q_(np.arange(6) * 10.0, "minute")
    src = _src(timepoints=tp)
    np.testing.assert_array_equal(src.timepoints.magnitude, tp.magnitude)
    np.testing.assert_array_equal(src[::2].timepoints.magnitude, [0, 20, 40])


def test_uncalibrated_is_none_and_fluent():
    assert _src().timepoints is None
    np.testing.assert_array_equal(
        _src().with_frame_interval("5 minute").timepoints[:2].magnitude, [0, 5]
    )


# --- source pixel-size calibration ------------------------------------------


def test_pixel_size_string_and_quantity():
    src = _src(pixel_size=0.065 * ureg.micrometer)
    assert src.pixel_size == 0.065 * ureg.micrometer
    assert _src(pixel_size="0.1 micrometer").pixel_size == Q_("0.1 micrometer")


def test_crop_keeps_pixel_size():
    src = _src(pixel_size=0.065 * ureg.micrometer)
    assert src[:, 1:4, 2:6].pixel_size == 0.065 * ureg.micrometer


def test_spatial_step_scales_pixel_size():
    src = _src(pixel_size=0.065 * ureg.micrometer)
    assert src[:, ::2, ::2].pixel_size == 0.13 * ureg.micrometer


# --- overlay temporal slicing + detection timestamps ------------------------


def _overlay(n=10):
    conts = [
        Contour([[0, 0], [2, 0], [2, 2], [0, 2]], -1, frame=f, id=100 + f)
        for f in range(n)
    ]
    return Overlay(conts)


def test_overlay_id_lookup_still_works():
    ov = _overlay()
    assert ov[105].id == 105 and ov[105].frame == 5


def test_overlay_cut_remaps_frames_and_timestamps():
    ov = _overlay().with_frame_interval(15 * ureg.minute)
    cut = ov[:5]
    assert sorted(cut.frames()) == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(cut.timestamps.magnitude, [0, 15, 30, 45, 60])
    # original overlay not mutated
    assert ov[105].frame == 5


def test_overlay_subsample_doubles_interval():
    ov = _overlay().with_frame_interval(15 * ureg.minute)
    np.testing.assert_array_equal(ov[::2].timestamps.magnitude, [0, 30, 60, 90, 120])


def test_detection_time_stamped():
    ov = _overlay().with_frame_interval(15 * ureg.minute)
    assert ov[104].time == 60 * ureg.minute


# --- extractor integration (pull from source) -------------------------------


def _single_contour_overlay(frame=0, id=23):
    # 2x3 rectangle -> area 6 px, longer edge 3 px
    return Overlay([Contour([[0, 0], [2, 0], [2, 3], [0, 3]], -1, frame=frame, id=id)])


def test_extractors_pull_pixel_size_and_interval():
    ps = 0.07
    img = _src(frame_interval=15 * ureg.minute, pixel_size=ps * ureg.micrometer)
    ov = _single_contour_overlay(frame=2)

    df = ExtractorExecutor().execute(
        ov, img, extractors=[FrameEx(), AreaEx(), LengthEx(), TimeEx()]
    )
    assert df["area"].iloc[0] == pytest.approx(6 * ps**2)
    assert df["length"].iloc[0] == pytest.approx(3 * ps)
    assert df["time"].iloc[0] == pytest.approx(2 * 15 / 60)  # hours
    assert df.attrs["units"]["area"] == "micrometer ** 2"


def test_explicit_input_unit_overrides_source():
    img = _src(pixel_size=0.07 * ureg.micrometer)
    ov = _single_contour_overlay()
    df = ExtractorExecutor().execute(
        ov, img, extractors=[AreaEx(input_unit=(1 * ureg.micrometer) ** 2)]
    )
    assert df["area"].iloc[0] == pytest.approx(6.0)  # 1 um/px wins over source 0.07


def test_backward_compatible_explicit_units_no_calibration():
    ps = 0.07
    img = _src()  # no calibration
    ov = _single_contour_overlay()
    df = ExtractorExecutor().execute(
        ov,
        img,
        extractors=[
            FrameEx(),
            AreaEx(input_unit=(ps * ureg.micrometer) ** 2),
            TimeEx(input_unit="15 minute"),
        ],
    )
    assert df["area"].iloc[0] == pytest.approx(6 * ps**2)
    assert df["time"].iloc[0] == pytest.approx(0.0)


def test_timeex_auto_without_time_info_raises():
    img = _src()
    ov = _single_contour_overlay()
    with pytest.raises(ValueError, match="no time information"):
        ExtractorExecutor().execute(ov, img, extractors=[FrameEx(), TimeEx()])
