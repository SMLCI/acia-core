"""Python-surface unit tests for ``acia.notebook.ROICropper``.

These tests cover the Python/traitlet surface only (construction, traits,
``fit_to_points`` geometry, ``.spec``/``.cropped()``/``.save()``). The ESM
JavaScript (canvas draw + drag/resize/rotate handles) is NOT exercised here --
it cannot run headless and is validated only by a real Jupyter/Colab/marimo
run.
"""

import numpy as np
import pytest

pytest.importorskip("anywidget")

from acia.base import RotatedCropSpec  # noqa: E402
from acia.notebook import ROICropper  # noqa: E402
from acia.segm.local import THWCSequenceSource  # noqa: E402

T, H, W, C = 3, 40, 60, 3


def _src(t=T, h=H, w=W, c=C):
    stack = np.arange(t * h * w * c, dtype=np.uint8).reshape(t, h, w, c)
    return THWCSequenceSource(stack)


# --- construction ------------------------------------------------------------


def test_construct_defaults_and_image_b64():
    cropper = ROICropper(_src())
    assert cropper.image_b64.startswith("data:image/png;base64,")
    assert cropper.image_w == W
    assert cropper.image_h == H
    assert cropper.center_x == W / 2.0
    assert cropper.center_y == H / 2.0
    assert cropper.angle == 0.0
    assert cropper.width > 0
    assert cropper.height > 0


def test_construct_multi_channel():
    for c in (2, 3):
        cropper = ROICropper(_src(c=c))
        assert cropper.image_b64.startswith("data:image/png;base64,")
        assert cropper.image_b64 != "data:image/png;base64,"


def test_construct_grayscale_single_channel():
    # (T, H, W, 1) -- displayed as RGB; image_b64 non-empty, no error.
    cropper = ROICropper(_src(c=1))
    assert cropper.image_b64.startswith("data:image/png;base64,")
    assert cropper.image_w == W
    assert cropper.image_h == H


def test_construct_explicit_default_size():
    cropper = ROICropper(_src(), width=12, height=8)
    assert cropper.width == 12
    assert cropper.height == 8


# --- spec / cropped / save ---------------------------------------------------


def test_spec_reflects_traits():
    cropper = ROICropper(_src())
    cropper.center_x = 25.0
    cropper.center_y = 18.0
    cropper.width = 10
    cropper.height = 8
    cropper.angle = 22.5
    assert cropper.spec == RotatedCropSpec(
        center=(25.0, 18.0), size=(10, 8), angle=22.5
    )


def test_cropped_size_matches_spec():
    cropper = ROICropper(_src())
    cropper.center_x = 30.0
    cropper.center_y = 20.0
    cropper.width = 14
    cropper.height = 9
    cropper.angle = 15.0
    crop = cropper.cropped()
    assert crop.size_w == 14
    assert crop.size_h == 9


def test_save_writes_capture_files(tmp_path):
    cropper = ROICropper(_src())
    cropper.center_x = 30.0
    cropper.center_y = 20.0
    cropper.width = 14
    cropper.height = 9
    result = cropper.save(tmp_path)
    assert isinstance(result, dict)
    assert (tmp_path / "0000.png").exists()
    assert (tmp_path / "0000.json").exists()


# --- fit_to_points -----------------------------------------------------------


def _rotate(points, deg, center):
    cx, cy = center
    th = np.deg2rad(deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    out = []
    for x, y in points:
        v = rot @ np.array([x - cx, y - cy])
        out.append([v[0] + cx, v[1] + cy])
    return out


def test_fit_to_points_axis_aligned():
    cropper = ROICropper(_src())
    # axis-aligned box: x in [10, 30], y in [20, 50] -> w=20, h=30
    pts = [[10, 20], [30, 20], [30, 50], [10, 50]]
    cropper.fit_to_points(pts)
    assert abs(cropper.angle) < 1e-3
    assert cropper.center_x == pytest.approx(20.0, abs=0.5)
    assert cropper.center_y == pytest.approx(35.0, abs=0.5)
    assert {cropper.width, cropper.height} == {20, 30}


def test_fit_to_points_rotated_30():
    cropper = ROICropper(_src())
    center = (20.0, 35.0)
    corners = [[10, 20], [30, 20], [30, 50], [10, 50]]
    rotated = _rotate(corners, 30, center)
    cropper.fit_to_points(rotated)
    # recovered angle matches the chosen CCW convention (normalized to (-45,45]).
    assert cropper.angle == pytest.approx(30.0, abs=0.5)
    assert cropper.center_x == pytest.approx(20.0, abs=0.5)
    assert cropper.center_y == pytest.approx(35.0, abs=0.5)
    assert {cropper.width, cropper.height} == {20, 30}


def test_fit_to_points_too_few_raises():
    cropper = ROICropper(_src())
    with pytest.raises(ValueError):
        cropper.fit_to_points([[0, 0], [1, 1]])


def test_fit_to_points_degenerate_raises():
    # >=3 collinear points -> cv2.minAreaRect returns a zero-extent rect, which
    # fit_to_points must reject before the max(1, ...) clamp.
    cropper = ROICropper(_src())
    with pytest.raises(ValueError):
        cropper.fit_to_points([[0, 0], [1, 1], [2, 2], [3, 3]])


def test_channel_out_of_range_raises():
    # multi-channel source + channel beyond the available range -> ValueError.
    with pytest.raises(ValueError):
        ROICropper(_src(c=3), channel=99)


def test_points_observer_seeds_box():
    # setting the points trait with >=3 points triggers fit_to_points.
    cropper = ROICropper(_src())
    cropper.points = [[10, 20], [30, 20], [30, 50], [10, 50]]
    assert abs(cropper.angle) < 1e-3
    assert {cropper.width, cropper.height} == {20, 30}


def test_points_observer_ignores_too_few():
    cropper = ROICropper(_src())
    before = (cropper.center_x, cropper.center_y, cropper.width, cropper.height)
    cropper.points = [[0, 0], [1, 1]]  # <3 -> observer ignores, no raise
    after = (cropper.center_x, cropper.center_y, cropper.width, cropper.height)
    assert before == after


# --- end-to-end round-trip: fit_to_points -> crop_rotated straightens ---------


def test_fit_then_crop_straightens():
    """Key sign-correctness test of the fit angle against the real warp.

    Draw a clearly rotated bright rectangle onto a black background, fit the box
    to that rectangle's corners, crop via ``crop_rotated`` and assert the output
    is (a) the expected size and (b) actually axis-aligned -- i.e. the bright
    region fills most of the crop, which only happens if fit_to_points' angle
    sign matches crop_rotated's (OpenCV CCW) convention end-to-end.
    """
    cv2 = pytest.importorskip("cv2")

    img_h, img_w = 200, 240
    frame = np.zeros((img_h, img_w), dtype=np.uint8)

    # A rotated rectangle of known size, rotated ~30 degrees about its center.
    rect_w, rect_h = 100, 60
    center = (img_w / 2.0, img_h / 2.0)
    angle = 30.0
    box = cv2.boxPoints((center, (rect_w, rect_h), angle)).astype(np.float32)
    corners = box.tolist()

    # Fill the rotated rectangle bright (255) on the black frame.
    cv2.fillConvexPoly(frame, box.astype(np.int32), 255)

    # Single-channel (T, H, W, 1) source.
    stack = frame[None, :, :, None]  # (1, H, W, 1)
    src = THWCSequenceSource(stack)

    cropper = ROICropper(src)
    cropper.fit_to_points(corners)

    # The fitted size must match the rectangle (modulo 90-degree swap).
    assert {cropper.width, cropper.height} == {rect_w, rect_h}

    cropped = cropper.cropped().get_frame(0).raw
    out = np.asarray(cropped)
    if out.ndim == 3:
        out = out[..., 0]

    # (a) expected size: (h, w) of the straightened axis-aligned output.
    assert out.shape == (cropper.height, cropper.width)

    # (b) actually axis-aligned: a correctly-straightened crop is almost entirely
    # the bright rectangle. If the angle sign were wrong, the warp would pull in
    # large black wedges, dropping the bright fraction well below this bound.
    bright_fraction = float(np.count_nonzero(out > 127) / out.size)
    assert bright_fraction > 0.95, f"bright_fraction={bright_fraction:.3f}"
