"""Tests for crop persistence and training-data capture (acia.crop_capture)."""

import json

import cv2
import numpy as np
import pytest

from acia.base import RotatedCropSpec
from acia.crop_capture import (
    _normalize_uint8,
    load_crop_spec,
    save_crop_capture,
)
from acia.segm.local import THWCSequenceSource


def _src(t=3, h=20, w=24, c=3, dtype=np.uint8):
    stack = np.arange(t * h * w * c, dtype=dtype).reshape(t, h, w, c)
    return THWCSequenceSource(stack), stack


def _spec():
    return RotatedCropSpec(center=(12.0, 10.0), size=(8, 6), angle=30.0)


# --- happy path --------------------------------------------------------------


def test_happy_path_files_and_metadata(tmp_path):
    src, stack = _src()
    spec = _spec()
    result = save_crop_capture(src, spec, tmp_path, frame=0, source_ref="/movie.tif")

    assert result["index"] == 0
    assert result["image"] == tmp_path / "0000.png"
    assert result["json"] == tmp_path / "0000.json"
    assert result["image"].exists()
    assert result["json"].exists()

    png = cv2.imread(str(result["image"]), cv2.IMREAD_UNCHANGED)
    assert png.dtype == np.uint8
    assert png.ndim == 2
    assert png.shape == (stack.shape[1], stack.shape[2])

    data = json.loads(result["json"].read_text(encoding="utf-8"))
    assert data["crop"] == spec.to_dict()
    assert data["box_type"] == "rotated"
    assert data["source"] == "/movie.tif"
    assert data["frame"] == 0
    assert data["image"] == "0000.png"
    assert data["image_shape"] == [stack.shape[1], stack.shape[2]]


# --- enumeration -------------------------------------------------------------


def test_enumeration_increments(tmp_path):
    src, _ = _src()
    spec = _spec()
    r0 = save_crop_capture(src, spec, tmp_path)
    r1 = save_crop_capture(src, spec, tmp_path)
    r2 = save_crop_capture(src, spec, tmp_path)

    assert [r0["index"], r1["index"], r2["index"]] == [0, 1, 2]
    assert (tmp_path / "0000.json").exists()
    assert (tmp_path / "0001.json").exists()
    assert (tmp_path / "0002.json").exists()


# --- provenance --------------------------------------------------------------


def test_in_memory_source_null_provenance(tmp_path):
    src, _ = _src()  # THWCSequenceSource has no filename/imageId
    spec = _spec()
    result = save_crop_capture(src, spec, tmp_path)
    data = json.loads(result["json"].read_text(encoding="utf-8"))
    assert data["source"] is None


def test_auto_detected_filename_provenance(tmp_path):
    src, _ = _src()
    src.filename = "/data/local.tif"  # mimic LocalSequenceSource.filename
    spec = _spec()
    result = save_crop_capture(src, spec, tmp_path)
    data = json.loads(result["json"].read_text(encoding="utf-8"))
    assert data["source"] == "/data/local.tif"


def test_source_ref_overrides_filename(tmp_path):
    src, _ = _src()
    src.filename = "/data/local.tif"
    spec = _spec()
    result = save_crop_capture(src, spec, tmp_path, source_ref="/explicit.tif")
    data = json.loads(result["json"].read_text(encoding="utf-8"))
    assert data["source"] == "/explicit.tif"


# --- normalization -----------------------------------------------------------


def test_uint16_frame_normalized_to_uint8(tmp_path):
    src, _ = _src(dtype=np.uint16)
    spec = _spec()
    result = save_crop_capture(src, spec, tmp_path)
    png = cv2.imread(str(result["image"]), cv2.IMREAD_UNCHANGED)
    assert png.dtype == np.uint8
    assert png.max() == 255  # min-max scaling reaches full range
    assert png.min() == 0


def test_multi_channel_channel_selection(tmp_path):
    # Channel 0 is a gradient (observable scaling); channel 1 is flat (constant).
    h, w, c = 4, 5, 3
    frame = np.zeros((1, h, w, c), dtype=np.uint8)
    frame[0, :, :, 0] = np.arange(h * w).reshape(h, w)
    frame[0, ..., 1] = 200
    src = THWCSequenceSource(frame)
    spec = RotatedCropSpec(center=(2.0, 2.0), size=(2, 2), angle=0.0)
    r0 = save_crop_capture(src, spec, tmp_path, channel=0)
    png0 = cv2.imread(str(r0["image"]), cv2.IMREAD_UNCHANGED)
    assert png0.ndim == 2
    assert png0.shape == (h, w)
    # uint8 gradient channel passes through unchanged (no clip requested).
    np.testing.assert_array_equal(png0, np.arange(h * w).reshape(h, w))

    r1 = save_crop_capture(src, spec, tmp_path, channel=1)
    png1 = cv2.imread(str(r1["image"]), cv2.IMREAD_UNCHANGED)
    assert (png1 == 200).all()  # constant uint8 channel passes through unchanged


def test_flat_frame_all_zero_png(tmp_path):
    frame = np.full((1, 6, 6, 1), 42, dtype=np.uint16)
    src = THWCSequenceSource(frame)
    spec = RotatedCropSpec(center=(3.0, 3.0), size=(2, 2), angle=0.0)
    result = save_crop_capture(src, spec, tmp_path)
    png = cv2.imread(str(result["image"]), cv2.IMREAD_UNCHANGED)
    assert (png == 0).all()


# --- _normalize_uint8 unit behavior -----------------------------------------


def test_normalize_uint8_passthrough_2d():
    arr = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    out = _normalize_uint8(arr)
    np.testing.assert_array_equal(out, arr)
    assert out.dtype == np.uint8


def test_normalize_uint8_clip_percentiles():
    arr = np.arange(100, dtype=np.uint16).reshape(10, 10)
    out = _normalize_uint8(arr, clip_percentiles=(10.0, 90.0))
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255


# --- round-trip --------------------------------------------------------------


def test_round_trip(tmp_path):
    src, _ = _src()
    spec = _spec()
    result = save_crop_capture(src, spec, tmp_path)
    loaded = load_crop_spec(result["json"])
    assert loaded == spec


# --- bad json ----------------------------------------------------------------


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_crop_spec(tmp_path / "nope.json")


def test_load_missing_crop_key_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"box_type": "rotated"}), encoding="utf-8")
    with pytest.raises(KeyError):
        load_crop_spec(bad)


# --- review regressions ------------------------------------------------------


def test_enumeration_beyond_9999_no_overwrite(tmp_path):
    # 5-digit stems must still be counted, else captures past 9999 overwrite 10000
    (tmp_path / "9999.json").write_text("{}", encoding="utf-8")
    (tmp_path / "10000.json").write_text("{}", encoding="utf-8")
    src, _ = _src()
    result = save_crop_capture(src, _spec(), tmp_path)
    assert result["index"] == 10001


def test_imageid_zero_provenance_preserved(tmp_path):
    src, _ = _src()  # no filename
    src.imageId = 0  # a valid OMERO id; must not be dropped as falsy
    result = save_crop_capture(src, _spec(), tmp_path)
    data = json.loads(result["json"].read_text(encoding="utf-8"))
    assert data["source"] == "0"


def test_nonfinite_frame_raises(tmp_path):
    frame = np.zeros((1, 2, 2, 1), dtype=np.float32)  # (T, H, W, C)
    frame[0, 0, 0, 0] = np.nan
    src = THWCSequenceSource(frame)
    with pytest.raises(ValueError):
        save_crop_capture(src, RotatedCropSpec((0.0, 0.0), (1, 1), 0.0), tmp_path)


def test_normalize_rounds_not_truncates():
    # 50 over range [0,100] -> 127.5 -> rounds to 128 (truncation would give 127)
    out = _normalize_uint8(np.array([[0, 50, 100]], dtype=np.uint16))
    np.testing.assert_array_equal(out, np.array([[0, 128, 255]], dtype=np.uint8))


def test_imwrite_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: False)
    src, _ = _src()
    with pytest.raises(OSError):
        save_crop_capture(src, _spec(), tmp_path)
