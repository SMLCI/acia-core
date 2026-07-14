"""Unit tests for :mod:`acia.selection` (manifest + ``load_selection``).

Laziness / apply-to-other-file / position-range / label recovery are exercised
with a fake ``SequenceFile`` (no pixel reads). Round-trip and the fingerprint
warning use a real ``SelectionManifest`` and a real temp ``.tif``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from acia import ureg
from acia.base import RotatedCropSpec
from acia.selection import (
    RoiSelection,
    SelectionManifest,
    load_selection,
    save_selection,
)


# --------------------------------------------------------------------------- #
# fakes (assert laziness without pixel IO)
# --------------------------------------------------------------------------- #
class _FakeCropped:
    def __init__(self, spec, pixel_size):
        self._spec = spec
        self._pixel_size = pixel_size
        self.frames_read = 0

    @property
    def size_w(self):
        return self._spec.size[0]

    @property
    def size_h(self):
        return self._spec.size[1]

    @property
    def pixel_size(self):
        return self._pixel_size

    def get_frame(self, i):  # pragma: no cover - must not be called by load
        self.frames_read += 1
        return None


class _FakeSource:
    def __init__(self, pixel_size):
        self._pixel_size = pixel_size

    def crop_rotated(self, spec):
        return _FakeCropped(spec, self._pixel_size)


class _FakeSeqFile:
    def __init__(self, n_positions, pixel_size, path="fake.nd2"):
        self.n = n_positions
        self._pixel_size = pixel_size
        self.path = path
        self.format = "nd2"

    def position(self, i):
        if not 0 <= i < self.n:
            raise ValueError(f"position {i} out of range for {self.n}")
        return _FakeSource(self._pixel_size)


def _mk_manifest(specs, *, roi_mode="multi", source=None):
    """specs: list of (position, (cx,cy), (w,h), angle, label)."""
    sels = [
        RoiSelection(
            position=p,
            roi=RotatedCropSpec(center=c, size=s, angle=a),
            label=lbl,
            id=f"sel{i}",
        )
        for i, (p, c, s, a, lbl) in enumerate(specs)
    ]
    return SelectionManifest(
        source=source or {"path": "fake.nd2", "format": "nd2", "pixel_size_um": 0.07},
        selections=sels,
        roi_mode=roi_mode,
    )


# --------------------------------------------------------------------------- #
# round-trip
# --------------------------------------------------------------------------- #
class TestRoundTrip(unittest.TestCase):
    def test_to_from_dict(self):
        m = _mk_manifest(
            [
                (37, (50.0, 60.0), (20, 16), 12.5, "colony A"),
                (52, (30.0, 40.0), (14, 14), 0.0, "colony A"),
            ]
        )
        d = m.to_dict()
        json.dumps(d)  # JSON-safe
        m2 = SelectionManifest.from_dict(d)
        self.assertEqual(len(m2.selections), 2)
        self.assertEqual(m2.roi_mode, "multi")
        self.assertEqual(m2.selections[0].spec.size, (20, 16))
        self.assertEqual(m2.selections[0].label, "colony A")

    def test_save_and_load(self):
        m = _mk_manifest([(0, (25.0, 30.0), (20, 16), 10.0, "A")])
        with tempfile.TemporaryDirectory() as d:
            path = save_selection(m, d)
            self.assertTrue(Path(path).name == "selection.json")
            loaded = SelectionManifest.load(path)
            self.assertEqual(loaded.selections[0].spec.size, (20, 16))

    def test_save_writes_previews(self):
        m = _mk_manifest([(0, (25.0, 30.0), (20, 16), 10.0, "A")])
        with tempfile.TemporaryDirectory() as d:
            save_selection(m, d, previews={"sel0": b"\x89PNG\r\n\x1a\n"})
            self.assertTrue((Path(d) / "previews" / "sel0.png").exists())


# --------------------------------------------------------------------------- #
# load_selection (lazy, via fake)
# --------------------------------------------------------------------------- #
class TestLoadSelection(unittest.TestCase):
    def test_lazy_and_size_and_label(self):
        m = _mk_manifest(
            [
                (37, (50.0, 60.0), (20, 16), 0.0, "colony A"),
                (37, (30.0, 40.0), (14, 14), 0.0, "colony B"),
            ]
        )
        fake = _FakeSeqFile(108, ureg.Quantity(0.0733, "micrometer"))
        crops = load_selection(m, source=fake)
        self.assertEqual(len(crops), 2)
        # sizes come from the ROI specs
        self.assertEqual((crops[0].size_w, crops[0].size_h), (20, 16))
        self.assertEqual((crops[1].size_w, crops[1].size_h), (14, 14))
        # no pixel reads happened while reconstructing
        self.assertEqual(crops[0].frames_read, 0)
        # calibration from the source; label recovered
        self.assertEqual(crops[0].pixel_size, ureg.Quantity(0.0733, "micrometer"))
        self.assertEqual(crops[0].label, "colony A")
        self.assertEqual(crops[0].selection.id, "sel0")

    def test_apply_to_other_file(self):
        m = _mk_manifest([(3, (50.0, 60.0), (20, 16), 0.0, "A")])
        other = _FakeSeqFile(10, ureg.Quantity(0.5, "micrometer"))
        crops = load_selection(m, source=other)
        self.assertEqual(crops[0].pixel_size, ureg.Quantity(0.5, "micrometer"))

    def test_position_out_of_range(self):
        m = _mk_manifest([(5, (50.0, 60.0), (20, 16), 0.0, "A")])
        fake = _FakeSeqFile(2, ureg.Quantity(0.07, "micrometer"))
        with self.assertRaises(ValueError):
            load_selection(m, source=fake)

    def test_empty_selections(self):
        m = SelectionManifest(source={"path": "x.nd2"}, selections=[])
        fake = _FakeSeqFile(2, ureg.Quantity(0.07, "micrometer"))
        self.assertEqual(load_selection(m, source=fake), [])


# --------------------------------------------------------------------------- #
# real crop + fingerprint warning
# --------------------------------------------------------------------------- #
class TestRealCropAndFingerprint(unittest.TestCase):
    def _write_tif(self, d):
        path = Path(d) / "stack.tif"
        stack = (np.random.rand(2, 60, 50) * 1000).astype(np.uint16)
        tifffile.imwrite(path, stack)
        return path

    def test_real_lazy_crop_from_open_sequence(self):
        from acia.segm.open import open_sequence

        with tempfile.TemporaryDirectory() as d:
            path = self._write_tif(d)
            seqfile = open_sequence(path)
            m = _mk_manifest(
                [(0, (25.0, 30.0), (20, 16), 10.0, "A")],
                source={"path": str(path), "format": "tiff"},
            )
            crops = load_selection(m, source=seqfile)
            self.assertEqual((crops[0].size_h, crops[0].size_w), (16, 20))

    def test_fingerprint_mismatch_warns(self):
        from acia.segm.open import open_sequence

        with tempfile.TemporaryDirectory() as d:
            path = self._write_tif(d)
            seqfile = open_sequence(path)
            m = _mk_manifest(
                [(0, (25.0, 30.0), (20, 16), 0.0, "A")],
                source={"path": str(path), "fingerprint": {"size": 12345}},
            )
            with self.assertWarns(UserWarning):
                load_selection(m, source=seqfile)


if __name__ == "__main__":
    unittest.main()
