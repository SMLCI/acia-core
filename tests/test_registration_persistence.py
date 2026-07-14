"""Unit tests for :mod:`acia.registration_persistence` (manifest + ``load_registration``).

Laziness / apply-to-other-file / position-range recovery are exercised with a
fake ``SequenceFile`` (no pixel reads). Round-trip and the fingerprint warning
use a real ``RegistrationManifest`` and a real temp ``.tif``, mirroring
``tests/test_selection.py``'s structure exactly.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from acia.registration import FrameTransform
from acia.registration_persistence import (
    RegistrationManifest,
    RegistrationRecord,
    load_registration,
    save_registration,
)


# --------------------------------------------------------------------------- #
# fakes (assert laziness without pixel IO)
# --------------------------------------------------------------------------- #
class _FakeRegistered:
    def __init__(self, transforms):
        self.transforms = transforms
        self.frames_read = 0

    def get_frame(self, i):  # pragma: no cover - must not be called by load
        self.frames_read += 1
        return None


class _FakeSource:
    def register(self, transforms):
        return _FakeRegistered(transforms)


class _FakeSeqFile:
    def __init__(self, n_positions, path="fake.nd2"):
        self.n = n_positions
        self.path = path
        self.format = "nd2"

    def position(self, i):
        if not 0 <= i < self.n:
            raise ValueError(f"position {i} out of range for {self.n}")
        return _FakeSource()


def _mk_manifest(records_spec, *, source=None):
    """records_spec: list of (position, {frame: (dx, dy, theta)}, failed_frames)."""
    records = [
        RegistrationRecord(
            position=p,
            method="GradientECC",
            transforms={
                frame: FrameTransform(dx=dx, dy=dy, theta=theta)
                for frame, (dx, dy, theta) in transforms.items()
            },
            failed_frames=failed or {},
        )
        for p, transforms, failed in records_spec
    ]
    return RegistrationManifest(
        source=source or {"path": "fake.nd2", "format": "nd2"},
        records=records,
        method="GradientECC",
    )


# --------------------------------------------------------------------------- #
# round-trip
# --------------------------------------------------------------------------- #
class TestRoundTrip(unittest.TestCase):
    def test_frame_transform_roundtrip(self):
        transform = FrameTransform(dx=1.5, dy=-2.25, theta=3.0)
        data = transform.to_dict()
        self.assertEqual(FrameTransform.from_dict(data), transform)

    def test_registration_record_to_from_dict(self):
        record = RegistrationRecord(
            position=3,
            method="GradientECC",
            transforms={
                0: FrameTransform(1.0, 2.0, 0.0),
                5: FrameTransform(3.0, 4.0, 1.5),
            },
            reference_frame=0,
            failed_frames={7: "RegistrationError: no signal"},
            notes="looks good",
        )
        data = record.to_dict()
        json.dumps(data)  # JSON-safe: int keys become str keys
        self.assertEqual(set(data["transforms"].keys()), {"0", "5"})
        self.assertEqual(set(data["failed_frames"].keys()), {"7"})

        record2 = RegistrationRecord.from_dict(data)
        self.assertEqual(record2, record)
        self.assertEqual(set(record2.transforms.keys()), {0, 5})
        self.assertEqual(set(record2.failed_frames.keys()), {7})

    def test_registration_manifest_to_from_dict(self):
        m = _mk_manifest(
            [
                (0, {0: (1.0, 2.0, 0.0), 1: (1.1, 2.1, 0.0)}, None),
                (2, {}, {0: "RegistrationError: blank frame"}),
            ]
        )
        d = m.to_dict()
        json.dumps(d)  # JSON-safe
        self.assertEqual(d["schema"], "acia.registration/v1")
        m2 = RegistrationManifest.from_dict(d)
        self.assertEqual(len(m2.records), 2)
        self.assertEqual(m2.method, "GradientECC")
        self.assertEqual(m2.records[0].transforms[1], FrameTransform(1.1, 2.1, 0.0))
        self.assertEqual(
            m2.records[1].failed_frames[0], "RegistrationError: blank frame"
        )

    def test_save_and_load(self):
        m = _mk_manifest([(0, {0: (1.0, 2.0, 0.0)}, None)])
        with tempfile.TemporaryDirectory() as d:
            path = save_registration(m, d)
            self.assertTrue(Path(path).name == "registration_transforms.json")
            loaded = RegistrationManifest.load(path)
            self.assertEqual(
                loaded.records[0].transforms[0], FrameTransform(1.0, 2.0, 0.0)
            )

    def test_save_creates_missing_directory(self):
        m = _mk_manifest([(0, {0: (1.0, 2.0, 0.0)}, None)])
        with tempfile.TemporaryDirectory() as d:
            nested = Path(d) / "nested" / "output"
            path = save_registration(m, nested)
            self.assertTrue(Path(path).exists())


# --------------------------------------------------------------------------- #
# load_registration (lazy, via fake)
# --------------------------------------------------------------------------- #
class TestLoadRegistration(unittest.TestCase):
    def test_lazy_reconstruction_by_position(self):
        m = _mk_manifest(
            [
                (2, {0: (1.0, 2.0, 0.0), 1: (1.1, 2.1, 0.0)}, None),
                (5, {0: (0.5, 0.5, 0.0)}, {3: "RegistrationError: blank frame"}),
            ]
        )
        fake = _FakeSeqFile(10)
        sources = load_registration(m, source=fake)
        self.assertEqual(set(sources.keys()), {2, 5})
        # no pixel reads happened while reconstructing
        self.assertEqual(sources[2].frames_read, 0)
        self.assertEqual(sources[5].frames_read, 0)
        self.assertEqual(sources[2].transforms[1], FrameTransform(1.1, 2.1, 0.0))

    def test_apply_to_other_file(self):
        m = _mk_manifest([(3, {0: (1.0, 2.0, 0.0)}, None)])
        other = _FakeSeqFile(10)
        sources = load_registration(m, source=other)
        self.assertIn(3, sources)

    def test_position_out_of_range(self):
        m = _mk_manifest([(5, {0: (1.0, 2.0, 0.0)}, None)])
        fake = _FakeSeqFile(2)
        with self.assertRaises(ValueError):
            load_registration(m, source=fake)

    def test_empty_records(self):
        m = RegistrationManifest(source={"path": "x.nd2"}, records=[])
        fake = _FakeSeqFile(2)
        self.assertEqual(load_registration(m, source=fake), {})


# --------------------------------------------------------------------------- #
# real reconstruction + fingerprint warning
# --------------------------------------------------------------------------- #
class TestRealSourceAndFingerprint(unittest.TestCase):
    def _write_tif(self, d):
        path = Path(d) / "stack.tif"
        stack = (np.random.rand(4, 30, 25) * 1000).astype(np.uint16)
        tifffile.imwrite(path, stack)
        return path

    def test_real_lazy_source_from_open_sequence(self):
        from acia.base import RegisteredSequenceSource
        from acia.segm.open import open_sequence

        with tempfile.TemporaryDirectory() as d:
            path = self._write_tif(d)
            seqfile = open_sequence(path)
            m = _mk_manifest(
                [(0, {0: (1.0, 2.0, 0.0)}, None)],
                source={"path": str(path), "format": "tiff"},
            )
            sources = load_registration(m, source=seqfile)
            self.assertIsInstance(sources[0], RegisteredSequenceSource)

    def test_fingerprint_mismatch_warns(self):
        from acia.segm.open import open_sequence

        with tempfile.TemporaryDirectory() as d:
            path = self._write_tif(d)
            seqfile = open_sequence(path)
            m = _mk_manifest(
                [(0, {0: (1.0, 2.0, 0.0)}, None)],
                source={"path": str(path), "fingerprint": {"size": 12345}},
            )
            with self.assertWarns(UserWarning):
                load_registration(m, source=seqfile)


if __name__ == "__main__":
    unittest.main()
