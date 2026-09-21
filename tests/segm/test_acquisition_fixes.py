"""Unit tests for :mod:`acia.segm.acquisition_fixes` (double-shot slip correction).

A fake ``nd2`` module simulates a multi-position acquisition in which the
microscope took one image twice: every raw slot after the slip holds the image of
the slot before it. Each image's pixels encode its *true* ``(t, p)`` and its stage
XY is the true position's, so the tests can check both the correction and the
detection against ground truth.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
import warnings

import numpy as np

from acia.segm.acquisition_fixes import (
    AcquisitionSlip,
    SlotMap,
    detect_acquisition_slips,
    load_fixes,
    save_fixes,
    sidecar_path,
)

T, P = 5, 4
_POS_XY = [(1000.0 * p, 500.0 * p) for p in range(P)]


def _encode(t, p):
    return 100 * t + p


def _raw_slots(slips_flat):
    """The true (t, p) each raw slot holds after double shots at ``slips_flat``."""
    true = [divmod(j, P) for j in range(T * P)]
    raw = []
    j = 0
    for k in range(T * P):
        if k in slips_flat:
            raw.append(raw[-1])  # the repeat of the previous image
        else:
            raw.append(true[j])
            j += 1
    return raw


def _make_fake_nd2(slips_flat=(), *, with_stage=True):
    raw = _raw_slots(set(slips_flat))
    base = np.zeros((P, T, 3, 4), dtype=np.uint16)  # ND2 order P, T, Y, X
    stage = {}
    for k, (tt, pp) in enumerate(raw):
        t_raw, p_raw = divmod(k, P)
        base[p_raw, t_raw] = _encode(tt, pp)
        stage[(t_raw, p_raw)] = _POS_XY[pp]
    sizes = {"P": P, "T": T, "Y": 3, "X": 4}
    # sequence index order of a T-outer, P-inner ND2 experiment
    loops = tuple({"T": t, "P": p} for t in range(T) for p in range(P))

    class _Dask:
        shape = base.shape

        def __getitem__(self, index):
            return base[index]

    class _FakeND2File:
        def __init__(self, path):
            self.path = path
            self.dtype = np.uint16
            self.metadata = types.SimpleNamespace(channels=[])
            self.loop_indices = loops

        @property
        def sizes(self):
            return dict(sizes)

        def voxel_size(self):
            return types.SimpleNamespace(x=0.1, y=0.1, z=1.0)

        def to_dask(self):
            return _Dask()

        def frame_metadata(self, seq):
            if not with_stage:
                raise RuntimeError("no per-frame metadata")
            loop = loops[seq]
            x, y = stage[(loop["T"], loop["P"])]
            pos = types.SimpleNamespace(stagePositionUm=(x, y, 0.0))
            return types.SimpleNamespace(channels=[types.SimpleNamespace(position=pos)])

        def close(self):
            pass

    module = types.ModuleType("nd2")
    module.ND2File = _FakeND2File  # type: ignore[attr-defined]
    return module


class _FakeND2Case(unittest.TestCase):
    slips_flat: tuple = ()
    with_stage = True

    def setUp(self):
        self._saved = sys.modules.get("nd2")
        sys.modules["nd2"] = _make_fake_nd2(self.slips_flat, with_stage=self.with_stage)
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "exp.nd2")
        open(self.path, "wb").close()

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("nd2", None)
        else:
            sys.modules["nd2"] = self._saved
        self._tmp.cleanup()

    def open(self, **kwargs):
        from acia.segm.open import open_sequence

        return open_sequence(self.path, **kwargs)


# --------------------------------------------------------------------------- #
# SlotMap
# --------------------------------------------------------------------------- #
class TestSlotMap(unittest.TestCase):
    def test_no_slips_is_identity(self):
        m = SlotMap(T, P, [])
        for t in range(T):
            for p in range(P):
                self.assertEqual(m.raw_slot(t, p), (t, p))
        self.assertEqual([m.size_t(p) for p in range(P)], [T] * P)

    def test_single_slip_shifts_everything_after(self):
        m = SlotMap(T, P, [(1, 2)])  # raw (1, 2) repeats (1, 1)
        self.assertEqual(m.raw_slot(1, 1), (1, 1))
        self.assertEqual(m.raw_slot(1, 2), (1, 3))
        self.assertEqual(m.raw_slot(1, 3), (2, 0))  # carries into t + 1
        self.assertEqual(m.raw_slot(3, 3), (4, 0))
        self.assertEqual(m.raw_slot(4, 2), (4, 3))
        self.assertIsNone(m.raw_slot(4, 3))  # the lost last image
        self.assertEqual([m.size_t(p) for p in range(P)], [T, T, T, T - 1])

    def test_slips_accumulate(self):
        m = SlotMap(T, P, [(1, 2), (1, 3), (3, 0)])
        raw = _raw_slots({6, 7, 12})
        for k, (tt, pp) in enumerate(raw):
            if k in (6, 7, 12):
                continue
            self.assertEqual(m.raw_slot(tt, pp), divmod(k, P))
        self.assertEqual([m.size_t(p) for p in range(P)], [T, T - 1, T - 1, T - 1])

    def test_slip_at_position_zero(self):
        m = SlotMap(T, P, [(2, 0)])  # repeats (1, P - 1)
        self.assertEqual(m.raw_slot(1, P - 1), (1, P - 1))
        self.assertEqual(m.raw_slot(2, 0), (2, 1))

    def test_accepts_dicts_and_slips(self):
        a = SlotMap(T, P, [{"t": 1, "position": 2}]).slips
        b = SlotMap(T, P, [AcquisitionSlip(1, 2)]).slips
        self.assertEqual(a, b)

    def test_validation(self):
        with self.assertRaises(ValueError):
            SlotMap(T, P, [(0, 0)])
        with self.assertRaises(ValueError):
            SlotMap(T, P, [(T, 0)])
        with self.assertRaises(ValueError):
            SlotMap(T, P, [(1, P)])
        with self.assertRaises(ValueError):
            SlotMap(T, P, [(1, 1), (1, 1)])


# --------------------------------------------------------------------------- #
# SequenceFile integration
# --------------------------------------------------------------------------- #
class TestCorrectedSequenceFile(_FakeND2Case):
    slips_flat = (6,)  # raw (1, 2)

    def _value(self, src, t):
        return int(np.asarray(src.get_frame(t).raw)[0, 0, 0])

    def test_raw_order_is_shifted(self):
        f = self.open(acquisition_fixes=[])
        self.assertEqual(self._value(f.position(2), 1), _encode(1, 1))  # the repeat
        self.assertEqual(self._value(f.position(3), 1), _encode(1, 2))

    def test_fixes_restore_every_true_frame(self):
        f = self.open(acquisition_fixes=[(1, 2)])
        for p in range(P):
            src = f.position(p)
            for t in range(src.size_t):
                frame = src.get_frame(t)
                self.assertEqual(int(np.asarray(frame.raw)[0, 0, 0]), _encode(t, p))
                self.assertEqual(frame.frame, t)
        self.assertEqual(f.position(3).size_t, T - 1)
        with self.assertRaises(IndexError):
            f.position(3).get_frame(T - 1)

    def test_raw_position_stays_raw(self):
        f = self.open(acquisition_fixes=[(1, 2)])
        self.assertEqual(self._value(f.raw_position(2), 1), _encode(1, 1))

    def test_metadata_records_fixes(self):
        f = self.open(acquisition_fixes=[(1, 2)])
        block = f.metadata.to_dict()
        self.assertEqual(block["acquisition_fixes"], [{"t": 1, "position": 2}])
        self.assertEqual(block["num_timepoints"], T)
        json.dumps(block)

    def test_thumbnail_clamps_to_shortened_position(self):
        f = self.open(acquisition_fixes=[(1, 2)])
        self.assertEqual(f.thumbnail(3, downscale=1, frame=T - 1).shape, (3, 4, 3))

    def test_sidecar_is_picked_up_and_overridable(self):
        save_fixes(self.path, [(1, 2)], note="double shot")
        self.assertTrue(os.path.exists(sidecar_path(self.path)))
        self.assertEqual(self.open().acquisition_fixes, [AcquisitionSlip(1, 2)])
        self.assertEqual(self.open(acquisition_fixes=[]).acquisition_fixes, [])
        f = self.open()
        self.assertEqual(self._value(f.position(2), 1), _encode(1, 2))

    def test_sidecar_for_another_file_shape_raises(self):
        save_fixes(self.path, [(1, 2)], num_t=T + 1, num_p=P)
        with self.assertRaises(ValueError):
            _ = self.open().acquisition_fixes
        with self.assertRaises(ValueError):
            load_fixes(self.path, num_t=T, num_p=P)

    def test_no_sidecar_means_no_fixes(self):
        self.assertEqual(load_fixes(self.path), [])
        self.assertEqual(self.open().acquisition_fixes, [])

    def test_fixes_rejected_for_single_position_formats(self):
        from acia.segm.open import open_sequence

        with self.assertRaises(ValueError):
            _ = open_sequence("a.tif", acquisition_fixes=[(1, 0)]).acquisition_fixes

    def test_selection_reopens_with_manifest_fixes(self):
        from acia.base import RotatedCropSpec
        from acia.selection import (
            RoiSelection,
            SelectionManifest,
            load_selection,
            make_source_block,
        )

        f = self.open(acquisition_fixes=[(1, 2)])
        spec = RotatedCropSpec(center=(2.0, 1.5), size=(2, 2), angle=0.0)
        manifest = SelectionManifest(
            source=make_source_block(f),
            roi_mode="single",
            selections=[RoiSelection(position=2, roi=spec)],
        )
        # no sidecar: the manifest alone carries the fixes
        (crop,) = load_selection(manifest)
        self.assertEqual(int(np.asarray(crop.get_frame(1).raw).max()), _encode(1, 2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_selection(manifest, source=self.open(acquisition_fixes=[]))
        self.assertTrue(any("acquisition fixes" in str(w.message) for w in caught))


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
class TestDetection(_FakeND2Case):
    slips_flat = (6, 13)  # raw (1, 2) and (3, 1)

    def test_detects_every_slip_from_stage_xy(self):
        f = self.open(acquisition_fixes=[])
        xy = f.stage_positions()
        self.assertEqual(xy.shape, (T, P, 2))
        self.assertEqual(
            detect_acquisition_slips(f),
            [AcquisitionSlip(1, 2), AcquisitionSlip(3, 1)],
        )

    def test_detected_slips_correct_the_file(self):
        slips = detect_acquisition_slips(self.open(acquisition_fixes=[]))
        f = self.open(acquisition_fixes=slips)
        for p in range(P):
            for t in range(f.position(p).size_t):
                self.assertEqual(f.position(p).stage_xy(t), _POS_XY[p])

    def test_clean_file_has_no_suggestions(self):
        xy = np.array([[_POS_XY[p] for p in range(P)] for _ in range(T)])
        self.assertEqual(detect_acquisition_slips(xy), [])

    def test_positions_too_close_raise(self):
        xy = np.zeros((T, P, 2))
        xy[:, :, 0] = np.arange(P) * 0.5
        with self.assertRaises(ValueError):
            detect_acquisition_slips(xy, tol_um=2.0)

    def test_single_position_raises(self):
        with self.assertRaises(ValueError):
            detect_acquisition_slips(np.zeros((T, 1, 2)))


class TestDetectionWithoutStage(_FakeND2Case):
    with_stage = False

    def test_no_stage_metadata_raises(self):
        f = self.open(acquisition_fixes=[])
        self.assertTrue(np.isnan(f.stage_positions()).all())
        with self.assertRaises(ValueError):
            detect_acquisition_slips(f)


if __name__ == "__main__":
    unittest.main()
