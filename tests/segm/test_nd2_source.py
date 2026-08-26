"""Unit tests for :class:`acia.segm.nd2_source.ND2SequenceSource`.

These tests never import the real ``nd2`` package or open a real ``.nd2`` file.
A fake ``nd2`` module is injected into ``sys.modules`` (mirroring the
``sys.modules``-fake pattern in ``tests/segm/test_release.py``). The fake's
``to_dask()`` returns a *recording* array that tracks every index and every
materialization, so we can assert laziness: constructing the source and reading
``size_*``/``pixel_size`` must compute no frame, and ``get_frame(i)`` must
compute exactly the single ``[position, i]`` slice (never the whole array).
"""

from __future__ import annotations

import sys
import types
import unittest

import numpy as np

from acia import ureg


class _RecordingSlice:
    """A sub-array returned by ``_RecordingDask[index]`` that records compute."""

    def __init__(self, base: np.ndarray, index, record: dict) -> None:
        self._base = base
        self._index = index
        self._record = record

    def __array__(self, dtype=None):
        # np.asarray(slice) materializes ONLY this slice; record the compute.
        self._record["computes"].append(self._index)
        result = self._base[self._index]
        if dtype is not None:
            result = result.astype(dtype)
        return result


class _RecordingDask:
    """A dask-like array that records every __getitem__ index and every compute.

    Backed by a full numpy array, but ``np.asarray(self)`` (the whole array) is
    forbidden so the test fails loudly if the implementation ever materializes
    everything at once.
    """

    def __init__(self, base: np.ndarray, record: dict) -> None:
        self._base = base
        self._record = record
        self.shape = base.shape

    def __getitem__(self, index) -> _RecordingSlice:
        self._record["indices"].append(index)
        return _RecordingSlice(self._base, index, self._record)

    def __array__(self, dtype=None):  # pragma: no cover - must never be called
        raise AssertionError(
            "np.asarray() was called on the whole dask array (forbidden)"
        )


class _FakeVoxel:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


def _make_fake_nd2(sizes: dict[str, int], voxel_x: float = 0.65):
    """Build a fake ``nd2`` module whose ND2File exposes the given ``sizes``.

    The backing numpy array is ``np.arange(prod(sizes)).reshape(sizes order)``
    so individual planes can be compared against the expected slice. A shared
    ``record`` dict tracks indices/computes/close for laziness assertions.
    """
    record: dict = {"indices": [], "computes": [], "closed": 0}
    shape = tuple(sizes.values())
    base = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)

    class _FakeND2File:
        def __init__(self, path: str) -> None:
            self.path = path

        @property
        def sizes(self) -> dict[str, int]:
            return dict(sizes)

        def voxel_size(self) -> _FakeVoxel:
            return _FakeVoxel(voxel_x, voxel_x, 1.0)

        def to_dask(self) -> _RecordingDask:
            return _RecordingDask(base, record)

        def close(self) -> None:
            record["closed"] += 1

    module = types.ModuleType("nd2")
    module.ND2File = _FakeND2File  # type: ignore[attr-defined]
    return module, record, base


class _ND2TestBase(unittest.TestCase):
    """Installs/removes a fake ``nd2`` module around each test."""

    sizes: dict[str, int] = {"P": 2, "T": 5, "C": 2, "Y": 8, "X": 10}
    voxel_x: float = 0.65

    def setUp(self) -> None:
        self._saved_nd2 = sys.modules.get("nd2")
        self.module, self.record, self.base = _make_fake_nd2(self.sizes, self.voxel_x)
        sys.modules["nd2"] = self.module

    def tearDown(self) -> None:
        if self._saved_nd2 is None:
            sys.modules.pop("nd2", None)
        else:
            sys.modules["nd2"] = self._saved_nd2

    def make_source(self, **kwargs):
        from acia.segm.nd2_source import ND2SequenceSource

        return ND2SequenceSource("fake.nd2", **kwargs)


class TestShapeMapping(_ND2TestBase):
    """Matrix row: single position TC-series {P:2,T:5,C:2,Y:8,X:10}, position=1."""

    sizes = {"P": 2, "T": 5, "C": 2, "Y": 8, "X": 10}

    def test_sizes(self) -> None:
        src = self.make_source(position=1)
        self.assertEqual(src.size_t, 5)
        self.assertEqual(src.size_h, 8)
        self.assertEqual(src.size_w, 10)
        self.assertEqual(src.size_c, 2)
        self.assertEqual(src.num_channels, 2)

    def test_get_frame_shape_and_content(self) -> None:
        src = self.make_source(position=1)
        frame = src.get_frame(0)
        self.assertEqual(frame.raw.shape, (8, 10, 2))
        # P=1, T=0 plane, channel moved last -> expected (Y, X, C)
        expected = np.moveaxis(self.base[1, 0], 0, -1)  # base axes P,T,C,Y,X
        np.testing.assert_array_equal(frame.raw, expected)
        self.assertEqual(frame.frame, 0)

    def test_get_frame_selects_position_and_time(self) -> None:
        src = self.make_source(position=1)
        f2 = src.get_frame(2)
        expected = np.moveaxis(self.base[1, 2], 0, -1)
        np.testing.assert_array_equal(f2.raw, expected)


class TestNoPAxis(_ND2TestBase):
    """Matrix row: no P axis -> single position; size_t==4, single channel."""

    sizes = {"T": 4, "C": 1, "Y": 6, "X": 6}

    def test_single_position(self) -> None:
        src = self.make_source()  # default position=0
        self.assertEqual(src.size_t, 4)
        frame = src.get_frame(0)
        self.assertEqual(frame.raw.shape, (6, 6, 1))
        expected = np.moveaxis(self.base[0], 0, -1)  # base axes T,C,Y,X -> select T=0
        np.testing.assert_array_equal(frame.raw, expected)


class TestGrayscaleNoC(_ND2TestBase):
    """Matrix row: grayscale (no C) -> size_c==1, frame shaped (H,W,1)."""

    sizes = {"T": 3, "Y": 5, "X": 5}

    def test_channel_axis_added(self) -> None:
        src = self.make_source()
        self.assertEqual(src.size_c, 1)
        self.assertEqual(src.num_channels, 1)
        frame = src.get_frame(1)
        self.assertEqual(frame.raw.shape, (5, 5, 1))
        expected = self.base[1][..., np.newaxis]  # base axes T,Y,X -> select T=1
        np.testing.assert_array_equal(frame.raw, expected)


class TestCalibration(_ND2TestBase):
    """Matrix rows: calibration from metadata and user override."""

    sizes = {"P": 2, "T": 5, "C": 2, "Y": 8, "X": 10}
    voxel_x = 0.65

    def test_pixel_size_from_metadata(self) -> None:
        src = self.make_source(position=0)
        self.assertEqual(src.pixel_size, ureg.Quantity(0.65, "micrometer"))

    def test_pixel_size_user_override(self) -> None:
        src = self.make_source(position=0, pixel_size="0.5 um")
        self.assertEqual(src.pixel_size, ureg.Quantity(0.5, "micrometer"))

    def test_frame_interval_override_builds_timepoints(self) -> None:
        src = self.make_source(position=0, frame_interval="30 s")
        tps = src.timepoints
        self.assertIsNotNone(tps)
        np.testing.assert_array_equal(tps.magnitude, np.arange(5) * 30.0)
        self.assertEqual(str(tps.units), "second")

    def test_no_timing_metadata_is_none(self) -> None:
        # No user timing and no derivable metadata -> timepoints None.
        src = self.make_source(position=0)
        self.assertIsNone(src.timepoints)


class TestValidation(_ND2TestBase):
    def test_position_out_of_range(self) -> None:
        src = self.make_source(position=5)  # only 2 positions
        with self.assertRaises(ValueError):
            _ = src.size_t

    def test_z_stack_rejected(self) -> None:
        self.sizes = {"P": 1, "T": 2, "Z": 7, "C": 1, "Y": 4, "X": 4}
        self.module, self.record, self.base = _make_fake_nd2(self.sizes, self.voxel_x)
        sys.modules["nd2"] = self.module
        src = self.make_source(position=0)
        with self.assertRaises(ValueError):
            _ = src.size_t

    def test_z_size_one_squeezed(self) -> None:
        self.sizes = {"T": 2, "Z": 1, "C": 1, "Y": 4, "X": 4}
        self.module, self.record, self.base = _make_fake_nd2(self.sizes, self.voxel_x)
        sys.modules["nd2"] = self.module
        src = self.make_source(position=0)
        frame = src.get_frame(0)
        self.assertEqual(frame.raw.shape, (4, 4, 1))
        # base axes T,Z,C,Y,X -> select T=0,Z=0; C moved last
        expected = np.moveaxis(self.base[0, 0], 0, -1)
        np.testing.assert_array_equal(frame.raw, expected)


class TestNd2NotInstalled(unittest.TestCase):
    """Matrix row: nd2 not importable -> ImportError with the pip hint."""

    def setUp(self) -> None:
        self._saved_nd2 = sys.modules.get("nd2")
        # Make `import nd2` fail.
        sys.modules["nd2"] = None  # type: ignore[assignment]

    def tearDown(self) -> None:
        if self._saved_nd2 is None:
            sys.modules.pop("nd2", None)
        else:
            sys.modules["nd2"] = self._saved_nd2

    def test_import_error_message(self) -> None:
        from acia.segm.nd2_source import ND2SequenceSource

        src = ND2SequenceSource("fake.nd2")
        with self.assertRaises(ImportError) as ctx:
            _ = src.size_t
        self.assertIn("pip install acia[nd2]", str(ctx.exception))


class TestLaziness(_ND2TestBase):
    """Matrix row: laziness — no frame computed on construct/metadata; one per get_frame."""

    sizes = {"P": 2, "T": 5, "C": 2, "Y": 8, "X": 10}

    def test_construction_and_metadata_compute_nothing(self) -> None:
        src = self.make_source(position=1)
        # constructing the source must not even open the reader
        self.assertEqual(self.record["indices"], [])
        self.assertEqual(self.record["computes"], [])

        # reading sizes/pixel_size opens the reader but materializes no pixels
        _ = (src.size_t, src.size_h, src.size_w, src.size_c, src.pixel_size)
        self.assertEqual(self.record["computes"], [])

    def test_get_frame_computes_exactly_one_slice(self) -> None:
        src = self.make_source(position=1)
        # warm the reader via metadata
        _ = src.size_t
        self.assertEqual(self.record["computes"], [])

        src.get_frame(2)
        self.assertEqual(len(self.record["computes"]), 1)
        index = self.record["computes"][0]
        # axes order P,T,C,Y,X -> P=1, T=2, C/Y/X kept
        self.assertEqual(index, (1, 2, slice(None), slice(None), slice(None)))

        # a second frame computes exactly one more slice, never the whole array
        src.get_frame(4)
        self.assertEqual(len(self.record["computes"]), 2)
        self.assertEqual(
            self.record["computes"][1], (1, 4, slice(None), slice(None), slice(None))
        )

    def test_close_called_on_del(self) -> None:
        src = self.make_source(position=0)
        _ = src.size_t  # open reader
        src.__del__()
        self.assertEqual(self.record["closed"], 1)


class TestFrameBounds(_ND2TestBase):
    """Review regression: get_frame validates bounds (no silent wrap / ignore)."""

    sizes = {"P": 2, "T": 5, "C": 2, "Y": 8, "X": 10}

    def test_negative_frame_raises(self) -> None:
        src = self.make_source(position=0)
        with self.assertRaises(IndexError):
            src.get_frame(-1)

    def test_out_of_range_frame_raises(self) -> None:
        src = self.make_source(position=0)
        with self.assertRaises(IndexError):
            src.get_frame(5)  # size_t == 5


class TestNoTAxisBounds(_ND2TestBase):
    """Review regression: a file with no T axis still bounds-checks the frame."""

    sizes = {"C": 1, "Y": 5, "X": 5}

    def test_single_frame_only(self) -> None:
        src = self.make_source()
        self.assertEqual(src.size_t, 1)
        src.get_frame(0)  # ok
        with self.assertRaises(IndexError):
            src.get_frame(1)  # would otherwise silently return frame 0's data


class TestReviewRegressions(_ND2TestBase):
    """Review regressions: unknown axis, zero voxel size."""

    sizes = {"P": 2, "T": 5, "C": 2, "Y": 8, "X": 10}

    def test_unknown_axis_rejected(self) -> None:
        self.sizes = {"T": 2, "S": 3, "Y": 4, "X": 4}  # 'S' = RGB sample axis
        self.module, self.record, self.base = _make_fake_nd2(self.sizes, self.voxel_x)
        sys.modules["nd2"] = self.module
        src = self.make_source(position=0)
        with self.assertRaises(ValueError):
            _ = src.size_t

    def test_zero_voxel_size_is_uncalibrated(self) -> None:
        self.module, self.record, self.base = _make_fake_nd2(self.sizes, voxel_x=0.0)
        sys.modules["nd2"] = self.module
        src = self.make_source(position=0)
        self.assertIsNone(src.pixel_size)  # 0 µm -> None, not Q_(0)


class TestAxisOrderGuard(unittest.TestCase):
    """Review regression: a to_dask()/sizes shape mismatch fails loudly + closes."""

    def setUp(self) -> None:
        self._saved_nd2 = sys.modules.get("nd2")

    def tearDown(self) -> None:
        if self._saved_nd2 is None:
            sys.modules.pop("nd2", None)
        else:
            sys.modules["nd2"] = self._saved_nd2

    def test_shape_mismatch_raises_and_closes(self) -> None:
        sizes = {"T": 2, "C": 1, "Y": 4, "X": 4}
        record: dict = {"indices": [], "computes": [], "closed": 0}
        wrong = np.zeros((2, 1, 4, 5), dtype=np.int64)  # X=5 != sizes X=4

        class _F:
            def __init__(self, path: str) -> None:
                self.path = path

            @property
            def sizes(self) -> dict[str, int]:
                return dict(sizes)

            def voxel_size(self) -> _FakeVoxel:
                return _FakeVoxel(0.65, 0.65, 1.0)

            def to_dask(self) -> _RecordingDask:
                return _RecordingDask(wrong, record)

            def close(self) -> None:
                record["closed"] += 1

        module = types.ModuleType("nd2")
        module.ND2File = _F  # type: ignore[attr-defined]
        sys.modules["nd2"] = module

        from acia.segm.nd2_source import ND2SequenceSource

        src = ND2SequenceSource("x.nd2")
        with self.assertRaises(ValueError):
            _ = src.size_t
        self.assertEqual(record["closed"], 1)  # reader closed on the failed open


if __name__ == "__main__":
    unittest.main()
