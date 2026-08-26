"""Unit tests for :class:`acia.segm.czi_source.CZISequenceSource`.

These tests never import the real ``aicspylibczi`` package or open a real
``.czi`` file. A fake ``aicspylibczi`` module is injected into ``sys.modules``
(mirroring the ``sys.modules``-fake pattern in ``test_nd2_source.py``). The
fake's ``read_image`` records every call, so we can assert laziness: constructing
the source and reading ``size_*``/``pixel_size`` must read no plane, and
``get_frame(i)`` must read exactly the single ``(scene=position, T=i)`` plane
(never the whole scene).
"""

from __future__ import annotations

import sys
import types
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from acia import ureg

_SCALING_M = 4.349e-8  # metres per pixel -> 0.04349 µm


def _make_meta(n_scenes: int, scaling_x_m: float | None = _SCALING_M) -> ET.Element:
    """Build a minimal CZI metadata tree with Scaling + Scene names."""
    scaling = ""
    if scaling_x_m is not None:
        scaling = (
            "<Scaling><Items>"
            f"<Distance Id='X'><Value>{scaling_x_m}</Value></Distance>"
            f"<Distance Id='Y'><Value>{scaling_x_m}</Value></Distance>"
            "</Items></Scaling>"
        )
    scenes = "".join(f"<Scene Index='{i}' Name='P{i + 1}'/>" for i in range(n_scenes))
    xml = (
        "<ImageDocument><Metadata>"
        f"{scaling}"
        "<Information><Image><Dimensions><S><Scenes>"
        f"{scenes}"
        "</Scenes></S></Dimensions></Image></Information>"
        "</Metadata></ImageDocument>"
    )
    return ET.fromstring(xml)


def _make_fake_czi(
    dims: str,
    sizes: dict[str, int],
    *,
    scaling_x_m: float | None = _SCALING_M,
    mosaic: bool = False,
):
    """Build a fake ``aicspylibczi`` module whose CziFile exposes ``dims``/``size``.

    ``read_image(**constraints)`` returns ``(array, shape_desc)`` where the array
    has the full dimensionality with constrained axes reduced to size 1 (matching
    aicspylibczi), and records the call for laziness assertions.
    """
    record: dict = {"reads": [], "closed": 0}
    shape = tuple(sizes[d] for d in dims)
    base = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    n_scenes = sizes.get("S", 1)

    class _FakeCziFile:
        def __init__(self, path: str) -> None:
            self.path = path

        @property
        def dims(self) -> str:
            return dims

        @property
        def size(self) -> tuple[int, ...]:
            return shape

        @property
        def meta(self) -> ET.Element:
            return _make_meta(n_scenes, scaling_x_m)

        def is_mosaic(self) -> bool:
            return mosaic

        def read_image(self, **constraints):
            record["reads"].append(dict(constraints))
            idx = []
            shape_desc = []
            for d, s in zip(dims, base.shape, strict=True):
                if d in constraints:
                    i = constraints[d]
                    idx.append(slice(i, i + 1))
                    shape_desc.append((d, 1))
                else:
                    idx.append(slice(None))
                    shape_desc.append((d, s))
            return base[tuple(idx)], shape_desc

        def close(self) -> None:
            record["closed"] += 1

    module = types.ModuleType("aicspylibczi")
    module.CziFile = _FakeCziFile  # type: ignore[attr-defined]
    return module, record, base


class _CZITestBase(unittest.TestCase):
    """Installs/removes a fake ``aicspylibczi`` module around each test."""

    dims = "STCYX"
    sizes: dict[str, int] = {"S": 2, "T": 5, "C": 2, "Y": 8, "X": 10}
    scaling_x_m: float | None = _SCALING_M
    mosaic = False

    def setUp(self) -> None:
        self._saved = sys.modules.get("aicspylibczi")
        self.module, self.record, self.base = _make_fake_czi(
            self.dims, self.sizes, scaling_x_m=self.scaling_x_m, mosaic=self.mosaic
        )
        sys.modules["aicspylibczi"] = self.module

    def tearDown(self) -> None:
        if self._saved is None:
            sys.modules.pop("aicspylibczi", None)
        else:
            sys.modules["aicspylibczi"] = self._saved

    def make_source(self, **kwargs):
        from acia.segm.czi_source import CZISequenceSource

        return CZISequenceSource("fake.czi", **kwargs)


class TestShapeMapping(_CZITestBase):
    """Matrix row: single-scene TC-series {S:2,T:5,C:2,Y:8,X:10}, position=1."""

    def test_sizes(self) -> None:
        src = self.make_source(position=1)
        self.assertEqual(src.size_t, 5)
        self.assertEqual(src.size_h, 8)
        self.assertEqual(src.size_w, 10)
        self.assertEqual(src.size_c, 2)
        self.assertEqual(src.num_channels, 2)
        self.assertEqual(src.n_scenes, 2)

    def test_get_frame_shape_and_content(self) -> None:
        src = self.make_source(position=1)
        frame = src.get_frame(0)
        self.assertEqual(frame.raw.shape, (8, 10, 2))
        # dims S,T,C,Y,X -> select S=1,T=0; channel moved last
        expected = np.moveaxis(self.base[1, 0], 0, -1)
        np.testing.assert_array_equal(frame.raw, expected)
        self.assertEqual(frame.frame, 0)

    def test_get_frame_selects_scene_and_time(self) -> None:
        src = self.make_source(position=1)
        f2 = src.get_frame(2)
        expected = np.moveaxis(self.base[1, 2], 0, -1)
        np.testing.assert_array_equal(f2.raw, expected)

    def test_scene_names(self) -> None:
        src = self.make_source(position=1)
        self.assertEqual(src.scene_names.get(0), "P1")
        self.assertEqual(src.scene_names.get(1), "P2")


class TestNoSAxis(_CZITestBase):
    """Matrix row: no S axis -> single scene; size_t==4, single channel."""

    dims = "TCYX"
    sizes = {"T": 4, "C": 1, "Y": 6, "X": 6}

    def test_single_scene(self) -> None:
        src = self.make_source()  # default position=0
        self.assertEqual(src.n_scenes, 1)
        self.assertEqual(src.size_t, 4)
        frame = src.get_frame(0)
        self.assertEqual(frame.raw.shape, (6, 6, 1))
        expected = np.moveaxis(self.base[0], 0, -1)  # dims T,C,Y,X -> T=0
        np.testing.assert_array_equal(frame.raw, expected)


class TestGrayscaleNoC(_CZITestBase):
    """Matrix row: grayscale (no C) -> size_c==1, frame shaped (H,W,1)."""

    dims = "STYX"
    sizes = {"S": 2, "T": 3, "Y": 5, "X": 5}

    def test_channel_axis_added(self) -> None:
        src = self.make_source(position=0)
        self.assertEqual(src.size_c, 1)
        self.assertEqual(src.num_channels, 1)
        frame = src.get_frame(1)
        self.assertEqual(frame.raw.shape, (5, 5, 1))
        expected = self.base[0, 1][..., np.newaxis]  # dims S,T,Y,X -> S=0,T=1
        np.testing.assert_array_equal(frame.raw, expected)


class TestCalibration(_CZITestBase):
    """Matrix rows: calibration from metadata and user override."""

    def test_pixel_size_from_metadata(self) -> None:
        src = self.make_source(position=0)
        self.assertAlmostEqual(
            src.pixel_size.to("micrometer").magnitude, 0.04349, places=5
        )
        self.assertEqual(str(src.pixel_size.units), "micrometer")

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
        src = self.make_source(position=0)
        self.assertIsNone(src.timepoints)


class TestValidation(_CZITestBase):
    def test_scene_out_of_range(self) -> None:
        src = self.make_source(position=5)  # only 2 scenes
        with self.assertRaises(ValueError):
            _ = src.size_t

    def test_mosaic_rejected(self) -> None:
        self.mosaic = True
        self.module, self.record, self.base = _make_fake_czi(
            self.dims, self.sizes, mosaic=True
        )
        sys.modules["aicspylibczi"] = self.module
        src = self.make_source(position=0)
        with self.assertRaises(ValueError):
            _ = src.size_t

    def test_z_stack_rejected(self) -> None:
        self.dims = "STZCYX"
        self.sizes = {"S": 1, "T": 2, "Z": 7, "C": 1, "Y": 4, "X": 4}
        self.module, self.record, self.base = _make_fake_czi(self.dims, self.sizes)
        sys.modules["aicspylibczi"] = self.module
        src = self.make_source(position=0)
        with self.assertRaises(ValueError):
            _ = src.size_t

    def test_z_size_one_squeezed(self) -> None:
        self.dims = "TZCYX"
        self.sizes = {"T": 2, "Z": 1, "C": 1, "Y": 4, "X": 4}
        self.module, self.record, self.base = _make_fake_czi(self.dims, self.sizes)
        sys.modules["aicspylibczi"] = self.module
        src = self.make_source(position=0)
        frame = src.get_frame(0)
        self.assertEqual(frame.raw.shape, (4, 4, 1))
        # dims T,Z,C,Y,X -> T=0,Z=0; C moved last
        expected = np.moveaxis(self.base[0, 0], 0, -1)
        np.testing.assert_array_equal(frame.raw, expected)

    def test_unknown_axis_rejected(self) -> None:
        self.dims = "BTYX"  # 'B' block axis is unsupported
        self.sizes = {"B": 2, "T": 2, "Y": 4, "X": 4}
        self.module, self.record, self.base = _make_fake_czi(self.dims, self.sizes)
        sys.modules["aicspylibczi"] = self.module
        src = self.make_source(position=0)
        with self.assertRaises(ValueError):
            _ = src.size_t

    def test_zero_scaling_is_uncalibrated(self) -> None:
        self.module, self.record, self.base = _make_fake_czi(
            self.dims, self.sizes, scaling_x_m=0.0
        )
        sys.modules["aicspylibczi"] = self.module
        src = self.make_source(position=0)
        self.assertIsNone(src.pixel_size)

    def test_missing_scaling_is_uncalibrated(self) -> None:
        self.module, self.record, self.base = _make_fake_czi(
            self.dims, self.sizes, scaling_x_m=None
        )
        sys.modules["aicspylibczi"] = self.module
        src = self.make_source(position=0)
        self.assertIsNone(src.pixel_size)


class TestCziNotInstalled(unittest.TestCase):
    """Matrix row: aicspylibczi not importable -> ImportError with the pip hint."""

    def setUp(self) -> None:
        self._saved = sys.modules.get("aicspylibczi")
        sys.modules["aicspylibczi"] = None  # type: ignore[assignment]

    def tearDown(self) -> None:
        if self._saved is None:
            sys.modules.pop("aicspylibczi", None)
        else:
            sys.modules["aicspylibczi"] = self._saved

    def test_import_error_message(self) -> None:
        from acia.segm.czi_source import CZISequenceSource

        src = CZISequenceSource("fake.czi")
        with self.assertRaises(ImportError) as ctx:
            _ = src.size_t
        self.assertIn("pip install acia[czi]", str(ctx.exception))


class TestLaziness(_CZITestBase):
    """Matrix row: no plane read on construct/metadata; one per get_frame."""

    def test_construction_and_metadata_read_nothing(self) -> None:
        src = self.make_source(position=1)
        self.assertEqual(self.record["reads"], [])
        _ = (src.size_t, src.size_h, src.size_w, src.size_c, src.pixel_size)
        self.assertEqual(self.record["reads"], [])

    def test_get_frame_reads_exactly_one_plane(self) -> None:
        src = self.make_source(position=1)
        _ = src.size_t  # warm the reader
        self.assertEqual(self.record["reads"], [])

        src.get_frame(2)
        self.assertEqual(len(self.record["reads"]), 1)
        self.assertEqual(self.record["reads"][0], {"S": 1, "T": 2})

        src.get_frame(4)
        self.assertEqual(len(self.record["reads"]), 2)
        self.assertEqual(self.record["reads"][1], {"S": 1, "T": 4})

    def test_close_called_on_del(self) -> None:
        src = self.make_source(position=0)
        _ = src.size_t  # open reader
        src.__del__()
        self.assertEqual(self.record["closed"], 1)


class TestFrameBounds(_CZITestBase):
    """get_frame validates bounds (no silent wrap / ignore)."""

    def test_negative_frame_raises(self) -> None:
        src = self.make_source(position=0)
        with self.assertRaises(IndexError):
            src.get_frame(-1)

    def test_out_of_range_frame_raises(self) -> None:
        src = self.make_source(position=0)
        with self.assertRaises(IndexError):
            src.get_frame(5)  # size_t == 5


class TestNoTAxisBounds(_CZITestBase):
    """A file with no T axis still bounds-checks the frame."""

    dims = "SCYX"
    sizes = {"S": 2, "C": 1, "Y": 5, "X": 5}

    def test_single_frame_only(self) -> None:
        src = self.make_source(position=0)
        self.assertEqual(src.size_t, 1)
        src.get_frame(0)  # ok
        with self.assertRaises(IndexError):
            src.get_frame(1)


if __name__ == "__main__":
    unittest.main()
