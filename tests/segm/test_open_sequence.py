"""Unit tests for :mod:`acia.segm.open` (``open_sequence`` + ``SequenceFile``).

ND2/CZI backends are exercised with fake reader modules injected into
``sys.modules`` (no real package, no real file); the TIFF backend is exercised
against a real small ``.tif`` written to a temp dir. Laziness is asserted via
recording fakes: constructing the handle and reading metadata/positions must read
no pixel frame; ``thumbnail`` reads exactly one.
"""

from __future__ import annotations

import json
import sys
import types
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import tifffile


# --------------------------------------------------------------------------- #
# fake ND2
# --------------------------------------------------------------------------- #
class _RecordingSlice:
    def __init__(self, base, index, record):
        self._base, self._index, self._record = base, index, record

    def __array__(self, dtype=None):
        self._record["computes"].append(self._index)
        out = self._base[self._index]
        return out.astype(dtype) if dtype is not None else out


class _RecordingDask:
    def __init__(self, base, record):
        self._base, self._record, self.shape = base, record, base.shape

    def __getitem__(self, index):
        return _RecordingSlice(self._base, index, self._record)

    def __array__(self, dtype=None):  # pragma: no cover
        raise AssertionError("whole-array materialization is forbidden")


def _make_fake_nd2(sizes, *, voxel_x=0.0733, channel="BF"):
    record = {"computes": []}
    shape = tuple(sizes.values())
    base = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)

    class _Chan:
        def __init__(self, name):
            self.channel = types.SimpleNamespace(name=name)

    class _FakeND2File:
        def __init__(self, path):
            self.path = path
            self.dtype = np.uint16
            self.metadata = types.SimpleNamespace(channels=[_Chan(channel)])

        @property
        def sizes(self):
            return dict(sizes)

        def voxel_size(self):
            return types.SimpleNamespace(x=voxel_x, y=voxel_x, z=1.0)

        def to_dask(self):
            return _RecordingDask(base, record)

        def close(self):
            pass

    module = types.ModuleType("nd2")
    module.ND2File = _FakeND2File  # type: ignore[attr-defined]
    return module, record, base


# --------------------------------------------------------------------------- #
# fake CZI
# --------------------------------------------------------------------------- #
def _make_fake_czi(dims, sizes, *, scaling_x_m=4.349e-8, channel="63x"):
    record = {"reads": []}
    shape = tuple(sizes[d] for d in dims)
    base = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    n_scenes = sizes.get("S", 1)
    scenes = "".join(f"<Scene Index='{i}' Name='P{i + 1}'/>" for i in range(n_scenes))
    xml = (
        "<Root><Metadata>"
        f"<Scaling><Items><Distance Id='X'><Value>{scaling_x_m}</Value></Distance>"
        "</Items></Scaling>"
        f"<Information><Image><Dimensions>"
        f"<Channels><Channel Name='{channel}'/></Channels>"
        f"<S><Scenes>{scenes}</Scenes></S>"
        "</Dimensions></Image></Information>"
        "</Metadata></Root>"
    )

    class _FakeCziFile:
        def __init__(self, path):
            self.path = path
            self.pixel_type = "gray16"

        @property
        def dims(self):
            return dims

        @property
        def size(self):
            return shape

        @property
        def meta(self):
            return ET.fromstring(xml)

        def is_mosaic(self):
            return False

        def read_image(self, **constraints):
            record["reads"].append(dict(constraints))
            idx, desc = [], []
            for d, s in zip(dims, base.shape, strict=True):
                if d in constraints:
                    idx.append(slice(constraints[d], constraints[d] + 1))
                    desc.append((d, 1))
                else:
                    idx.append(slice(None))
                    desc.append((d, s))
            return base[tuple(idx)], desc

        def close(self):
            pass

    module = types.ModuleType("aicspylibczi")
    module.CziFile = _FakeCziFile  # type: ignore[attr-defined]
    return module, record, base


class _FakeModuleBase(unittest.TestCase):
    def _install(self, name, module):
        self._saved = getattr(self, "_saved", {})
        self._saved[name] = sys.modules.get(name)
        sys.modules[name] = module

    def tearDown(self):
        for name, mod in getattr(self, "_saved", {}).items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
class TestDispatch(unittest.TestCase):
    def test_unknown_suffix_raises(self):
        from acia.segm.open import open_sequence

        with self.assertRaises(ValueError):
            open_sequence("movie.lif")

    def test_suffix_selects_format(self):
        from acia.segm.open import open_sequence

        self.assertEqual(open_sequence("a.nd2").format, "nd2")
        self.assertEqual(open_sequence("a.czi").format, "czi")
        self.assertEqual(open_sequence("a.tif").format, "tiff")
        self.assertEqual(open_sequence("a.TIFF").format, "tiff")

    def test_construct_opens_nothing(self):
        # No backend import / file open happens at construction time.
        from acia.segm.open import open_sequence

        f = open_sequence("nope.nd2")  # nd2 not installed here, but no access yet
        self.assertEqual(f.path, "nope.nd2")

    def test_backend_missing_raises_importerror(self):
        from acia.segm.open import open_sequence

        saved = sys.modules.get("nd2")
        sys.modules["nd2"] = None  # force ImportError
        try:
            with self.assertRaises(ImportError):
                _ = open_sequence("a.nd2").num_positions
        finally:
            if saved is None:
                sys.modules.pop("nd2", None)
            else:
                sys.modules["nd2"] = saved


# --------------------------------------------------------------------------- #
# ND2 backend
# --------------------------------------------------------------------------- #
class TestND2Backend(_FakeModuleBase):
    sizes = {"P": 108, "T": 6, "C": 1, "Y": 8, "X": 10}

    def setUp(self):
        module, self.record, self.base = _make_fake_nd2(self.sizes)
        self._install("nd2", module)

    def open(self):
        from acia.segm.open import open_sequence

        return open_sequence("a.nd2")

    def test_metadata(self):
        f = self.open()
        md = f.metadata
        self.assertEqual(f.num_positions, 108)
        self.assertEqual(md.num_timepoints, 6)
        self.assertEqual(md.channels, ["BF"])
        self.assertEqual(md.dtype, "uint16")
        self.assertAlmostEqual(
            md.pixel_size.to("micrometer").magnitude, 0.0733, places=4
        )

    def test_positions_lazy(self):
        f = self.open()
        self.assertEqual(len(f.positions), 108)
        self.assertEqual(f.positions[0].index, 0)

    def test_position_returns_source_and_caches(self):
        f = self.open()
        src = f.position(37)
        self.assertEqual(src.size_t, 6)
        self.assertIs(f.position(37), src)  # cached

    def test_position_out_of_range(self):
        f = self.open()
        with self.assertRaises(ValueError):
            f.position(200)

    def test_metadata_reads_no_pixels(self):
        f = self.open()
        _ = (f.metadata, f.num_positions, f.positions)
        self.assertEqual(self.record["computes"], [])

    def test_thumbnail_reads_one_frame(self):
        f = self.open()
        _ = f.metadata
        self.assertEqual(self.record["computes"], [])
        thumb = f.thumbnail(3, downscale=2)
        self.assertEqual(thumb.ndim, 3)
        self.assertEqual(thumb.shape[2], 3)
        self.assertEqual(thumb.dtype, np.uint8)
        self.assertEqual(len(self.record["computes"]), 1)

    def test_thumbnail_png_bytes(self):
        f = self.open()
        data = f.thumbnail_png(0, downscale=2)
        self.assertIsInstance(data, bytes)
        self.assertTrue(data.startswith(b"\x89PNG"))

    def test_user_pixel_size_override(self):
        from acia.segm.open import open_sequence

        f = open_sequence("a.nd2", pixel_size="0.5 um")
        self.assertEqual(f.metadata.pixel_size.to("micrometer").magnitude, 0.5)
        self.assertEqual(f.position(0).pixel_size.to("micrometer").magnitude, 0.5)

    def test_to_dict_json_safe(self):
        f = self.open()
        d = f.metadata.to_dict()
        json.dumps(d)  # must not raise
        self.assertEqual(d["num_positions"], 108)
        self.assertAlmostEqual(d["pixel_size_um"], 0.0733, places=4)
        self.assertIsNone(d["frame_interval_s"])


# --------------------------------------------------------------------------- #
# CZI backend
# --------------------------------------------------------------------------- #
class TestCZIBackend(_FakeModuleBase):
    def setUp(self):
        module, self.record, self.base = _make_fake_czi(
            "STCYX", {"S": 107, "T": 4, "C": 1, "Y": 8, "X": 10}
        )
        self._install("aicspylibczi", module)

    def open(self):
        from acia.segm.open import open_sequence

        return open_sequence("b.czi")

    def test_metadata(self):
        f = self.open()
        md = f.metadata
        self.assertEqual(md.num_positions, 107)
        self.assertEqual(md.channels, ["63x"])
        self.assertEqual(md.dtype, "uint16")
        self.assertAlmostEqual(
            md.pixel_size.to("micrometer").magnitude, 0.04349, places=5
        )

    def test_scene_names_in_positions(self):
        f = self.open()
        self.assertEqual(f.positions[0].name, "P1")
        self.assertEqual(f.positions[1].name, "P2")

    def test_metadata_reads_no_pixels(self):
        f = self.open()
        _ = (f.metadata, f.num_positions, f.positions)
        self.assertEqual(self.record["reads"], [])

    def test_thumbnail_reads_one_plane(self):
        f = self.open()
        _ = f.metadata
        f.thumbnail(2, downscale=2)
        self.assertEqual(len(self.record["reads"]), 1)
        self.assertEqual(self.record["reads"][0], {"S": 2, "T": 0})

    def test_position_delegates(self):
        f = self.open()
        src = f.position(5)
        self.assertEqual(src.size_h, 8)
        self.assertEqual(src.size_w, 10)


# --------------------------------------------------------------------------- #
# TIFF backend (real small file)
# --------------------------------------------------------------------------- #
class TestTiffBackend(unittest.TestCase):
    def test_open_and_metadata(self):
        import tempfile
        from pathlib import Path

        from acia.segm.open import open_sequence

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "stack.tif"
            stack = (np.random.rand(3, 8, 10) * 1000).astype(np.uint16)
            tifffile.imwrite(path, stack)

            f = open_sequence(path)
            self.assertEqual(f.format, "tiff")
            md = f.metadata
            self.assertEqual(md.num_positions, 1)
            self.assertEqual(md.num_timepoints, 3)
            self.assertEqual(md.sizes["Y"], 8)
            self.assertEqual(md.sizes["X"], 10)
            # true dtype/channel count, not the visualization-normalized uint8/RGB
            # LocalSequenceSource's default construction would otherwise produce
            self.assertEqual(md.dtype, "uint16")
            self.assertEqual(md.sizes["C"], 1)

            source = f.position(0)
            frame = source.get_frame(0)
            self.assertEqual(frame.raw.dtype, np.uint16)
            np.testing.assert_array_equal(frame.raw[..., 0], stack[0])

            thumb = f.thumbnail(0, downscale=2)
            self.assertEqual(thumb.shape[2], 3)

            with self.assertRaises(ValueError):
                f.position(1)  # single-position


if __name__ == "__main__":
    unittest.main()
