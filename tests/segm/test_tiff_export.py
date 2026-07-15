"""Unit tests for :func:`acia.segm.tiff_export.save_tiff_stack`.

Writes are exercised against a synthetic in-memory source and reopened with
``tifffile`` to assert shape/dtype/calibration tags. A recording source proves
one-frame-at-a-time writing (peak memory ~ one frame).
"""

from __future__ import annotations

import re
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import tifffile

from acia import ureg
from acia.segm.local import THWCSequenceSource
from acia.segm.tiff_export import save_tiff_stack


class _RecordingSource:
    """Minimal ImageSequenceSource that records the order frames are read."""

    def __init__(self, stack, *, pixel_size=None, timepoints=None):
        self.stack = stack
        self._pixel_size = pixel_size
        self._timepoints = timepoints
        self.reads: list[int] = []

    @property
    def size_t(self):
        return len(self.stack)

    def get_frame(self, i):
        self.reads.append(i)
        return types.SimpleNamespace(raw=self.stack[i])

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def timepoints(self):
        return self._timepoints


class TestSaveTiffStack(unittest.TestCase):
    def test_single_channel_roundtrip(self):
        stack = (np.random.rand(5, 8, 10, 1) * 1000).astype(np.uint16)
        src = THWCSequenceSource(stack, pixel_size=ureg.Quantity(0.0733, "micrometer"))
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(src, Path(d) / "out.tif")
            arr = tifffile.imread(path)
            self.assertEqual(arr.shape, (5, 8, 10))  # singleton channel squeezed
            self.assertEqual(arr.dtype, np.uint16)

    def test_multichannel_roundtrip(self):
        stack = (np.random.rand(3, 6, 6, 2) * 1000).astype(np.uint16)
        src = THWCSequenceSource(stack)
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(src, Path(d) / "mc.tif")
            with tifffile.TiffFile(path) as tf:
                axes = tf.series[0].axes
            self.assertIn("C", axes)

    def test_calibration_tags(self):
        stack = (np.random.rand(4, 8, 8, 1) * 1000).astype(np.uint16)
        tps = ureg.Quantity(np.arange(4) * 300.0, "second")
        src = THWCSequenceSource(
            stack, pixel_size=ureg.Quantity(0.0733, "micrometer"), timepoints=tps
        )
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(src, Path(d) / "cal.tif")
            with tifffile.TiffFile(path) as tf:
                ij = tf.imagej_metadata
                self.assertEqual(ij.get("unit"), "um")
                self.assertAlmostEqual(ij.get("finterval"), 300.0, places=3)

    def test_no_calibration_omits_tags(self):
        stack = (np.random.rand(2, 5, 5, 1) * 1000).astype(np.uint16)
        src = THWCSequenceSource(stack)  # no pixel_size / timepoints
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(src, Path(d) / "raw.tif")
            with tifffile.TiffFile(path) as tf:
                ij = tf.imagej_metadata or {}
            self.assertNotIn("unit", ij)
            self.assertNotIn("finterval", ij)

    def test_lazy_one_frame_at_a_time(self):
        stack = (np.random.rand(6, 5, 5, 1) * 1000).astype(np.uint16)
        rec = _RecordingSource(stack)
        with tempfile.TemporaryDirectory() as d:
            save_tiff_stack(rec, Path(d) / "lazy.tif")
        # every frame read exactly once, in order (never a bulk read)
        self.assertEqual(rec.reads, list(range(6)))

    def test_dtype_cast(self):
        stack = (np.random.rand(3, 5, 5, 1) * 1000).astype(np.uint16)
        src = THWCSequenceSource(stack)
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(src, Path(d) / "u8.tif", dtype="uint8")
            arr = tifffile.imread(path)
            self.assertEqual(arr.dtype, np.uint8)

    def test_creates_parent_dir(self):
        stack = (np.random.rand(2, 5, 5, 1) * 1000).astype(np.uint16)
        src = THWCSequenceSource(stack)
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(src, Path(d) / "new" / "sub" / "out.tif")
            self.assertTrue(Path(path).exists())

    def test_ome_calibration_and_channel_names(self):
        stack = (np.random.rand(4, 8, 8, 1) * 1000).astype(np.uint16)
        tps = ureg.Quantity(np.arange(4) * 300.0, "second")
        src = THWCSequenceSource(
            stack, pixel_size=ureg.Quantity(0.0733, "micrometer"), timepoints=tps
        )
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(
                src, Path(d) / "ome.tif", ome=True, channel_names=["100x PH"]
            )
            with tifffile.TiffFile(path) as tf:
                ome_xml = tf.ome_metadata
                arr = tf.asarray()
            self.assertIsNotNone(ome_xml)
            self.assertIn('PhysicalSizeX="0.0733', ome_xml)
            self.assertIn('TimeIncrement="300.0"', ome_xml)
            self.assertIn('Name="100x PH"', ome_xml)
            self.assertEqual(arr.shape, (4, 8, 8))

    def test_ome_no_calibration_or_channels_omits_tags(self):
        stack = (np.random.rand(2, 5, 5, 1) * 1000).astype(np.uint16)
        src = THWCSequenceSource(stack)  # no pixel_size / timepoints
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(src, Path(d) / "ome_raw.tif", ome=True)
            with tifffile.TiffFile(path) as tf:
                ome_xml = tf.ome_metadata
            self.assertNotIn("PhysicalSizeX", ome_xml)
            self.assertNotIn("TimeIncrement", ome_xml)
            channel_tag = re.search(r"<Channel\b[^>]*>", ome_xml).group(0)
            self.assertNotIn("Name=", channel_tag)  # no channel name fabricated

    def test_compression_roundtrip_and_smaller(self):
        # A synthetic gradient (not noise) compresses meaningfully with zlib.
        row = np.linspace(0, 65535, 200, dtype=np.uint16)
        stack = np.tile(row, (10, 200, 1))[..., None]
        src = THWCSequenceSource(stack)
        with tempfile.TemporaryDirectory() as d:
            uncompressed = save_tiff_stack(src, Path(d) / "raw.tif")
            compressed = save_tiff_stack(src, Path(d) / "zlib.tif", compression="zlib")
            self.assertTrue(
                np.array_equal(
                    tifffile.imread(uncompressed), tifffile.imread(compressed)
                )
            )
            self.assertLess(
                Path(compressed).stat().st_size, Path(uncompressed).stat().st_size
            )

    def test_lazy_one_frame_at_a_time_under_ome_and_compression(self):
        stack = (np.random.rand(6, 5, 5, 1) * 1000).astype(np.uint16)
        rec = _RecordingSource(stack)
        with tempfile.TemporaryDirectory() as d:
            save_tiff_stack(rec, Path(d) / "lazy_ome.tif", ome=True, compression="zlib")
        self.assertEqual(rec.reads, list(range(6)))


if __name__ == "__main__":
    unittest.main()
