"""Unit tests for :mod:`acia.segm.tiff_metadata` (OME/ImageJ calibration reading)
and its wiring into :class:`~acia.segm.local.LocalSequenceSource` /
:func:`~acia.segm.open.open_sequence`.

Round-trips through :func:`acia.segm.tiff_export.save_tiff_stack` (the writer
side) so no external fixture files are needed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from acia import ureg
from acia.segm.local import LocalSequenceSource, THWCSequenceSource
from acia.segm.tiff_export import save_tiff_stack
from acia.segm.tiff_metadata import read_tiff_calibration


def _calibrated_source(n=4, h=8, w=8, pixel_size_um=0.0733, interval_s=300.0):
    stack = (np.random.rand(n, h, w, 1) * 1000).astype(np.uint16)
    tps = ureg.Quantity(np.arange(n) * interval_s, "second")
    return THWCSequenceSource(
        stack, pixel_size=ureg.Quantity(pixel_size_um, "micrometer"), timepoints=tps
    )


class TestReadTiffCalibrationOME(unittest.TestCase):
    def test_roundtrip_pixel_size_and_frame_interval(self):
        src = _calibrated_source()
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(src, Path(d) / "ome.tif", ome=True)
            cal = read_tiff_calibration(path)

        self.assertEqual(cal.source, "ome")
        self.assertAlmostEqual(
            cal.pixel_size.to("micrometer").magnitude, 0.0733, places=4
        )
        self.assertAlmostEqual(
            cal.frame_interval.to("second").magnitude, 300.0, places=3
        )

    def test_no_calibration_returns_none_everywhere(self):
        stack = (np.random.rand(2, 5, 5, 1) * 1000).astype(np.uint16)
        src = THWCSequenceSource(stack)
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(src, Path(d) / "ome_raw.tif", ome=True)
            cal = read_tiff_calibration(path)

        self.assertIsNone(cal.pixel_size)
        self.assertIsNone(cal.frame_interval)
        self.assertIsNone(cal.timepoints)
        self.assertIsNone(cal.source)


class TestReadTiffCalibrationImageJ(unittest.TestCase):
    def test_roundtrip_pixel_size_and_frame_interval(self):
        src = _calibrated_source()
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(src, Path(d) / "ij.tif")  # imagej=True default
            cal = read_tiff_calibration(path)

        self.assertEqual(cal.source, "imagej")
        self.assertAlmostEqual(
            cal.pixel_size.to("micrometer").magnitude, 0.0733, places=3
        )
        self.assertAlmostEqual(
            cal.frame_interval.to("second").magnitude, 300.0, places=3
        )

    def test_no_calibration_returns_none(self):
        stack = (np.random.rand(2, 5, 5, 1) * 1000).astype(np.uint16)
        src = THWCSequenceSource(stack)
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(src, Path(d) / "ij_raw.tif")
            cal = read_tiff_calibration(path)

        self.assertIsNone(cal.pixel_size)
        self.assertIsNone(cal.frame_interval)
        self.assertIsNone(cal.source)


class TestReadTiffCalibrationPlainTiff(unittest.TestCase):
    def test_plain_tiff_no_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "plain.tif"
            tifffile.imwrite(path, (np.random.rand(3, 6, 6) * 255).astype(np.uint8))
            cal = read_tiff_calibration(path)

        self.assertIsNone(cal.pixel_size)
        self.assertIsNone(cal.frame_interval)
        self.assertIsNone(cal.timepoints)
        self.assertIsNone(cal.source)


class TestLocalSequenceSourceAutoCalibration(unittest.TestCase):
    def test_auto_detects_from_ome_tiff(self):
        src = _calibrated_source()
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(src, Path(d) / "ome.tif", ome=True)
            loaded = LocalSequenceSource(str(path))

            self.assertAlmostEqual(
                loaded.pixel_size.to("micrometer").magnitude, 0.0733, places=4
            )
            self.assertEqual(loaded.calibration_source, "ome")

    def test_explicit_override_wins_and_skips_file_read(self):
        src = _calibrated_source()
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(src, Path(d) / "ome.tif", ome=True)
            loaded = LocalSequenceSource(
                str(path),
                pixel_size=ureg.Quantity(1.0, "micrometer"),
                frame_interval=ureg.Quantity(1.0, "minute"),
                timepoints=ureg.Quantity(np.arange(4) * 1.0, "minute"),
            )

            self.assertEqual(loaded.pixel_size.to("micrometer").magnitude, 1.0)
            # every field was user-supplied -> no file metadata read attempted
            self.assertIsNone(loaded.calibration_source)

    def test_construction_does_no_io_for_unreachable_path(self):
        # Must not raise: constructing (and never touching a calibration
        # property) does no file access at all.
        LocalSequenceSource("smb://nonexistent-host/share/img.tif")

    def test_partial_override_still_auto_fills_missing_field(self):
        src = _calibrated_source()
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(src, Path(d) / "ome.tif", ome=True)
            loaded = LocalSequenceSource(
                str(path), pixel_size=ureg.Quantity(1.0, "micrometer")
            )

            # pixel_size: user override
            self.assertEqual(loaded.pixel_size.to("micrometer").magnitude, 1.0)
            # frame_interval: auto-detected from the file (not user-supplied)
            self.assertAlmostEqual(
                loaded.timepoints[1].to("second").magnitude, 300.0, places=3
            )


class TestOpenSequenceMetadataFrameInterval(unittest.TestCase):
    def test_metadata_reflects_auto_detected_frame_interval(self):
        from acia.segm.open import open_sequence

        src = _calibrated_source()
        with tempfile.TemporaryDirectory() as d:
            path = save_tiff_stack(src, Path(d) / "ome.tif", ome=True)
            sf = open_sequence(path)

            self.assertAlmostEqual(
                sf.metadata.frame_interval.to("second").magnitude, 300.0, places=3
            )
            self.assertAlmostEqual(sf.metadata.to_dict()["frame_interval_s"], 300.0)


if __name__ == "__main__":
    unittest.main()
