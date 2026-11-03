"""Tests for :func:`acia.viz.annotate_sequence` (time stamp + scale bar) and the
position/colour fixes in :func:`acia.viz.render_time` / ``render_scalebar``."""

import logging
import unittest

import numpy as np

from acia import ureg
from acia.segm.local import THWCSequenceSource
from acia.viz import (
    AnnotatedSequenceSource,
    annotate_sequence,
    render_scalebar,
    render_time,
)

GREY = 100


def _movie(t=3, h=200, w=300, frame_interval="5 min", pixel_size="0.1 µm"):
    stack = np.full((t, h, w, 3), GREY, dtype=np.uint8)
    return THWCSequenceSource(
        stack, frame_interval=frame_interval, pixel_size=pixel_size
    )


def _white_columns(rows: np.ndarray) -> np.ndarray:
    """Columns that contain a pure-white pixel anywhere in ``rows``."""
    return np.where((rows == 255).all(axis=-1).any(axis=0))[0]


class TestBorderPlacement(unittest.TestCase):
    def test_bottom_band_is_added_and_image_kept(self):
        src = _movie()
        out = annotate_sequence(src, band_height=40)
        self.assertIsInstance(out, AnnotatedSequenceSource)
        self.assertEqual((out.size_h, out.size_w, len(out)), (240, 300, 3))
        frame = out.get_frame(0).raw
        self.assertEqual(frame.shape, (240, 300, 3))
        self.assertEqual(frame.dtype, np.uint8)
        np.testing.assert_array_equal(frame[:200], src.get_frame(0).raw)

    def test_default_band_keeps_height_even(self):
        out = annotate_sequence(_movie(h=121))
        self.assertEqual(out.size_h % 2, 0)
        self.assertEqual(out.get_frame(0).raw.shape[0], out.size_h)

    def test_band_is_black_except_annotations(self):
        frame = annotate_sequence(_movie(), band_height=40).get_frame(0).raw
        band = frame[200:]
        values = set(np.unique(band))
        # only black background, white text/bar and anti-aliasing greys
        self.assertIn(0, values)
        self.assertIn(255, values)
        # a column strip in the middle (between time and bar) stays black
        self.assertTrue((band[:, 140:160] == 0).all())

    def test_top_band(self):
        src = _movie()
        frame = (
            annotate_sequence(src, band_height=30, placement="border_top")
            .get_frame(0)
            .raw
        )
        self.assertEqual(frame.shape, (230, 300, 3))
        np.testing.assert_array_equal(frame[30:], src.get_frame(0).raw)
        self.assertTrue((frame[:30] == 255).any())

    def test_scale_bar_length_matches_pixel_size(self):
        # 10 µm at 0.1 µm/px = 100 px; nothing else in the band's right half
        out = annotate_sequence(
            _movie(),
            band_height=40,
            show_time=False,
            scalebar_length="10 µm",
            margin=10,
        )
        band = out.get_frame(0).raw[200:]
        # the bar is the widest white row
        widths = (band == 255).all(axis=-1).sum(axis=1)
        self.assertEqual(widths.max(), 100)
        cols = np.where((band[widths.argmax()] == 255).all(axis=-1))[0]
        self.assertEqual((cols.min(), cols.max()), (300 - 10 - 100, 300 - 10 - 1))

    def test_yx_pixel_size_uses_x(self):
        src = _movie(pixel_size=None).with_pixel_size([0.2, 0.1] * ureg.micrometer)
        out = annotate_sequence(src, band_height=40, show_time=False)
        widths = (out.get_frame(0).raw[200:] == 255).all(axis=-1).sum(axis=1)
        self.assertEqual(widths.max(), 100)

    def test_time_text_changes_per_frame(self):
        out = annotate_sequence(_movie(), band_height=40, scalebar_length=None)
        b0 = out.get_frame(0).raw[200:]
        b2 = out.get_frame(2).raw[200:]
        self.assertFalse(np.array_equal(b0, b2))  # 00h 00m vs 00h 10m
        # the time sits on the left
        self.assertLess(_white_columns(b0).max(), 150)

    def test_explicit_calibration_overrides_uncalibrated_source(self):
        # rendered sequences lose calibration; the caller passes the original's
        rendered = THWCSequenceSource(np.zeros((2, 100, 200, 3), dtype=np.uint8))
        out = annotate_sequence(
            rendered,
            timepoints=[0, 30] * ureg.minute,
            pixel_size=0.5 * ureg.micrometer,
            band_height=30,
        )
        band = out.get_frame(1).raw[100:]
        self.assertTrue((band == 255).any())
        self.assertEqual(out.pixel_size, 0.5 * ureg.micrometer)
        self.assertEqual(len(out.timepoints), 2)


class TestOverlayPlacement(unittest.TestCase):
    def test_keeps_size_and_draws_in_corners(self):
        out = annotate_sequence(_movie(), placement="overlay", band_height=30)
        self.assertEqual((out.size_h, out.size_w), (200, 300))
        frame = out.get_frame(0).raw
        self.assertEqual(frame.shape, (200, 300, 3))
        # time top-left, bar bottom-right; the centre is untouched
        self.assertTrue((frame[:50, :150] == 255).any())
        self.assertTrue((frame[150:, 150:] == 255).any())
        self.assertTrue((frame[80:120, 100:200] == GREY).all())

    def test_invalid_placement(self):
        with self.assertRaises(ValueError):
            annotate_sequence(_movie(), placement="left")


class TestMissingCalibration(unittest.TestCase):
    def test_no_pixel_size_skips_bar_with_warning(self):
        src = _movie(pixel_size=None)
        with self.assertLogs("acia.viz.compose", logging.WARNING) as logs:
            out = annotate_sequence(src, band_height=40, show_time=False)
        self.assertIn("no pixel size", "\n".join(logs.output))
        self.assertTrue((out.get_frame(0).raw[200:] == 0).all())

    def test_no_timepoints_skips_time_with_warning(self):
        src = _movie(frame_interval=None)
        with self.assertLogs("acia.viz.compose", logging.WARNING) as logs:
            out = annotate_sequence(src, band_height=40, scalebar_length=None)
        self.assertIn("no timepoints", "\n".join(logs.output))
        self.assertTrue((out.get_frame(0).raw[200:] == 0).all())

    def test_too_short_bar_is_dropped(self):
        with self.assertLogs("acia.viz.compose", logging.WARNING):
            out = annotate_sequence(
                _movie(), band_height=40, show_time=False, scalebar_length="0.1 µm"
            )
        self.assertTrue((out.get_frame(0).raw[200:] == 0).all())


class TestLegacyRendererFixes(unittest.TestCase):
    def test_render_time_relative_position(self):
        src = _movie(t=1, h=100, w=200)
        out = render_time(src, (0.5, 0.5), [0 * ureg.minute], font_size=12)
        frame = out.get_frame(0).raw
        white = np.argwhere((frame == 255).all(axis=-1))
        # text starts at the resolved pixel position (100, 50), not at (0, 0)
        self.assertGreaterEqual(white[:, 1].min(), 100)
        self.assertGreaterEqual(white[:, 0].min(), 50)

    def test_render_scalebar_uses_color(self):
        src = _movie(t=1, h=100, w=200)
        out = render_scalebar(
            src,
            (10, 90),
            0.1 * ureg.micrometer,
            5 * ureg.micrometer,
            1 * ureg.micrometer,
            color=(255, 0, 0),
            show_text=False,
        )
        frame = out.get_frame(0).raw
        self.assertTrue((frame == [255, 0, 0]).all(axis=-1).any())
        self.assertFalse((frame == 255).all(axis=-1).any())


if __name__ == "__main__":
    unittest.main()
