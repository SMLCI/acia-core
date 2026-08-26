"""Tests for :mod:`acia.viz.compose` -- sequence composition (H/V, titles, grids)."""

import unittest

import numpy as np

from acia.segm.local import THWCSequenceSource
from acia.viz import ComposedSequenceSource, compose_sequences, label_sequence


def _clip(t, h, w, value, frame_interval=None):
    stack = np.full((t, h, w, 3), value, dtype=np.uint8)
    return THWCSequenceSource(stack, frame_interval=frame_interval)


class TestComposeShapes(unittest.TestCase):
    def test_horizontal_dims(self):
        a = _clip(4, 30, 50, 60)
        b = _clip(4, 20, 40, 200)
        h = compose_sequences([a, b], axis="horizontal", gap=4)
        self.assertEqual(len(h), 4)
        self.assertEqual(h.size_h, 30)  # max height
        self.assertEqual(h.size_w, 50 + 40 + 4)  # widths + one gap

    def test_vertical_dims(self):
        a = _clip(4, 30, 50, 60)
        b = _clip(4, 20, 40, 200)
        v = compose_sequences([a, b], axis="vertical", gap=6)
        self.assertEqual(v.size_w, 50)  # max width
        self.assertEqual(v.size_h, 30 + 20 + 6)

    def test_frames_are_uint8_rgb(self):
        a = _clip(3, 10, 10, 10)
        b = _clip(3, 10, 10, 20)
        comp = compose_sequences([a, b])
        fr = comp.get_frame(0).raw
        self.assertEqual(fr.dtype, np.uint8)
        self.assertEqual(fr.shape, (10, 20, 3))

    def test_panels_land_in_the_canvas(self):
        a = _clip(1, 10, 10, 60)
        b = _clip(1, 10, 10, 200)
        h = compose_sequences([a, b])  # no gap, equal heights
        fr = h.get_frame(0).raw
        self.assertEqual(int(fr[0, 0, 0]), 60)  # left panel
        self.assertEqual(int(fr[0, 10, 0]), 200)  # right panel


class TestComposeReconciliation(unittest.TestCase):
    def test_min_truncates(self):
        a = _clip(5, 10, 10, 1)
        b = _clip(3, 10, 10, 2)
        self.assertEqual(len(compose_sequences([a, b], n_frames="min")), 3)

    def test_max_holds_last(self):
        a = _clip(5, 10, 10, 1)
        b = _clip(3, 10, 10, 2)
        comp = compose_sequences([a, b], n_frames="max")
        self.assertEqual(len(comp), 5)
        # b ran out at frame 3; its last frame (value 2) is held on frame 4
        self.assertEqual(int(comp.get_frame(4).raw[0, 10, 0]), 2)


class TestComposeAlignment(unittest.TestCase):
    def test_align_start_vs_end(self):
        tall = _clip(1, 20, 10, 100)
        short = _clip(1, 10, 10, 200)
        start = compose_sequences([tall, short], align="start").get_frame(0).raw
        end = compose_sequences([tall, short], align="end").get_frame(0).raw
        # short panel (right half) sits at the top for start, bottom for end
        self.assertEqual(int(start[0, 10, 0]), 200)
        self.assertEqual(int(start[19, 10, 0]), 0)  # padding at bottom
        self.assertEqual(int(end[0, 10, 0]), 0)  # padding at top
        self.assertEqual(int(end[19, 10, 0]), 200)


class TestComposeGap(unittest.TestCase):
    def test_gap_color_painted(self):
        a = _clip(1, 10, 10, 60)
        b = _clip(1, 10, 10, 60)
        h = compose_sequences([a, b], axis="horizontal", gap=3, gap_color=(255, 0, 0))
        fr = h.get_frame(0).raw
        gap_cols = fr[:, 10:13]
        self.assertTrue((gap_cols[..., 0] == 255).all())
        self.assertTrue((gap_cols[..., 1] == 0).all())


class TestTitles(unittest.TestCase):
    def test_label_adds_band_height(self):
        a = _clip(4, 30, 50, 60)
        titled = label_sequence(a, "hello", height=28)
        self.assertEqual(titled.size_h, 30 + 28)
        self.assertEqual(titled.size_w, 50)
        self.assertEqual(len(titled), 4)

    def test_titles_sugar_matches_manual_labeling(self):
        a = _clip(3, 20, 20, 60)
        b = _clip(3, 20, 20, 200)
        sugar = compose_sequences([a, b], titles=["x", "y"], gap=2)
        manual = compose_sequences(
            [label_sequence(a, "x"), label_sequence(b, "y")], gap=2
        )
        self.assertEqual((sugar.size_h, sugar.size_w), (manual.size_h, manual.size_w))

    def test_titles_length_mismatch_raises(self):
        a = _clip(1, 10, 10, 1)
        with self.assertRaises(ValueError):
            compose_sequences([a, a], titles=["only one"])


class TestNestingAndCalibration(unittest.TestCase):
    def test_nesting_produces_grid(self):
        a = _clip(2, 10, 10, 10)
        b = _clip(2, 10, 10, 20)
        row = compose_sequences([a, b], axis="horizontal")  # 10 x 20
        grid = compose_sequences([row, row], axis="vertical")  # 20 x 20
        self.assertIsInstance(grid, ComposedSequenceSource)
        self.assertEqual((grid.size_h, grid.size_w), (20, 20))

    def test_calibration_inherited_and_sliced(self):
        a = _clip(4, 10, 10, 60, frame_interval="5 min")
        b = _clip(2, 10, 10, 200)  # forces min-reconciliation to 2 frames
        comp = compose_sequences([a, b], n_frames="min")
        self.assertEqual(len(comp), 2)
        tp = comp.timepoints
        self.assertIsNotNone(tp)
        self.assertEqual(len(tp), 2)  # sliced to composite length
        self.assertEqual(f"{tp.units:~P}", "min")

    def test_calibration_survives_title_band(self):
        a = _clip(3, 10, 10, 60, frame_interval="5 min")
        titled = label_sequence(a, "t")  # first child is the uncalibrated band
        self.assertIsNotNone(titled.timepoints)
        self.assertEqual(len(titled.timepoints), 3)


class TestValidation(unittest.TestCase):
    def test_empty_sources_raises(self):
        with self.assertRaises(ValueError):
            compose_sequences([])

    def test_bad_axis_raises(self):
        a = _clip(1, 10, 10, 1)
        with self.assertRaises(ValueError):
            compose_sequences([a, a], axis="diagonal")


if __name__ == "__main__":
    unittest.main()
