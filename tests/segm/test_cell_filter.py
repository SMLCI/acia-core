"""Tests for the physical-unit, pluggable ``CellFilter`` abstraction."""

import unittest

import numpy as np
import pint

from acia import Q_
from acia.base import Contour, Overlay
from acia.segm.filter import (
    AreaFilter,
    BoundaryClosenessFilter,
    CellFilter,
    CircularityFilter,
    LengthFilter,
    WidthFilter,
    apply_cell_filters,
)
from acia.segm.local import THWCSequenceSource


def _square(side: float, *, x0: float = 0.0, y0: float = 0.0, frame: int = 0, id=0):
    """Axis-aligned square contour of edge ``side`` (in px) with corner (x0, y0)."""
    return Contour(
        [
            [x0, y0],
            [x0 + side, y0],
            [x0 + side, y0 + side],
            [x0, y0 + side],
        ],
        score=-1,
        frame=frame,
        id=id,
    )


def _source(h: int = 200, w: int = 200, t: int = 1, pixel_size="0.5 micrometer"):
    """Calibrated, blank ``THWCSequenceSource`` of shape (t, h, w, 1)."""
    return THWCSequenceSource(
        np.zeros((t, h, w, 1), dtype=np.uint8), pixel_size=pixel_size
    )


class TestAreaFilter(unittest.TestCase):
    def test_physical_area_keeps_right_contours(self):
        # at 0.5 µm/px: 4px square -> 2µm side -> 4µm²; 8px -> 4µm side -> 16µm²;
        # 12px -> 6µm side -> 36µm²
        images = _source(pixel_size="0.5 micrometer")
        overlay = Overlay(
            [
                _square(4, id="small"),  # 4 µm²
                _square(8, x0=20, id="mid"),  # 16 µm²
                _square(12, x0=50, id="big"),  # 36 µm²
            ]
        )

        result = apply_cell_filters(
            overlay,
            [AreaFilter(Q_(2, "um**2"), Q_(20, "um**2"))],
            images=images,
        )

        self.assertEqual({c.id for c in result.contours}, {"small", "mid"})

    def test_area_value_is_calibrated_quantity(self):
        images = _source(pixel_size="0.5 micrometer")
        cont = _square(4)
        v = AreaFilter().value(cont, images=images)
        self.assertTrue(v.check("[length] ** 2"))
        self.assertAlmostEqual(v.to("um**2").magnitude, 4.0)

    def test_open_upper_bound(self):
        images = _source(pixel_size="0.5 micrometer")
        overlay = Overlay(
            [
                _square(4, id="small"),  # 4 µm²
                _square(12, x0=50, id="big"),  # 36 µm²
            ]
        )
        result = apply_cell_filters(
            overlay, [AreaFilter(vmin=Q_(10, "um**2"))], images=images
        )
        self.assertEqual({c.id for c in result.contours}, {"big"})

    def test_camera_invariance(self):
        # Same physical cells imaged at two pixel sizes must yield identical
        # kept ids despite different pixel counts.
        #
        # physical sides: 2 µm, 4 µm, 6 µm  ->  4, 16, 36 µm²
        # source A: 0.5 µm/px -> 4, 8, 12 px squares
        # source B: 1.0 µm/px -> 2, 4, 6  px squares
        images_a = _source(pixel_size="0.5 micrometer")
        images_b = _source(pixel_size=Q_(1.0, "micrometer"))

        overlay_a = Overlay(
            [
                _square(4, id="c2um"),
                _square(8, x0=20, id="c4um"),
                _square(12, x0=50, id="c6um"),
            ]
        )
        overlay_b = Overlay(
            [
                _square(2, id="c2um"),
                _square(4, x0=20, id="c4um"),
                _square(6, x0=50, id="c6um"),
            ]
        )

        cell_filter = AreaFilter(Q_(2, "um**2"), Q_(20, "um**2"))

        kept_a = {
            c.id for c in apply_cell_filters(overlay_a, [cell_filter], images=images_a)
        }
        kept_b = {
            c.id for c in apply_cell_filters(overlay_b, [cell_filter], images=images_b)
        }

        self.assertEqual(kept_a, kept_b)
        self.assertEqual(kept_a, {"c2um", "c4um"})


class TestLengthWidthFilters(unittest.TestCase):
    def test_length_and_width(self):
        images = _source(pixel_size="0.5 micrometer")
        # 4 x 10 px rectangle -> at 0.5 µm/px: width 2 µm, length 5 µm
        rect = Contour([[0, 0], [10, 0], [10, 4], [0, 4]], score=-1, frame=0, id="r")
        length = LengthFilter().value(rect, images=images)
        width = WidthFilter().value(rect, images=images)
        self.assertAlmostEqual(length.to("um").magnitude, 5.0, places=4)
        self.assertAlmostEqual(width.to("um").magnitude, 2.0, places=4)

    def test_length_filter_selection(self):
        images = _source(pixel_size="0.5 micrometer")
        overlay = Overlay(
            [
                Contour([[0, 0], [10, 0], [10, 4], [0, 4]], -1, 0, "long"),  # 5 µm
                Contour([[0, 0], [4, 0], [4, 4], [0, 4]], -1, 0, "short"),  # 2 µm
            ]
        )
        result = apply_cell_filters(
            overlay, [LengthFilter(vmin=Q_(3, "um"))], images=images
        )
        self.assertEqual({c.id for c in result.contours}, {"long"})


class TestCircularityFilter(unittest.TestCase):
    def test_circularity_dimensionless(self):
        images = _source()
        # square circularity ~ pi/4 ~ 0.785; a near-circular polygon ~ 1.0
        square = _square(10, id="sq")
        circle = Contour(
            [
                [np.cos(t) * 10 + 50, np.sin(t) * 10 + 50]
                for t in np.linspace(0, 2 * np.pi, 64, endpoint=False)
            ],
            -1,
            0,
            "circ",
        )
        overlay = Overlay([square, circle])
        result = apply_cell_filters(
            overlay, [CircularityFilter(vmin=0.9)], images=images
        )
        self.assertEqual({c.id for c in result.contours}, {"circ"})

    def test_circularity_value_is_dimensionless(self):
        images = _source()
        v = CircularityFilter().value(_square(10), images=images)
        self.assertTrue(v.dimensionless)


class TestMultipleFilters(unittest.TestCase):
    def test_and_of_filters(self):
        images = _source(pixel_size="0.5 micrometer")
        # square: large area + high-ish circularity; thin rect: large area, low circ
        square = _square(12, id="square")  # 36 µm², circ ~0.785
        thin = Contour(
            [[0, 0], [40, 0], [40, 3], [0, 3]], -1, 0, "thin"
        )  # area 120px²=30µm², circ low
        overlay = Overlay([square, thin])
        result = apply_cell_filters(
            overlay,
            [AreaFilter(vmin=Q_(20, "um**2")), CircularityFilter(vmin=0.6)],
            images=images,
        )
        # both pass area, only the square passes circularity
        self.assertEqual({c.id for c in result.contours}, {"square"})


class TestMultiFrameTimeModel(unittest.TestCase):
    def test_multi_frame_time_preserved(self):
        images = _source(t=3, pixel_size="0.5 micrometer")
        overlay = Overlay(
            [
                _square(8, frame=0, id="f0_keep"),  # 16 µm²
                _square(4, x0=20, frame=0, id="f0_drop"),  # 4 µm²
                _square(8, frame=1, id="f1_keep"),
                _square(8, frame=2, id="f2_keep"),
            ],
            frame_interval="15 minute",
        )
        before = overlay.timepoints
        self.assertIsNotNone(before)

        result = apply_cell_filters(
            overlay, [AreaFilter(vmin=Q_(10, "um**2"))], images=images
        )

        self.assertEqual(
            {c.id for c in result.contours}, {"f0_keep", "f1_keep", "f2_keep"}
        )
        # surviving contours keep their frames
        self.assertEqual({c.frame for c in result.contours}, {0, 1, 2})
        # time model preserved
        after = result.timepoints
        self.assertIsNotNone(after)
        np.testing.assert_allclose(
            after.to("minute").magnitude, before.to("minute").magnitude
        )

    def test_explicit_timepoints_preserved(self):
        images = _source(t=2, pixel_size="0.5 micrometer")
        tp = Q_(np.array([0.0, 42.0]), "minute")
        overlay = Overlay(
            [
                _square(8, frame=0, id="a"),
                _square(8, frame=1, id="b"),
            ],
            timepoints=tp,
        )
        result = apply_cell_filters(
            overlay, [AreaFilter(vmin=Q_(1, "um**2"))], images=images
        )
        np.testing.assert_allclose(
            result.timepoints.to("minute").magnitude, [0.0, 42.0]
        )


class TestBoundaryClosenessFilter(unittest.TestCase):
    def test_drops_near_border_cells(self):
        # 20x20 px frame, 0.5 µm/px -> 1 µm == 2 px margin
        images = _source(h=20, w=20, pixel_size="0.5 micrometer")
        overlay = Overlay(
            [
                _square(4, x0=8, y0=8, id="center"),  # bbox 8..12, far from border
                _square(4, x0=0, y0=8, id="left_edge"),  # touches x=0
            ]
        )
        result = apply_cell_filters(
            overlay, [BoundaryClosenessFilter(Q_(1, "um"))], images=images
        )
        self.assertEqual({c.id for c in result.contours}, {"center"})

    def test_range_is_open_above(self):
        f = BoundaryClosenessFilter(Q_(1, "um"))
        self.assertEqual(f.range, (Q_(1, "um"), None))

    def test_value_is_min_distance(self):
        images = _source(h=20, w=20, pixel_size="0.5 micrometer")
        # bbox at x in [2,6], y in [3,7]; distances px: left=2, top=3,
        # right=20-6=14, bottom=20-7=13 -> min 2 px -> 1 µm
        cont = Contour([[2, 3], [6, 3], [6, 7], [2, 7]], -1, 0, "c")
        v = BoundaryClosenessFilter(Q_(0, "um")).value(cont, images=images)
        self.assertAlmostEqual(v.to("um").magnitude, 1.0)


class TestErrorHandling(unittest.TestCase):
    def test_images_none_raises(self):
        overlay = Overlay([_square(8)])
        with self.assertRaises(ValueError):
            apply_cell_filters(overlay, [AreaFilter()], images=None)

    def test_uncalibrated_source_raises(self):
        images = _source(pixel_size=None)
        overlay = Overlay([_square(8)])
        with self.assertRaises(ValueError):
            apply_cell_filters(overlay, [AreaFilter()], images=images)

    def test_dimensionality_mismatch_raises(self):
        images = _source(pixel_size="0.5 micrometer")
        overlay = Overlay([_square(8)])
        # area filter given a length (µm) range -> pint dimensionality error
        with self.assertRaises(pint.DimensionalityError):
            apply_cell_filters(
                overlay, [AreaFilter(Q_(2, "um"), Q_(20, "um"))], images=images
            )

    def test_base_value_not_implemented(self):
        images = _source()
        with self.assertRaises(NotImplementedError):
            CellFilter().value(_square(8), images=images)


class TestEmptyResult(unittest.TestCase):
    def test_empty_overlay_no_crash(self):
        images = _source(pixel_size="0.5 micrometer")
        overlay = Overlay([_square(4, id="tiny")])  # 4 µm²
        result = apply_cell_filters(
            overlay, [AreaFilter(vmin=Q_(1000, "um**2"))], images=images
        )
        self.assertIsInstance(result, Overlay)
        self.assertEqual(len(result.contours), 0)


class TestCustomFilter(unittest.TestCase):
    def test_trivial_custom_filter_participates(self):
        # A custom filter with no registry: just subclass + set name + value.
        class XPositionFilter(CellFilter):
            name = "x_position"

            def value(self, cont, *, images):
                # x-centroid calibrated to µm via pixel_size
                x_px = float(cont.center[0])
                ps = images.pixel_size
                return x_px * ps

        images = _source(pixel_size="0.5 micrometer")
        overlay = Overlay(
            [
                _square(4, x0=0, id="left"),  # centroid x=2px -> 1 µm
                _square(4, x0=100, id="right"),  # centroid x=102px -> 51 µm
            ]
        )
        result = apply_cell_filters(
            overlay, [XPositionFilter(vmin=Q_(10, "um"))], images=images
        )
        self.assertEqual({c.id for c in result.contours}, {"right"})


class TestDegenerateGeometry(unittest.TestCase):
    def test_degenerate_contour_dropped_not_crash(self):
        # A collinear (zero-area) contour must not crash Length/Width/Circularity
        # (minimum_rotated_rectangle collapses to a line; perimeter may be 0) and
        # should be dropped by any positive lower bound, keeping the real cell.
        images = _source(pixel_size="0.5 micrometer")
        degenerate = Contour([[0, 0], [5, 0], [10, 0]], -1, 0, "line")
        good = _square(8, x0=20, id="ok")  # 16 µm², real geometry
        overlay = Overlay([degenerate, good])

        for cell_filter in (
            AreaFilter(vmin=Q_(1, "um**2")),
            LengthFilter(vmin=Q_(1, "um")),
            WidthFilter(vmin=Q_(0.5, "um")),
            CircularityFilter(vmin=0.1),
        ):
            result = apply_cell_filters(overlay, [cell_filter], images=images)
            self.assertEqual(
                {c.id for c in result.contours}, {"ok"}, msg=cell_filter.name
            )


if __name__ == "__main__":
    unittest.main()
