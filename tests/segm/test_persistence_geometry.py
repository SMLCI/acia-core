"""Persisted geometry must keep its exact size; only display may dilate.

Rasterising a polygon has two defensible answers a pixel apart. ``rasterio``
follows the pixel-centre rule -- a 10 px square covers 10x10 -- and
``cv2.fillPoly`` fills a closed polygon inclusively, covering 11x11. The mask
renderers take the cv2 shortcut because it is cheaper and a pixel of outline
does not matter on screen. Everything that **saves, loads, exports or measures**
must not: a cell that changes size when it is written to disk and read back
corrupts every downstream area, length and growth-rate number.

These tests pin that split across a spread of geometries, chosen so the ones
that suffer most from an inclusive fill are represented -- thin rods, 1 px
slivers and 2 px specks lose the largest *fraction* of their area to a
one-pixel dilation (a 1 px sliver more than doubles).

Being exact *once* is not enough: a dataset gets segmented, saved, reloaded,
re-exported and saved again, and each cycle re-traces the mask into a polygon
and re-fills it, so a half-pixel bias compounds. ``TestRepeatedPersistenceIsStable``
runs ten full cycles and requires the geometry to be a fixed point --
byte-identical every time, not merely equal in area.
``TestDisplayRasterizerWouldCompound`` shows the alternative: on the display
rasterizer a thin rod grows from 220 px to over 600 px in six cycles.

The geometries are shared by the polygon tests and the mask tests so a failure
names the same shape in both.
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acia.base import Contour, Overlay
from acia.segm.formats import (
    load_segmentation,
    overlay_from_masks,
    read_ctc_segmentation_native,
    save_segmentation,
)
from acia.segm.rasterize import frame_label_mask
from acia.tracking.output import CTCTrackingHelper
from acia.utils import polygon_to_mask

HEIGHT = WIDTH = 96

_ANGLES = np.linspace(0, 2 * np.pi, 60, endpoint=False)


def polygon_geometries():
    """``(name, (N, 2) outline)`` covering the shapes rasterisation trips over."""
    yield "square", np.array([(10, 10), (30, 10), (30, 30), (10, 30)], np.float32)
    # a rod: the bacterium-shaped case, where a 1 px skin is a large area fraction
    yield "rod_thin", np.array([(5, 40), (60, 40), (60, 44), (5, 44)], np.float32)
    yield (
        "circle",
        np.stack([30 + 15 * np.cos(_ANGLES), 50 + 15 * np.sin(_ANGLES)], 1).astype(
            np.float32
        ),
    )
    yield (
        "ellipse_rotated",
        np.stack(
            [
                50
                + 18 * np.cos(_ANGLES) * np.cos(0.6)
                - 6 * np.sin(_ANGLES) * np.sin(0.6),
                50
                + 18 * np.cos(_ANGLES) * np.sin(0.6)
                + 6 * np.sin(_ANGLES) * np.cos(0.6),
            ],
            1,
        ).astype(np.float32),
    )
    # concave: a convex-hull shortcut anywhere in the chain would show up here
    yield (
        "L_concave",
        np.array(
            [(10, 10), (40, 10), (40, 20), (20, 20), (20, 45), (10, 45)], np.float32
        ),
    )
    yield "at_border", np.array([(0, 0), (14, 0), (14, 14), (0, 14)], np.float32)
    # sub-pixel vertices: int truncation rather than rounding would shift this
    yield (
        "subpixel_vertices",
        np.array([(10.3, 10.7), (29.8, 10.2), (30.4, 29.1), (9.6, 30.9)], np.float32),
    )
    yield "tiny_2px", np.array([(50, 50), (52, 50), (52, 52), (50, 52)], np.float32)


def mask_geometries():
    """``(name, HxW label image)`` with a single label 1, same spread of shapes."""

    def blank():
        return np.zeros((HEIGHT, WIDTH), np.uint16)

    mask = blank()
    mask[10:30, 10:30] = 1
    yield "square", mask

    mask = blank()
    mask[40:44, 5:60] = 1
    yield "rod_thin", mask

    mask = blank()
    cv2.circle(mask, (50, 50), 15, 1, -1)
    yield "circle", mask

    mask = blank()
    cv2.ellipse(mask, (50, 50), (18, 6), 35.0, 0, 360, 1, -1)
    yield "ellipse_rotated", mask

    mask = blank()
    mask[10:45, 10:20] = 1
    mask[10:20, 10:40] = 1
    yield "L_concave", mask

    mask = blank()
    mask[0:15, 0:15] = 1
    yield "at_corner", mask

    mask = blank()
    mask[50:52, 50:52] = 1
    yield "tiny_2px", mask

    mask = blank()
    mask[20:21, 10:40] = 1
    yield "sliver_1px", mask


def contour_of(coords, cid=1, label=1, frame=0):
    return Contour(coords, 0.9, frame, cid, label)


def rasterize_exact(contours):
    """What every persistence path rasterizes with."""
    return frame_label_mask(
        list(contours), height=HEIGHT, width=WIDTH, exact_polygons=True
    )


class TestRasterisationIsExact(unittest.TestCase):
    """The persistence rasterizer must equal the full-frame one it replaced."""

    def test_matches_full_frame_rasterize(self):
        for name, coords in polygon_geometries():
            with self.subTest(geometry=name):
                cont = contour_of(coords)
                legacy = polygon_to_mask(cont.polygon, HEIGHT, WIDTH)
                fast = rasterize_exact([cont]) > 0
                np.testing.assert_array_equal(
                    fast, legacy, err_msg=f"{name} rasterized differently"
                )


class TestPolygonPersistencePreservesSize(unittest.TestCase):
    """save_segmentation -> load_segmentation must not move a single pixel."""

    def _roundtrip(self, overlay):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "segmentation"
            save_segmentation(path, overlay)
            return load_segmentation(path)

    def test_coordinates_and_mask_survive_exactly(self):
        for name, coords in polygon_geometries():
            with self.subTest(geometry=name):
                cont = contour_of(coords)
                before = rasterize_exact([cont])

                reloaded = self._roundtrip(Overlay([cont], frames=[0]))
                after = rasterize_exact(list(reloaded))

                np.testing.assert_array_equal(
                    np.asarray(next(iter(reloaded)).coordinates),
                    np.asarray(cont.coordinates),
                    err_msg=f"{name}: coordinates changed",
                )
                np.testing.assert_array_equal(
                    after, before, err_msg=f"{name}: rasterized size changed"
                )

    def test_many_cells_in_one_frame_keep_their_areas(self):
        """A full frame, not one cell at a time -- labels must not bleed."""
        contours = [
            contour_of(coords, cid=i + 1, label=i + 1)
            for i, (_, coords) in enumerate(polygon_geometries())
        ]
        overlay = Overlay(contours, frames=[0])

        before = rasterize_exact(contours)
        after = rasterize_exact(list(self._roundtrip(overlay)))

        np.testing.assert_array_equal(after, before)
        for cont in contours:
            self.assertEqual(
                int((after == cont.label).sum()),
                int((before == cont.label).sum()),
                f"label {cont.label} changed area",
            )


class TestMaskPersistencePreservesSize(unittest.TestCase):
    """A mask-backed segmentation must survive the polygon archive unchanged."""

    def test_area_survives_save_and_load(self):
        for name, mask in mask_geometries():
            with self.subTest(geometry=name):
                overlay = overlay_from_masks(mask[np.newaxis])
                expected = int(mask.astype(bool).sum())

                with tempfile.TemporaryDirectory() as td:
                    path = Path(td) / "segmentation"
                    save_segmentation(path, overlay)
                    reloaded = load_segmentation(path)

                got = int((rasterize_exact(list(reloaded)) > 0).sum())

                self.assertEqual(
                    got, expected, f"{name}: {expected} px became {got} px on reload"
                )

    def test_ctc_export_preserves_area(self):
        """The CTC label-mask export is the other persistence route."""
        for name, mask in mask_geometries():
            with self.subTest(geometry=name):
                overlay = overlay_from_masks(mask[np.newaxis])
                lookup = {cont.id: 0 for cont in overlay}

                exported = CTCTrackingHelper.convert_overlay_to_ctc_mask(
                    overlay, lookup, HEIGHT, WIDTH
                )

                self.assertEqual(
                    int((exported > 0).sum()),
                    int(mask.astype(bool).sum()),
                    f"{name}: CTC export changed the cell's area",
                )

    def test_ctc_masks_reload_with_the_same_area(self):
        """Written to disk as TIFFs and read back by the native reader."""
        import tifffile

        for name, mask in mask_geometries():
            with self.subTest(geometry=name):
                overlay = overlay_from_masks(mask[np.newaxis])
                lookup = {cont.id: 0 for cont in overlay}
                exported = CTCTrackingHelper.convert_overlay_to_ctc_mask(
                    overlay, lookup, HEIGHT, WIDTH
                )

                with tempfile.TemporaryDirectory() as td:
                    tifffile.imwrite(Path(td) / "mask0000.tif", exported)
                    reloaded = read_ctc_segmentation_native(Path(td))

                got = int((rasterize_exact(list(reloaded)) > 0).sum())
                self.assertEqual(
                    got,
                    int(mask.astype(bool).sum()),
                    f"{name}: area changed through the CTC round trip",
                )


class TestFragmentedCellsLoseTheirSmallerParts(unittest.TestCase):
    """The one case where the polygon archive cannot preserve size.

    A polygon has a single ring, so a label whose mask has disconnected
    components cannot be written faithfully: ``save_segmentation`` stores
    ``cont.coordinates``, which is the *largest* part (see
    ``acia.utils.largest_polygon``) and reports how many detections were
    affected. This is pre-existing and documented, pinned here so it stays a
    known, counted loss rather than becoming a silent one.

    The mask-based CTC export has no such limit and keeps both parts.
    """

    @staticmethod
    def _fragmented_mask():
        mask = np.zeros((HEIGHT, WIDTH), np.uint16)
        mask[5:12, 5:12] = 1  # 49 px, the larger part
        mask[60:66, 60:66] = 1  # 36 px, dropped by the polygon archive
        return mask

    def test_polygon_archive_keeps_only_the_largest_part(self):
        mask = self._fragmented_mask()
        overlay = overlay_from_masks(mask[np.newaxis])
        self.assertTrue(next(iter(overlay)).is_fragmented)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "segmentation"
            save_segmentation(path, overlay)
            reloaded = load_segmentation(path)

        self.assertEqual(int((rasterize_exact(list(reloaded)) > 0).sum()), 49)

    def test_ctc_export_keeps_both_parts(self):
        mask = self._fragmented_mask()
        overlay = overlay_from_masks(mask[np.newaxis])
        lookup = {cont.id: 0 for cont in overlay}

        exported = CTCTrackingHelper.convert_overlay_to_ctc_mask(
            overlay, lookup, HEIGHT, WIDTH
        )

        self.assertEqual(int((exported > 0).sum()), 49 + 36)


class TestDisplayRasterizerIsAllowedToDilate(unittest.TestCase):
    """Why persistence cannot take the renderers' shortcut.

    ``cv2.fillPoly`` is the cheaper default of ``frame_label_mask``. It is fine
    for a rendered frame and wrong for anything stored or measured; these
    numbers are the reason the two paths exist, and pinning them means a change
    of default cannot pass unnoticed.
    """

    def test_display_path_inflates_every_geometry(self):
        for name, coords in polygon_geometries():
            with self.subTest(geometry=name):
                cont = contour_of(coords)
                exact = int((rasterize_exact([cont]) > 0).sum())
                display = int(
                    (
                        frame_label_mask([cont], HEIGHT, WIDTH, exact_polygons=False)
                        > 0
                    ).sum()
                )
                self.assertGreater(
                    display, exact, f"{name}: expected the display path to dilate"
                )

    def test_thin_structures_suffer_most(self):
        """A 1 px dilation more than doubles a 2 px speck."""
        tiny = contour_of(
            np.array([(50, 50), (52, 50), (52, 52), (50, 52)], np.float32)
        )

        exact = int((rasterize_exact([tiny]) > 0).sum())
        display = int(
            (frame_label_mask([tiny], HEIGHT, WIDTH, exact_polygons=False) > 0).sum()
        )

        self.assertEqual(exact, 4)  # 2x2
        self.assertEqual(display, 9)  # 3x3


class TestRepeatedPersistenceIsStable(unittest.TestCase):
    """Geometry must survive *many* save/load cycles, not just one.

    A rasterizer can be exact once and still drift: each cycle re-traces the
    mask into a polygon and re-fills it, so any half-pixel bias compounds. The
    exact path is a fixed point -- the first round trip reaches a stable
    geometry and every later one reproduces it byte for byte.

    ``TestDisplayRasterizerWouldCompound`` shows what the alternative looks
    like, and is why this test exists.
    """

    CYCLES = 10

    def _save_load(self, overlay):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "segmentation"
            save_segmentation(path, overlay)
            return load_segmentation(path)

    def test_polygon_coordinates_are_bit_identical_across_cycles(self):
        """A polygon archive round trip must not perturb a single coordinate."""
        for name, coords in polygon_geometries():
            with self.subTest(geometry=name):
                overlay = Overlay([contour_of(coords)], frames=[0])
                original = np.asarray(next(iter(overlay)).coordinates).copy()

                for cycle in range(self.CYCLES):
                    overlay = self._save_load(overlay)
                    np.testing.assert_array_equal(
                        np.asarray(next(iter(overlay)).coordinates),
                        original,
                        err_msg=f"{name}: coordinates drifted at cycle {cycle + 1}",
                    )

    def test_mask_survives_repeated_mask_polygon_mask_cycles(self):
        """The round trip a re-segmented, re-saved dataset actually goes through.

        mask -> polygon archive -> reload -> rasterize -> mask -> ... Each cycle
        crosses both conversions, which is where a biased rasterizer compounds.
        """
        for name, mask in mask_geometries():
            with self.subTest(geometry=name):
                overlay = overlay_from_masks(mask[np.newaxis])
                expected = None

                for cycle in range(self.CYCLES):
                    reloaded = self._save_load(overlay)
                    label_mask = rasterize_exact(list(reloaded)) > 0

                    if expected is None:
                        expected = label_mask
                        self.assertEqual(
                            int(label_mask.sum()),
                            int(mask.astype(bool).sum()),
                            f"{name}: first cycle already changed the area",
                        )
                    else:
                        # the whole mask, not just its area: a shape that drifts
                        # while keeping its pixel count must fail too
                        np.testing.assert_array_equal(
                            label_mask,
                            expected,
                            err_msg=f"{name}: geometry drifted at cycle {cycle + 1}",
                        )

                    overlay = overlay_from_masks(
                        label_mask.astype(np.uint16)[np.newaxis]
                    )

    def test_many_cells_stay_stable_together(self):
        """A whole frame through repeated cycles -- areas must all hold."""
        contours = [
            contour_of(coords, cid=i + 1, label=i + 1)
            for i, (_, coords) in enumerate(polygon_geometries())
        ]
        overlay = Overlay(contours, frames=[0])
        expected = rasterize_exact(contours)

        for cycle in range(self.CYCLES):
            overlay = self._save_load(overlay)
            np.testing.assert_array_equal(
                rasterize_exact(list(overlay)),
                expected,
                err_msg=f"frame drifted at cycle {cycle + 1}",
            )


class TestDisplayRasterizerWouldCompound(unittest.TestCase):
    """Why the persistence paths must never take the renderers' shortcut.

    Fed back through the mask -> polygon -> mask loop, the inclusive fill adds a
    ring of pixels *every* cycle. This is the concrete failure the
    ``exact_polygons`` flag exists to prevent, pinned so nobody can flip the
    default for persistence without this test failing loudly.
    """

    @staticmethod
    def _cycle_areas(coords, exact, cycles=6):
        cont = contour_of(coords)
        areas = []
        for _ in range(cycles):
            label_mask = (
                frame_label_mask(
                    [cont], height=HEIGHT, width=WIDTH, exact_polygons=exact
                )
                > 0
            )
            areas.append(int(label_mask.sum()))
            instance = next(
                iter(overlay_from_masks(label_mask.astype(np.uint16)[np.newaxis]))
            )
            cont = contour_of(np.asarray(instance.coordinates, np.float32))
        return areas

    def test_exact_path_is_a_fixed_point(self):
        for name, coords in polygon_geometries():
            with self.subTest(geometry=name):
                areas = self._cycle_areas(coords, exact=True)
                self.assertEqual(
                    len(set(areas)), 1, f"{name}: exact path drifted -- {areas}"
                )

    def test_display_path_grows_every_cycle(self):
        """A thin rod nearly triples in six cycles; a square gains ~70%."""
        rod = dict(polygon_geometries())["rod_thin"]

        areas = self._cycle_areas(rod, exact=False)

        self.assertTrue(
            all(
                later > earlier
                for earlier, later in zip(areas, areas[1:], strict=False)
            ),
            f"expected monotonic growth on the display path, got {areas}",
        )
        self.assertGreater(areas[-1], 2 * areas[0])


if __name__ == "__main__":
    unittest.main()
