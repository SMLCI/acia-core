"""Utils for segmentation testing"""

import unittest

import numpy as np
from shapely.geometry import Polygon

from acia import ureg
from acia.base import Contour, Overlay
from acia.segm.utils import compute_indices, merge_cells_to_colonies
from acia.utils import mask_to_polygons, polygon_to_mask


def _square(
    x: float, y: float, frame: int, cont_id: int, size: float = 10.0
) -> Contour:
    """A ``size``x``size`` cell with its lower-left corner at ``(x, y)``."""
    coords = [[x, y], [x + size, y], [x + size, y + size], [x, y + size]]
    return Contour(np.array(coords, dtype=float), 1.0, frame, cont_id)


class TestIndexing(unittest.TestCase):
    """Test the linearization of z and t stacks"""

    def test_both(self):
        setup = dict(size_t=4, size_z=4)

        self.assertEqual(compute_indices(0, **setup), (0, 0))
        self.assertEqual(compute_indices(1, **setup), (0, 1))
        self.assertEqual(compute_indices(8, **setup), (2, 0))
        self.assertEqual(compute_indices(3, **setup), (0, 3))
        self.assertEqual(compute_indices(10, **setup), (2, 2))

    def test_only_t(self):
        setup = dict(size_t=50, size_z=1)

        for i in range(setup["size_t"]):
            self.assertEqual(compute_indices(i, **setup), (i, 0))

    def test_only_z(self):
        setup = dict(size_t=1, size_z=50)

        for i in range(setup["size_z"]):
            self.assertEqual(compute_indices(i, **setup), (0, i))


class TestMaskPolygon(unittest.TestCase):
    """Test consistent mask <-> polygon conversions"""

    def test_realistic_polygon_mask(self):
        """Test: polygon -> mask -> polygon -> mask transformation. Check wheter the area stays persistent"""

        polygon = Polygon(
            [
                [87.0, 312.0],
                [86.0, 313.0],
                [85.0, 313.0],
                [83.0, 315.0],
                [83.0, 317.0],
                [82.0, 318.0],
                [82.0, 320.0],
                [81.0, 321.0],
                [81.0, 324.0],
                [80.0, 325.0],
                [80.0, 329.0],
                [79.0, 330.0],
                [79.0, 335.0],
                [78.0, 336.0],
                [78.0, 341.0],
                [77.0, 342.0],
                [77.0, 349.0],
                [78.0, 350.0],
                [78.0, 351.0],
                [81.0, 354.0],
                [86.0, 354.0],
                [87.0, 353.0],
                [88.0, 353.0],
                [89.0, 352.0],
                [89.0, 351.0],
                [90.0, 350.0],
                [90.0, 349.0],
                [91.0, 348.0],
                [91.0, 344.0],
                [92.0, 343.0],
                [92.0, 339.0],
                [93.0, 338.0],
                [93.0, 333.0],
                [94.0, 332.0],
                [94.0, 328.0],
                [95.0, 327.0],
                [95.0, 324.0],
                [96.0, 323.0],
                [96.0, 316.0],
                [93.0, 313.0],
                [92.0, 313.0],
                [91.0, 312.0],
            ]
        )

        mask = polygon_to_mask(polygon, 500, 500)

        mask_area = np.sum(mask)
        polgon_area = polygon.area

        # np.testing.assert_array_equal(polygon.bounds, mask_bounds(mask))

        self.assertEqual(mask_area, polgon_area)

        re_poly = mask_to_polygons(mask)

        self.assertEqual(polygon.area, re_poly.area)
        np.testing.assert_array_equal(polygon.bounds, re_poly.bounds)

        re_mask = polygon_to_mask(re_poly, 500, 500)

        self.assertEqual(np.sum(mask), np.sum(re_mask))
        np.testing.assert_array_equal(mask, re_mask)

    @staticmethod
    def tets_mask_poly_iter():
        """Testing consistency of multiple mask -> polygon -> mask ... transformations"""

        masks = [np.load("tests/resources/mask.npy")]

        height, width = masks[0].shape
        polygons = [mask_to_polygons(masks[-1])]

        num_iters = 5
        for _ in range(num_iters):
            # convert polygon -> mask and mask -> polygon
            masks.append(polygon_to_mask(polygons[-1], height, width))
            polygons.append(mask_to_polygons(masks[-1]))

            # consisteny with first entry
            np.testing.assert_array_equal(masks[0], masks[-1])

            np.testing.assert_array_equal(polygons[0].centroid, polygons[-1].centroid)
            np.testing.assert_almost_equal(polygons[0].area, polygons[-1])

    def test_simple_mask_to_polygon(self):
        """Test conversion of a simple polygon to a mask and back"""

        polygon = Polygon([[0, 0], [10, 0], [10, 10], [0, 10]])

        polygon_area = polygon.area

        mask = polygon_to_mask(polygon, 100, 100)

        mask_area = np.sum(mask)

        self.assertEqual(polygon_area, mask_area)

        re_polygon = mask_to_polygons(mask)

        self.assertEqual(polygon.area, re_polygon.area)
        self.assertEqual(polygon.centroid, re_polygon.centroid)

        np.testing.assert_array_almost_equal(
            [re_polygon.centroid.x, re_polygon.centroid.y], [5, 5]
        )

    def test_empty_mask_returns_none(self):
        """Test that an empty mask returns None"""
        mask = np.zeros((100, 100), dtype=bool)

        result = mask_to_polygons(mask)

        self.assertIsNone(result)

    def test_single_pixel_mask(self):
        """Test mask with a single pixel"""
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True

        result = mask_to_polygons(mask)

        self.assertIsNotNone(result)
        # Single pixel should produce a polygon with area ~1
        self.assertGreater(result.area, 0)

    def test_multiple_disconnected_regions(self):
        """Test mask with multiple disconnected regions produces MultiPolygon"""
        from shapely.geometry import MultiPolygon

        mask = np.zeros((100, 100), dtype=bool)
        # Two separate regions
        mask[10:20, 10:20] = True
        mask[60:70, 60:70] = True

        result = mask_to_polygons(mask)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, MultiPolygon)
        self.assertEqual(len(result.geoms), 2)


class TestMergeCellsToColonies(unittest.TestCase):
    """Test colony blob merging from a single-cell overlay"""

    @staticmethod
    def _multi_colony_overlay(start_frame: int = 0) -> Overlay:
        """Frames with 1, 2 and 3 well-separated cells (one colony blob each)."""
        contours = [_square(10, 10, start_frame, 0)]
        contours += [
            _square(10, 10, start_frame + 1, 1),
            _square(200, 200, start_frame + 1, 2),
        ]
        contours += [
            _square(10, 10, start_frame + 2, 3),
            _square(200, 200, start_frame + 2, 4),
            _square(400, 400, start_frame + 2, 5),
        ]
        return Overlay(contours)

    def test_ids_are_unique_across_frames(self):
        """Blobs of the same frame must not share an id (property extraction joins on it)"""
        colonies = merge_cells_to_colonies(self._multi_colony_overlay(), expand=2)

        ids = [cont.id for cont in colonies]
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(set(ids)), len(ids))

    def test_blobs_per_frame(self):
        """Well-separated cells stay separate colony blobs"""
        colonies = merge_cells_to_colonies(self._multi_colony_overlay(), expand=2)

        frames = [cont.frame for cont in colonies]
        self.assertEqual([frames.count(f) for f in (0, 1, 2)], [1, 2, 3])

    def test_nearby_cells_merge_into_one_blob(self):
        """A large expand bridges the gap between neighbouring cells"""
        contours = [_square(10, 10, 0, 0), _square(25, 10, 0, 1)]
        colonies = merge_cells_to_colonies(Overlay(contours), expand=10)

        self.assertEqual(len(colonies), 1)

    def test_frames_are_not_renumbered(self):
        """The real frame numbers survive, also for an overlay not starting at 0"""
        colonies = merge_cells_to_colonies(
            self._multi_colony_overlay(start_frame=5), expand=2
        )

        frames = sorted({cont.frame for cont in colonies})
        self.assertEqual(frames, [5, 6, 7])

    def test_time_model_is_propagated(self):
        """The colony overlay stays calibrated on its own"""
        overlay = self._multi_colony_overlay().with_frame_interval(5 * ureg.minute)

        colonies = merge_cells_to_colonies(overlay, expand=2)

        self.assertIsNotNone(colonies.timepoints)
        np.testing.assert_allclose(
            colonies.timepoints.to("minute").magnitude, [0, 5, 10]
        )

    def test_frame_without_detections_is_skipped(self):
        """An empty frame yields no colony blob and does not shift the others"""
        contours = [_square(10, 10, 0, 0), _square(10, 10, 2, 1)]
        colonies = merge_cells_to_colonies(Overlay(contours), expand=2)

        self.assertEqual(sorted(cont.frame for cont in colonies), [0, 2])


if __name__ == "__main__":
    unittest.main()
