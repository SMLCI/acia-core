"""Utils for segmentation testing"""

import unittest

import numpy as np
from shapely.geometry import Polygon

from acia.segm.utils import compute_indices
from acia.utils import mask_to_polygons, polygon_to_mask


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


if __name__ == "__main__":
    unittest.main()
