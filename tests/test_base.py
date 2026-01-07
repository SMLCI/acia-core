"""Test acia base functionality"""

import unittest

import numpy as np

from acia.base import Contour, Instance


class TestContour(unittest.TestCase):
    """Test contour functionality"""

    def test_center(self):
        contour = [[0, 0], [1, 0], [1, 1], [0, 1]]

        self.assertTrue(contour is not None)

        # simple contour
        np.testing.assert_array_equal(
            Contour(contour, 0, 0, 0).center, np.array([0.5, 0.5], dtype=np.float32)
        )

        # unequal point sampling
        contour = [[0, 0], [0.5, 0], [1, 0], [1, 1], [0, 1]]
        np.testing.assert_array_equal(
            Contour(contour, 0, 0, 0).center, np.array([0.5, 0.5], dtype=np.float32)
        )

    def test_rasterization(self):
        """Make sure that contour to mask rasterization preserves area"""

        contours = [[[0, 0], [10, 0], [10, 10], [0, 10]]]

        for coordinates in contours:
            cont = Contour(coordinates, -1, 0, -1)
            mask = cont.toMask(40, 40)

            self.assertEqual(cont.area, np.sum(mask))


class TestInstance(unittest.TestCase):
    """Test contour functionality"""

    def test_center(self):
        mask = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 3, 3, 0],
                [0, 3, 3, 3, 3],
                [0, 2, 2, 0, 0],
            ],
            dtype=np.uint8,
        )

        instance = Instance(mask, 0, 3)

        self.assertEqual(instance.area, 6)

        poly = instance.polygon

        self.assertEqual(poly.area, 6)

        instance = Instance(mask, 0, 2)
        self.assertEqual(instance.area, 2)

        poly = instance.polygon
        self.assertEqual(poly.area, 2)


if __name__ == "__main__":
    unittest.main()
