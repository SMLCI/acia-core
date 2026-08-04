"""
Testing global utilities
"""

import unittest
from functools import partial

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box

from acia.utils import largest_polygon, lut_mapping, mask_to_polygons

# lut function for mapping from [0, 1000] to [0, 255]
lut_func = partial(lut_mapping, in_min=0, in_max=1000, out_min=0, out_max=255)


class TestMappingLUT(unittest.TestCase):
    """Test mapping lut functionality"""

    def test_number_mapping(self):
        """Test raw number mapping"""
        self.assertEqual(lut_func(0), 0)
        self.assertEqual(lut_func(1000), 255)
        self.assertEqual(lut_func(-5), 0)
        self.assertEqual(lut_func(1005), 255)

        self.assertEqual(lut_func(np.array([1]), dtype=np.uint8), 0)

    def test_image_mapping(self):
        """Test full image mapping"""
        image = np.array([[5, 1000], [16000, 5000]], dtype=np.int16)

        mapped_image = lut_func(image, dtype=np.uint8)

        for gt, pred in zip(image.flatten(), mapped_image.flatten(), strict=False):
            self.assertEqual(
                pred, np.floor(np.clip(gt, 0, 1000) / 1000 * 255).astype(np.uint8)
            )


class TestLargestPolygon(unittest.TestCase):
    """`largest_polygon` reduces a multi-part polygon to one outline."""

    def test_none_passes_through(self):
        self.assertIsNone(largest_polygon(None))

    def test_single_polygon_is_returned_unchanged(self):
        poly = box(0, 0, 2, 2)
        self.assertIs(largest_polygon(poly), poly)

    def test_multipolygon_yields_its_largest_part(self):
        big, small = box(0, 0, 10, 10), box(20, 20, 21, 21)
        result = largest_polygon(MultiPolygon([small, big]))
        self.assertIsInstance(result, Polygon)
        self.assertEqual(result.area, big.area)

    def test_empty_multipolygon_yields_none(self):
        self.assertIsNone(largest_polygon(MultiPolygon([])))

    def test_disconnected_mask_produces_a_multipolygon(self):
        """The upstream condition this exists for: two blobs sharing a label
        come back from `mask_to_polygons` as a MultiPolygon, which has no
        `exterior` for a caller needing a single closed outline."""
        mask = np.zeros((20, 20), dtype=bool)
        mask[3:8, 3:8] = True
        mask[12:15, 12:15] = True
        poly = mask_to_polygons(mask)
        self.assertIsInstance(poly, MultiPolygon)
        self.assertFalse(hasattr(poly, "exterior"))
        self.assertIsInstance(largest_polygon(poly), Polygon)


if __name__ == "__main__":
    unittest.main()
