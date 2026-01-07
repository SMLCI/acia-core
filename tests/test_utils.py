"""
Testing global utilities
"""

import unittest
from functools import partial

import numpy as np

from acia.utils import lut_mapping

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


if __name__ == "__main__":
    unittest.main()
