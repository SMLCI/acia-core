"""Fluorescence extraction must not change its numbers when it gets faster.

``FluorescenceEx.extract_fluorescence`` used to build a frame-sized
``np.ma.masked_array`` per cell per channel and re-decode the channel inside the
contour loop. It now gathers each cell's pixels inside its bounding box via
``acia.segm.rasterize.contour_pixels``.

That is a measurement path, so the bar here is exact equality with the old
implementation -- not "close enough". These tests pin the pixel *set* and the
order it is returned in, because a summarising operator is free to care about
both (``np.median`` does not, a percentile-of-first-k would).
"""

import unittest

import numpy as np
from numpy import ma

from acia.base import Contour, Instance, Overlay
from acia.segm.formats import overlay_from_masks
from acia.segm.local import THWCSequenceSource
from acia.segm.rasterize import contour_pixels

HEIGHT = WIDTH = 48


def legacy_pixels(cont, raw_image):
    """The pre-optimisation pixel gather, transcribed verbatim."""
    height, width = raw_image.shape[:2]
    roi_mask = cont.toMask(height=height, width=width)
    masked_roi: ma.MaskedArray = ma.masked_array(raw_image, mask=~roi_mask)
    return masked_roi.compressed()


def gradient_image(seed=0):
    """A frame where every pixel differs, so a wrong gather cannot pass by luck."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4096, size=(HEIGHT, WIDTH)).astype(np.uint16)


def square_contour(x, y, size, cid, label=None):
    coords = np.array(
        [(x, y), (x + size, y), (x + size, y + size), (x, y + size)], dtype=np.float32
    )
    return Contour(coords, 0.9, 0, cid, label)


class TestContourPixelsMatchesLegacy(unittest.TestCase):
    """The gathered values -- and their order -- must be unchanged."""

    def test_polygon_contour(self):
        image = gradient_image()
        cont = square_contour(6, 9, 11, cid=1, label=1)

        np.testing.assert_array_equal(
            contour_pixels(cont, image), legacy_pixels(cont, image)
        )

    def test_polygon_contour_touching_the_border(self):
        """A cell against the frame edge must not lose or gain pixels."""
        image = gradient_image(seed=3)
        cont = square_contour(0, 0, 7, cid=1, label=1)

        np.testing.assert_array_equal(
            contour_pixels(cont, image), legacy_pixels(cont, image)
        )

    def test_mask_backed_instance(self):
        image = gradient_image(seed=1)
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
        mask[10:20, 12:25] = 1
        mask[30:35, 5:9] = 2
        overlay = overlay_from_masks(mask[np.newaxis])

        for cont in overlay:
            with self.subTest(label=cont.label):
                np.testing.assert_array_equal(
                    contour_pixels(cont, image), legacy_pixels(cont, image)
                )

    def test_fragmented_instance_gathers_every_component(self):
        """A label split into two blobs must contribute both."""
        image = gradient_image(seed=2)
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
        mask[5:10, 5:10] = 1
        mask[30:34, 30:34] = 1
        overlay = overlay_from_masks(mask[np.newaxis])
        cont = next(iter(overlay))

        got = contour_pixels(cont, image)

        np.testing.assert_array_equal(got, legacy_pixels(cont, image))
        self.assertEqual(len(got), 25 + 16)

    def test_empty_instance_returns_nothing(self):
        """A label absent from its mask yields no pixels rather than raising."""
        image = gradient_image()
        empty = Instance(
            mask=np.zeros((HEIGHT, WIDTH), dtype=np.uint16), frame=0, label=7, id=7
        )

        self.assertEqual(len(contour_pixels(empty, image)), 0)


class TestExtractFluorescenceUnchanged(unittest.TestCase):
    """End-to-end: the extractor's numbers must be identical."""

    def _legacy_extract(self, overlay, image, channels, channel_names, op):
        rows = []
        for cont in overlay:
            row = {"id": cont.id}
            for ch_id, channel in enumerate(channels):
                raw_image = image.get_channel(channel)
                row[channel_names[ch_id]] = op(legacy_pixels(cont, raw_image))
            rows.append(row)
        return rows

    def _check(self, overlay):
        from acia.analysis import FluorescenceEx

        rng = np.random.default_rng(7)
        stack = rng.integers(0, 4096, size=(1, HEIGHT, WIDTH, 2)).astype(np.uint16)
        source = THWCSequenceSource(stack)
        image = source.get_frame(0)

        channels, names = [0, 1], ["gfp", "rfp"]

        new = FluorescenceEx.extract_fluorescence(
            overlay, image, channels, names, np.median
        )
        old = self._legacy_extract(overlay, image, channels, names, np.median)

        self.assertEqual(len(new), len(old))
        for name in names:
            np.testing.assert_array_equal(
                new[name].to_numpy(),
                np.array([row[name] for row in old]),
                err_msg=f"channel {name} changed",
            )

    def test_contour_overlay(self):
        self._check(
            Overlay(
                [
                    square_contour(6, 9, 11, cid=1, label=1),
                    square_contour(25, 20, 9, cid=2, label=2),
                ]
            )
        )

    def test_instance_overlay(self):
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
        mask[10:20, 12:25] = 1
        mask[30:35, 5:9] = 2
        self._check(overlay_from_masks(mask[np.newaxis]))


if __name__ == "__main__":
    unittest.main()
