"""Tests for the fast mask-rasterization path shared by the mask renderers.

Covers acia.viz._frame_label_mask, _to_uint8_rgb and _blend_overlay, plus the
renderer-level behaviour that depends on them.
"""

import unittest

import numpy as np

from acia.base import Contour, Instance
from acia.segm.formats import overlay_from_masks
from acia.segm.local import THWCSequenceSource
from acia.viz import (
    _blend_overlay,
    _frame_label_mask,
    _to_uint8_rgb,
    colorize_instance_mask,
    render_segmentation_mask,
    render_tracking_mask,
)


def square_mask(height, width, label, y, x, size=10, dtype=np.uint16):
    """Full-frame label mask holding a single square instance."""
    mask = np.zeros((height, width), dtype=dtype)
    mask[y : y + size, x : x + size] = label
    return mask


class TestFrameLabelMask(unittest.TestCase):
    """_frame_label_mask must agree across its shared/per-instance/polygon paths."""

    def test_shared_mask_path_matches_per_instance_path(self):
        """The shared-mask fast path must equal the general per-instance path."""
        height = width = 60
        shared = np.zeros((height, width), dtype=np.uint16)
        for label in (1, 2, 3):
            shared[10 * label : 10 * label + 8, 5:20] = label

        # instances sharing one full-frame mask (as overlay_from_masks builds them)
        shared_instances = [
            Instance(mask=shared, frame=0, label=label, id=label) for label in (1, 2, 3)
        ]
        # same geometry, but each instance owning a private mask -> slow path
        private_instances = [
            Instance(
                mask=np.where(shared == label, label, 0).astype(np.uint16),
                frame=0,
                label=label,
                id=label,
            )
            for label in (1, 2, 3)
        ]

        # guard the precondition the fast path keys on
        self.assertTrue(all(i.mask is shared for i in shared_instances))
        self.assertFalse(
            all(i.mask is private_instances[0].mask for i in private_instances)
        )

        from_shared = _frame_label_mask(shared_instances, height, width)
        from_private = _frame_label_mask(private_instances, height, width)

        np.testing.assert_array_equal(from_shared, from_private)
        np.testing.assert_array_equal(from_shared, shared)

    def test_shared_mask_path_drops_labels_absent_from_the_overlay(self):
        """Labels present in the mask but filtered out of the overlay must not render."""
        shared = np.zeros((40, 40), dtype=np.uint16)
        shared[5:15, 5:15] = 1
        shared[20:30, 20:30] = 2

        # only instance 1 survives (e.g. after filtering)
        result = _frame_label_mask(
            [Instance(mask=shared, frame=0, label=1, id=1)], 40, 40
        )

        self.assertTrue(np.all(result[5:15, 5:15] == 1))
        self.assertTrue(np.all(result[20:30, 20:30] == 0))

    def test_higher_label_wins_on_overlap(self):
        """Overlapping instances keep np.maximum semantics: the higher label wins."""
        low = square_mask(40, 40, 1, 5, 5, size=20)
        high = square_mask(40, 40, 2, 10, 10, size=20)
        instances = [
            Instance(mask=high, frame=0, label=2, id="high"),
            Instance(mask=low, frame=0, label=1, id="low"),  # added after on purpose
        ]

        result = _frame_label_mask(instances, 40, 40)

        self.assertEqual(result[15, 15], 2)  # overlapping region
        self.assertEqual(result[7, 7], 1)  # low only
        self.assertEqual(result[27, 27], 2)  # high only

    def test_empty_overlay_returns_zero_mask(self):
        result = _frame_label_mask([], 20, 30)

        self.assertEqual(result.shape, (20, 30))
        self.assertFalse(result.any())

    def test_labels_above_uint16_range_survive(self):
        """Regression: labels were cast to uint16 and wrapped past 65535."""
        mask = np.zeros((40, 40), dtype=np.uint32)
        mask[5:15, 5:15] = 70000

        result = _frame_label_mask(
            [Instance(mask=mask, frame=0, label=70000, id=1)], 40, 40
        )

        self.assertEqual(result[10, 10], 70000)

    def test_contour_polygons_are_rasterized(self):
        """Contour overlays go through the polygon path rather than toMask."""
        coords = np.array([[10, 10], [30, 10], [30, 30], [10, 30]])
        contour = Contour(coordinates=coords, score=1.0, frame=0, id=1, label=7)

        result = _frame_label_mask([contour], 50, 50)

        self.assertEqual(result[20, 20], 7)
        self.assertEqual(result[45, 45], 0)

    def test_enumerate_fallback_labels_unlabelled_contours(self):
        """With enumerate_fallback, label-less contours get their 1-based position."""
        coords = np.array([[5, 5], [15, 5], [15, 15], [5, 15]])
        contours = [
            Contour(coordinates=coords + 20 * i, score=1.0, frame=0, id=i, label=None)
            for i in range(2)
        ]

        result = _frame_label_mask(contours, 60, 60, enumerate_fallback=True)

        self.assertEqual(result[10, 10], 1)
        self.assertEqual(result[30, 30], 2)


class TestToUint8Rgb(unittest.TestCase):
    """_to_uint8_rgb must always hand back a private HxWx3 uint8 buffer."""

    def test_uint8_rgb_input_is_copied(self):
        """Regression: renderers draw in place, so the result must not alias."""
        image = np.full((10, 10, 3), 50, dtype=np.uint8)

        result = _to_uint8_rgb(image)
        result[0, 0] = 255

        self.assertIsNot(result, image)
        self.assertEqual(image[0, 0, 0], 50)

    def test_grayscale_is_expanded_to_rgb(self):
        result = _to_uint8_rgb(np.full((10, 12), 7, dtype=np.uint8))

        self.assertEqual(result.shape, (10, 12, 3))

    def test_single_channel_is_expanded_to_rgb(self):
        result = _to_uint8_rgb(np.full((10, 12, 1), 7, dtype=np.uint8))

        self.assertEqual(result.shape, (10, 12, 3))

    def test_uint16_is_scaled_into_the_uint8_range(self):
        result = _to_uint8_rgb(np.full((4, 4), 30000, dtype=np.uint16))

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result[0, 0, 0], 30000 // 256)

    def test_float_is_scaled_and_clipped(self):
        image = np.array([[0.0, 0.5, 2.0]], dtype=np.float32)

        result = _to_uint8_rgb(image)

        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result[0, :, 0], [0, 127, 255])


class TestBlendOverlay(unittest.TestCase):
    """_blend_overlay decides foreground from the label mask, not the colors."""

    def test_background_pixels_are_untouched(self):
        image = np.full((10, 10, 3), 200, dtype=np.uint8)
        label_mask = np.zeros((10, 10), dtype=np.uint32)
        label_mask[2:5, 2:5] = 1
        colored = colorize_instance_mask(
            label_mask, color_lut=np.array([[0, 0, 0], [100, 100, 100]], dtype=np.uint8)
        )

        result = _blend_overlay(image, colored, label_mask, alpha=0.5)

        self.assertEqual(result[8, 8, 0], 200)

    def test_black_coloured_instance_still_counts_as_foreground(self):
        """Regression: foreground was derived from the colors, so (0,0,0) vanished."""
        image = np.full((10, 10, 3), 200, dtype=np.uint8)
        label_mask = np.zeros((10, 10), dtype=np.uint32)
        label_mask[2:5, 2:5] = 1
        black_lut = np.zeros((2, 3), dtype=np.uint8)
        colored = colorize_instance_mask(label_mask, color_lut=black_lut)

        result = _blend_overlay(image, colored, label_mask, alpha=0.5)

        self.assertEqual(result[3, 3, 0], 100)  # 0.5 * 200 + 0.5 * 0
        self.assertEqual(result[8, 8, 0], 200)


class TestMaskRenderersDoNotMutateSource(unittest.TestCase):
    """Rendering is read-only with respect to the caller's image data."""

    def _source_and_overlay(self):
        masks = np.zeros((2, 40, 40), dtype=np.uint16)
        masks[:, 5:15, 5:15] = 1
        source = THWCSequenceSource(np.full((2, 40, 40, 3), 60, dtype=np.uint8))
        return source, overlay_from_masks(masks)

    def test_render_tracking_mask_leaves_source_untouched(self):
        source, overlay = self._source_and_overlay()
        pristine = source.image_stack.copy()

        render_tracking_mask(source, overlay, show_label_numbers=True)

        np.testing.assert_array_equal(source.image_stack, pristine)

    def test_render_segmentation_mask_leaves_source_untouched(self):
        source, overlay = self._source_and_overlay()
        pristine = source.image_stack.copy()

        render_segmentation_mask(source, overlay)

        np.testing.assert_array_equal(source.image_stack, pristine)


class TestRenderTrackingMaskImageFormats(unittest.TestCase):
    """render_tracking_mask normalizes the frame before blending."""

    def test_uint16_source_gets_a_visible_overlay(self):
        """Regression: 0-255 overlay colors were invisible against a 0-65535 frame."""
        masks = np.zeros((1, 40, 40), dtype=np.uint16)
        masks[0, 5:15, 5:15] = 1
        source = THWCSequenceSource(np.full((1, 40, 40, 3), 30000, dtype=np.uint16))

        result = (
            render_tracking_mask(source, overlay_from_masks(masks)).get_frame(0).raw
        )

        self.assertEqual(result.dtype, np.uint8)
        foreground = result[10, 10]
        background = result[30, 30]
        self.assertFalse(np.array_equal(foreground, background))

    def test_labels_above_uint16_range_are_rendered(self):
        """Regression: a track label of 70000 wrapped to 4464 and could vanish."""
        masks = np.zeros((1, 40, 40), dtype=np.uint32)
        masks[0, 5:15, 5:15] = 70000
        source = THWCSequenceSource(np.full((1, 40, 40, 3), 50, dtype=np.uint8))

        result = (
            render_tracking_mask(source, overlay_from_masks(masks)).get_frame(0).raw
        )

        self.assertFalse(np.all(result[5:15, 5:15] == 50))
        self.assertTrue(np.all(result[30:, 30:] == 50))


class TestInstanceDerivedValueCaching(unittest.TestCase):
    """Instance caches its center; the caches must follow mask/label changes."""

    def test_center_is_cached(self):
        instance = Instance(mask=square_mask(40, 40, 1, 10, 10), frame=0, label=1, id=1)

        self.assertIs(instance.center, instance.center)

    def test_center_cache_is_dropped_when_the_mask_changes(self):
        instance = Instance(mask=square_mask(40, 40, 1, 5, 5), frame=0, label=1, id=1)
        before = instance.center

        instance.mask = square_mask(40, 40, 1, 25, 25)

        self.assertNotEqual(instance.center, before)

    def test_center_cache_is_dropped_when_the_label_changes(self):
        """The relabel in acia.tracking.utils rewrites label and mask together."""
        mask = np.zeros((40, 40), dtype=np.uint16)
        mask[5:15, 5:15] = 1
        mask[25:35, 25:35] = 2
        instance = Instance(mask=mask, frame=0, label=1, id=1)
        before = instance.center

        instance.label = 2

        self.assertNotEqual(instance.center, before)


if __name__ == "__main__":
    unittest.main()
