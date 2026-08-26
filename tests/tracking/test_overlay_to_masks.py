"""Tests for the Overlay -> tracker label-stack conversion.

``overlay_to_masks`` feeds every tracking backend (Trackastra, LapTrack, PyUAT,
Ultrack). It used to rasterize each cell over the whole frame; it now goes
through ``acia.segm.rasterize.frame_label_mask``. These tests pin the conversion
against a literal transcription of the old loop, so a future rewrite has to keep
producing the same stack.

The output is byte-identical to the old loop's: the tracking path asks
``frame_label_mask`` for ``exact_polygons``, so polygons keep rasterio's
pixel-centre rule rather than taking the renderers' ``cv2.fillPoly`` shortcut,
which fills inclusively and would dilate every cell by a pixel.

One deliberate behaviour change is tested explicitly rather than asserted away:
frames are now indexed absolutely, which fixes a shift for overlays whose first
detection is not on frame 0 (see ``TestFrameAlignment``).
"""

import contextlib
import unittest

import numpy as np

from acia.base import Contour, Instance, Overlay
from acia.segm.formats import overlay_from_masks
from acia.tracking.processor.utils import overlay_to_masks

HEIGHT = WIDTH = 64


def legacy_overlay_to_masks(segmentation: Overlay, height: int, width: int):
    """The pre-optimisation implementation, transcribed verbatim.

    ``overlay_to_masks`` -> ``Overlay.toMasks(binary_mask=False)``, inlined so the
    reference cannot drift when the library changes.
    """
    masks = []
    for frame_ov in segmentation.timeIterator():
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(frame_ov) > 0:
            local_mask = np.zeros((height, width), dtype=np.uint16)
            for i, cont in enumerate(frame_ov):
                cont_mask = cont.toMask(height=height, width=width)
                label = i + 1
                if cont.label is not None:
                    with contextlib.suppress(ValueError):
                        label = int(cont.label)
                cont_mask = cont_mask.astype(np.uint16) * label
                local_mask = np.maximum(cont_mask, local_mask)
            mask = local_mask
        masks.append(mask)
    return np.stack(masks)


def square_contour(x, y, size, frame, cid, label=None):
    """Axis-aligned square outline, as a polygon-backed contour."""
    coords = np.array(
        [(x, y), (x + size, y), (x + size, y + size), (x, y + size)], dtype=np.float32
    )
    return Contour(coords, 0.9, frame, cid, label)


def labelled_frame(*boxes, dtype=np.uint16):
    """Full-frame label image from ``(label, y, x, size)`` tuples."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=dtype)
    for label, y, x, size in boxes:
        mask[y : y + size, x : x + size] = label
    return mask


class TestAgainstLegacyImplementation(unittest.TestCase):
    """The new stack must match what the old per-cell loop produced."""

    def test_instance_overlay_is_identical(self):
        """Mask-backed instances go through the LUT/window paths -- exactly equal."""
        masks = np.stack(
            [
                labelled_frame((1, 5, 5, 10), (2, 30, 30, 8)),
                labelled_frame((1, 6, 6, 10), (2, 32, 31, 8), (3, 50, 12, 6)),
            ]
        )
        overlay = overlay_from_masks(masks)

        new = overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)
        old = legacy_overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)

        np.testing.assert_array_equal(new, old)

    def test_private_mask_instances_are_identical(self):
        """Instances owning separate masks take the window path, not the LUT."""
        shared = labelled_frame((1, 5, 5, 10), (2, 30, 30, 8))
        instances = [
            Instance(
                mask=np.where(shared == label, label, 0).astype(np.uint16),
                frame=0,
                label=label,
                id=label,
            )
            for label in (1, 2)
        ]
        # guard the precondition: these must NOT hit the shared-mask fast path
        self.assertIsNot(instances[0].mask, instances[1].mask)
        overlay = Overlay(instances)

        np.testing.assert_array_equal(
            overlay_to_masks(overlay, height=HEIGHT, width=WIDTH),
            legacy_overlay_to_masks(overlay, height=HEIGHT, width=WIDTH),
        )

    def test_contour_overlay_is_identical(self):
        """Polygons must rasterize exactly as they did before.

        The tracking path asks for ``exact_polygons``: windowed rasterio, same
        pixel-centre rule as the full-frame rasterize it replaced. The cheaper
        ``cv2.fillPoly`` the renderers use fills a closed polygon inclusively
        and would dilate every cell by a pixel -- a 10 px square covering 11 px
        -- which is geometry the tracker associates on.
        """
        overlay = Overlay(
            [
                square_contour(5, 5, 10, frame=0, cid=1, label=1),
                square_contour(30, 30, 8, frame=0, cid=2, label=2),
                square_contour(12, 40, 9, frame=1, cid=3, label=1),
            ]
        )

        new = overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)
        old = legacy_overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)

        np.testing.assert_array_equal(new, old)

    def test_cv2_rasterizer_would_have_dilated_cells(self):
        """Pin the reason the tracking path does not take the renderers' shortcut."""
        from acia.segm.rasterize import frame_label_mask

        cont = [square_contour(5, 5, 10, frame=0, cid=1, label=1)]
        exact = frame_label_mask(cont, HEIGHT, WIDTH, exact_polygons=True)
        fast = frame_label_mask(cont, HEIGHT, WIDTH, exact_polygons=False)

        self.assertEqual(int((exact == 1).sum()), 100)  # 10x10
        self.assertEqual(int((fast == 1).sum()), 121)  # 11x11 -- dilated

    def test_mixed_contour_and_instance_frame(self):
        """A frame holding both flavours has to interleave them by label.

        Neither the shared-mask LUT nor the batched polygon pass can handle this
        frame alone, so it exercises the fallback where instance windows and
        polygon fills are written into one buffer in label order.
        """
        mask = labelled_frame((4, 30, 30, 8))
        overlay = Overlay(
            [
                square_contour(5, 5, 10, frame=0, cid=1, label=1),
                Instance(mask=mask, frame=0, label=4, id=2),
                square_contour(40, 40, 6, frame=0, cid=3, label=9),
            ]
        )

        new = overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)
        old = legacy_overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)

        np.testing.assert_array_equal(new, old)
        self.assertEqual(sorted(np.unique(new)), [0, 1, 4, 9])

    def test_instance_mask_smaller_than_the_requested_frame(self):
        """An instance whose mask is not frame-sized must not break the stack.

        The shared-mask LUT returns an array shaped like the instance's mask, so
        it can only stand in for a frame-sized rasterisation when the mask is
        frame-sized.
        """
        small = np.zeros((HEIGHT // 2, WIDTH // 2), dtype=np.uint16)
        small[4:9, 4:9] = 1
        overlay = Overlay([Instance(mask=small, frame=0, label=1, id=1)])

        masks = overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)

        self.assertEqual(masks.shape, (1, HEIGHT, WIDTH))
        self.assertEqual(int((masks[0] == 1).sum()), 25)

    def test_label_fallback_matches_legacy(self):
        """None and non-integer labels fall back to the 1-based position."""
        for label in (None, "cell", 7):
            with self.subTest(label=label):
                overlay = Overlay(
                    [
                        square_contour(5, 5, 10, frame=0, cid=1, label=label),
                        square_contour(30, 30, 8, frame=0, cid=2, label=label),
                    ]
                )
                new = overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)
                old = legacy_overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)
                self.assertEqual(sorted(np.unique(new)), sorted(np.unique(old)))


class TestFrameAlignment(unittest.TestCase):
    """Frames are indexed absolutely, so masks line up with the image stack."""

    def test_leading_empty_frames_are_preserved(self):
        """A first detection on frame 3 must land in masks[3], not masks[0].

        The old implementation built the stack from ``timeIterator()``, which
        starts at the first *populated* frame -- handing the tracker a
        segmentation shifted three frames against its images.
        """
        overlay = Overlay([square_contour(5, 5, 10, frame=3, cid=1, label=1)])

        masks = overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)

        self.assertEqual(len(masks), 4)
        for empty in range(3):
            self.assertEqual(masks[empty].max(), 0, f"frame {empty} should be empty")
        self.assertEqual(masks[3].max(), 1)

        # the bug this fixes: the old code put frame 3's cell at index 0
        legacy = legacy_overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)
        self.assertEqual(len(legacy), 1)

    def test_trailing_empty_frames_from_declared_extent(self):
        """An explicit frame list keeps frames with no surviving cells."""
        overlay = Overlay(
            [square_contour(5, 5, 10, frame=0, cid=1, label=1)],
            frames=list(range(5)),
        )

        masks = overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)

        self.assertEqual(len(masks), 5)
        self.assertEqual(masks[0].max(), 1)
        self.assertEqual(masks[1:].max(), 0)

    def test_empty_overlay_returns_empty_stack(self):
        """An overlay with no detections must not raise."""
        masks = overlay_to_masks(Overlay([]), height=HEIGHT, width=WIDTH)
        self.assertEqual(masks.shape, (0, HEIGHT, WIDTH))


class TestLabelSemantics(unittest.TestCase):
    """Labelling rules the trackers depend on."""

    def test_higher_label_wins_on_overlap(self):
        """Overlapping cells resolve the same way np.maximum used to."""
        overlay = Overlay(
            [
                square_contour(10, 10, 20, frame=0, cid=1, label=1),
                square_contour(15, 15, 20, frame=0, cid=2, label=5),
            ]
        )

        masks = overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)

        # the overlap region belongs to the higher label
        self.assertEqual(masks[0][20, 20], 5)

    def test_label_zero_stays_background(self):
        """An explicit integer label of 0 is background, as it always was."""
        overlay = Overlay([square_contour(5, 5, 10, frame=0, cid=1, label=0)])

        masks = overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)

        self.assertEqual(masks.max(), 0)

    def test_large_labels_widen_the_stack(self):
        """Labels past uint16 must widen the dtype rather than wrap in it."""
        big = 70_000
        overlay = Overlay([square_contour(5, 5, 10, frame=0, cid=1, label=big)])

        masks = overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)

        self.assertEqual(masks.dtype, np.uint32)
        self.assertEqual(masks.max(), big)

    def test_default_dtype_is_uint16(self):
        """Ordinary label ranges keep the historical uint16 stack."""
        overlay = Overlay([square_contour(5, 5, 10, frame=0, cid=1, label=3)])
        self.assertEqual(
            overlay_to_masks(overlay, height=HEIGHT, width=WIDTH).dtype, np.uint16
        )

    def test_fragmented_instance_keeps_both_components(self):
        """A label split into two blobs must rasterize both parts."""
        mask = labelled_frame((1, 5, 5, 6), (1, 40, 40, 6))
        overlay = overlay_from_masks(mask[np.newaxis])

        masks = overlay_to_masks(overlay, height=HEIGHT, width=WIDTH)

        self.assertEqual(masks[0][7, 7], 1)
        self.assertEqual(masks[0][42, 42], 1)


if __name__ == "__main__":
    unittest.main()
