"""Test acia base functionality"""

import unittest
import warnings

import cv2
import numpy as np
import pytest

from acia import ureg
from acia.base import (
    ArrayImage,
    BaseImage,
    Contour,
    ImageSequenceSource,
    Instance,
    RegisteredSequenceSource,
)
from acia.registration import FrameTransform


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


def _textured_frame(seed: int = 0, size: int = 200) -> np.ndarray:
    """A blurred-noise frame with real texture -- mirrors
    ``tests/test_registration.py::_textured_frame``."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, (size, size), dtype=np.uint8).astype(np.float32)
    return cv2.GaussianBlur(frame, (3, 3), 0)


def _warp(frame: np.ndarray, dx: float, dy: float, theta: float) -> np.ndarray:
    """Warp ``frame`` by a rigid ``(dx, dy, theta)`` about its own center.

    Mirrors ``tests/test_registration.py::_warp`` (and
    ``RotatedCropSequenceSource``'s own warp-matrix convention): rotation
    about the frame center via ``cv2.getRotationMatrix2D`` plus an explicit
    translation added to the matrix's translation column.
    """
    h, w = frame.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, theta, 1.0)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR)


class _GraySource(ImageSequenceSource):
    """Minimal source yielding 2D grayscale (H, W) frames (mirrors
    ``tests/segm/test_crop_rotated.py::_GraySource``)."""

    def __init__(self, stack: np.ndarray):
        self.stack = stack  # (T, H, W)

    def get_frame(self, frame: int) -> BaseImage:
        return ArrayImage(self.stack[frame], frame=frame)

    @property
    def size_t(self) -> int:
        return self.stack.shape[0]

    @property
    def size_h(self) -> int:
        return self.stack.shape[1]

    @property
    def size_w(self) -> int:
        return self.stack.shape[2]

    @property
    def size_c(self) -> int:
        return 1

    @property
    def num_channels(self) -> int:
        return 1


class TestRegisteredSequenceSource:
    """`ImageSequenceSource.register()` / `RegisteredSequenceSource`."""

    def test_stored_transform_undoes_known_drift(self):
        reference = _textured_frame(seed=0)
        dx_true, dy_true, theta_true = 6.0, -4.0, 3.0
        drifted = _warp(reference, dx_true, dy_true, theta_true)

        # `apply_correction` inverts the same (dx, dy, theta) that produced
        # the forward warp above (see `_warp`'s and `apply_correction`'s
        # shared matrix convention), so the ground-truth undo transform is
        # exactly the one used to create `drifted`.
        transform = FrameTransform(dx=dx_true, dy=dy_true, theta=theta_true)

        stack = np.stack([reference, drifted])
        parent = _GraySource(stack)
        registered = parent.register({1: transform})

        corrected = registered.get_frame(1).raw
        uncorrected_diff = np.abs(
            drifted.astype(float) - reference.astype(float)
        ).mean()
        corrected_diff = np.abs(
            corrected.astype(float) - reference.astype(float)
        ).mean()
        assert corrected_diff < uncorrected_diff * 0.5

    def test_missing_transform_returns_uncorrected_and_warns(self):
        reference = _textured_frame(seed=1)
        stack = np.stack([reference, reference])
        parent = _GraySource(stack)
        registered = parent.register({})  # no stored transform for frame 0

        with pytest.warns(UserWarning, match="no stored correction"):
            frame = registered.get_frame(0).raw

        np.testing.assert_array_equal(frame, reference)

    def test_register_returns_registered_source_wrapping_parent(self):
        reference = _textured_frame(seed=2)
        stack = np.stack([reference])
        parent = _GraySource(stack)
        registered = parent.register({})

        assert isinstance(registered, RegisteredSequenceSource)
        assert registered.parent is parent
        assert registered.size_t == parent.size_t
        assert registered.size_h == parent.size_h
        assert registered.size_w == parent.size_w
        assert registered.size_c == parent.size_c
        assert registered.num_channels == parent.num_channels

    def test_pixel_size_and_timepoints_delegate_to_parent(self):
        reference = _textured_frame(seed=3)
        stack = np.stack([reference, reference])
        parent = _GraySource(stack).with_pixel_size(0.5 * ureg.micrometer)
        registered = parent.register({})

        assert registered.pixel_size == parent.pixel_size


class TestRegisteredSequenceSourceOnMissing:
    """`on_missing` policy for frames with no stored transform."""

    def test_rejects_an_unknown_policy(self):
        parent = _GraySource(np.stack([_textured_frame(seed=0)]))
        with pytest.raises(ValueError, match="on_missing"):
            parent.register({}, on_missing="improvise")

    def test_error_policy_raises_instead_of_degrading(self):
        parent = _GraySource(np.stack([_textured_frame(seed=1)]))
        registered = parent.register({}, on_missing="error")
        with pytest.raises(KeyError, match="no stored correction"):
            registered.get_frame(0)

    def test_nearest_policy_corrects_with_a_neighbours_transform(self):
        """A missing frame left uncorrected is off by the *full* accumulated
        drift; a neighbour's transform is far closer to right."""
        reference = _textured_frame(seed=2)
        dx, dy, theta = 6.0, -4.0, 3.0
        drifted = _warp(reference, dx, dy, theta)

        # Frame 1 has a transform; frame 2 (same drift) has none.
        stack = np.stack([reference, drifted, drifted])
        parent = _GraySource(stack)
        registered = parent.register(
            {1: FrameTransform(dx=dx, dy=dy, theta=theta)}, on_missing="nearest"
        )

        with pytest.warns(UserWarning, match="nearest available"):
            corrected = registered.get_frame(2).raw

        uncorrected_diff = np.abs(
            drifted.astype(float) - reference.astype(float)
        ).mean()
        corrected_diff = np.abs(
            corrected.astype(float) - reference.astype(float)
        ).mean()
        assert corrected_diff < uncorrected_diff * 0.5

    def test_nearest_policy_falls_back_to_uncorrected_when_nothing_is_stored(self):
        reference = _textured_frame(seed=3)
        parent = _GraySource(np.stack([reference]))
        registered = parent.register({}, on_missing="nearest")
        with pytest.warns(UserWarning, match="no stored correction"):
            frame = registered.get_frame(0).raw
        np.testing.assert_array_equal(frame, reference)

    def test_warns_once_per_index_not_once_per_read(self):
        """A lazy crop -> write pipeline reads each frame on every pass; one
        warning per pass would bury the signal it is meant to carry."""
        reference = _textured_frame(seed=4)
        parent = _GraySource(np.stack([reference, reference]))
        registered = parent.register({})

        with pytest.warns(UserWarning, match="no stored correction"):
            registered.get_frame(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            registered.get_frame(0)  # same index again: already reported

    def test_missing_frames_collects_every_gap_for_one_summary(self):
        reference = _textured_frame(seed=5)
        parent = _GraySource(np.stack([reference, reference, reference]))
        registered = parent.register({1: FrameTransform(dx=1.0, dy=0.0, theta=0.0)})

        assert registered.missing_frames == set()
        with pytest.warns(UserWarning):
            registered.get_frame(0)
            registered.get_frame(2)
        assert registered.missing_frames == {0, 2}


if __name__ == "__main__":
    unittest.main()
