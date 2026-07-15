"""Unit tests for :mod:`acia.registration` (5-way registration-method comparison).

Each concrete :class:`~acia.registration.RegistrationMethod` gets a
synthetic-ground-truth test: a structured/textured frame is warped by a known
``(dx, dy, theta)`` via ``cv2.warpAffine`` (built the same way
:class:`~acia.base.RotatedCropSequenceSource` builds its warp matrix -- a
rotation about the frame center via ``cv2.getRotationMatrix2D`` plus an
explicit translation added to the matrix's translation column), and the
method is asserted to recover that transform within tolerance. The shared
I/O edge cases (zero motion, multi-channel input, blank-frame raise,
per-method insufficient-signal raise) are covered once each, reusing methods
across cases where the spec allows.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from acia.base import RotatedCropSpec
from acia.registration import (
    FeatureRANSACEuclidean,
    FrameTransform,
    GradientECC,
    HoughLineRigidFit,
    MaskedTemplateCorrelation,
    PhaseCorrelationHighpass,
    RegistrationError,
    _parabolic_refine,
    apply_correction,
    run_comparison,
)

SIZE = 200


def _warp(frame: np.ndarray, dx: float, dy: float, theta: float) -> np.ndarray:
    """Warp ``frame`` by a rigid ``(dx, dy, theta)`` about its own center.

    Mirrors the rotation convention used throughout ``acia.base``
    (``cv2.getRotationMatrix2D``, CCW degrees) plus an explicit translation
    added to the matrix's translation column -- the same pattern
    :class:`~acia.base.RotatedCropSequenceSource._warp` uses for its crop
    matrix, just without the crop (same-size in, same-size out).
    """
    h, w = frame.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, theta, 1.0)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR)


def _textured_frame(seed: int = 0, size: int = SIZE) -> np.ndarray:
    """A blurred-noise frame with real texture but no straight edges."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, (size, size), dtype=np.uint8).astype(np.float32)
    return cv2.GaussianBlur(frame, (3, 3), 0)


def _structured_frame(seed: int = 1, size: int = SIZE) -> np.ndarray:
    """A noise + grid-lines + filled-rectangles frame with corners and lines.

    Used for the feature/line/gradient-based methods, which need real
    structure (corners for ORB, straight edges for Hough, sharp gradients
    for ECC) to lock onto.
    """
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 80, (size, size), dtype=np.uint8).astype(np.float32)
    for x in range(0, size, 20):
        cv2.line(frame, (x, 0), (x, size), 255, 1)
    for y in range(0, size, 20):
        cv2.line(frame, (0, y), (size, y), 255, 1)
    cv2.rectangle(frame, (30, 30), (80, 90), 200, -1)
    cv2.rectangle(frame, (120, 140), (170, 180), 180, -1)
    return frame.astype(np.uint8)


def _grid_frame(seed: int = 4, size: int = SIZE) -> np.ndarray:
    """A sparse, thin-line grid frame tuned for :class:`HoughLineRigidFit`.

    Fewer, thinner, more widely-spaced lines than :func:`_structured_frame`
    so ``cv2.HoughLinesP`` reliably detects one clean segment per physical
    grid line (line density/thickness tuning to make the test reliable, not
    a change to the acceptance tolerance).
    """
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 15, (size, size), dtype=np.uint8).astype(np.float32)
    for x in range(30, size, 45):
        cv2.line(frame, (x, 0), (x, size), 255, 1)
    for y in range(30, size, 45):
        cv2.line(frame, (0, y), (size, y), 255, 1)
    return frame.astype(np.uint8)


def _blank_frame(size: int = SIZE, value: int = 128) -> np.ndarray:
    """A uniform-intensity (textureless) frame."""
    return np.full((size, size), value, dtype=np.uint8)


def _single_line_frame(seed: int = 5, size: int = SIZE) -> np.ndarray:
    """A frame with exactly one clean, matchable straight line.

    Used to exercise :class:`~acia.registration.HoughLineRigidFit`'s "fewer
    than 2 matched line pairs" branch: with only one line (one orientation)
    available, no rigid transform can be constrained regardless of matching.
    """
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 15, (size, size), dtype=np.uint8).astype(np.float32)
    cv2.line(frame, (100, 0), (100, size), 255, 2)
    return frame.astype(np.uint8)


def _default_mask_rect() -> RotatedCropSpec:
    return RotatedCropSpec(center=(100.0, 100.0), size=(60, 60), angle=0.0)


class TestFrameTransform(unittest.TestCase):
    """Round-trip and default-field behavior of the plain data container."""

    def test_to_dict_from_dict_roundtrip(self):
        transform = FrameTransform(dx=1.5, dy=-2.25, theta=3.0)
        data = transform.to_dict()
        self.assertEqual(data, {"dx": 1.5, "dy": -2.25, "theta": 3.0})
        self.assertEqual(FrameTransform.from_dict(data), transform)

    def test_theta_defaults_to_zero_but_is_explicit_in_to_dict(self):
        transform = FrameTransform(dx=1.0, dy=2.0)
        self.assertEqual(transform.theta, 0.0)
        self.assertEqual(transform.to_dict()["theta"], 0.0)

    def test_from_dict_defaults_missing_theta(self):
        transform = FrameTransform.from_dict({"dx": 1.0, "dy": 2.0})
        self.assertEqual(transform.theta, 0.0)


class TestPhaseCorrelationHighpass(unittest.TestCase):
    """Translation-only recovery via high-pass phase correlation."""

    def test_translation_happy_path(self):
        reference = _textured_frame(seed=0)
        dx_true, dy_true = 6.3, -4.1
        frame = _warp(reference, dx_true, dy_true, 0.0)

        transform = PhaseCorrelationHighpass().estimate(reference, frame)

        self.assertAlmostEqual(transform.dx, dx_true, delta=0.5)
        self.assertAlmostEqual(transform.dy, dy_true, delta=0.5)
        self.assertEqual(transform.theta, 0.0)

    def test_zero_motion(self):
        reference = _textured_frame(seed=0)
        transform = PhaseCorrelationHighpass().estimate(reference, reference)
        self.assertAlmostEqual(transform.dx, 0.0, delta=0.5)
        self.assertAlmostEqual(transform.dy, 0.0, delta=0.5)
        self.assertEqual(transform.theta, 0.0)

    def test_blank_frame_raises(self):
        blank = _blank_frame()
        with self.assertRaises(RegistrationError):
            PhaseCorrelationHighpass().estimate(blank, blank)

    def test_multi_channel_input_matches_grayscale_accuracy(self):
        reference_gray = _textured_frame(seed=0)
        dx_true, dy_true = 6.3, -4.1
        frame_gray = _warp(reference_gray, dx_true, dy_true, 0.0)

        for channels in (1, 3, 4):
            with self.subTest(channels=channels):
                reference = np.stack([reference_gray] * channels, axis=-1)
                frame = np.stack([frame_gray] * channels, axis=-1)
                transform = PhaseCorrelationHighpass().estimate(reference, frame)
                self.assertAlmostEqual(transform.dx, dx_true, delta=0.5)
                self.assertAlmostEqual(transform.dy, dy_true, delta=0.5)
                self.assertEqual(transform.theta, 0.0)


class TestMaskedTemplateCorrelation(unittest.TestCase):
    """Translation-only recovery via masked normalized template matching."""

    def test_translation_happy_path(self):
        reference = _textured_frame(seed=3)
        dx_true, dy_true = 7.2, -5.4
        frame = _warp(reference, dx_true, dy_true, 0.0)

        method = MaskedTemplateCorrelation(_default_mask_rect(), search_margin=30)
        transform = method.estimate(reference, frame)

        self.assertAlmostEqual(transform.dx, dx_true, delta=0.5)
        self.assertAlmostEqual(transform.dy, dy_true, delta=0.5)
        self.assertEqual(transform.theta, 0.0)

    def test_zero_motion(self):
        reference = _textured_frame(seed=3)
        method = MaskedTemplateCorrelation(_default_mask_rect(), search_margin=30)
        transform = method.estimate(reference, reference)
        self.assertAlmostEqual(transform.dx, 0.0, delta=0.5)
        self.assertAlmostEqual(transform.dy, 0.0, delta=0.5)
        self.assertEqual(transform.theta, 0.0)

    def test_blank_frame_raises(self):
        blank = _blank_frame()
        method = MaskedTemplateCorrelation(_default_mask_rect(), search_margin=30)
        with self.assertRaises(RegistrationError):
            method.estimate(blank, blank)

    def test_shift_beyond_search_margin_raises(self):
        reference = _textured_frame(seed=3)
        # true shift (50px) far exceeds the tiny search_margin below
        frame = _warp(reference, 50.0, 0.0, 0.0)
        method = MaskedTemplateCorrelation(_default_mask_rect(), search_margin=8)
        with self.assertRaises(RegistrationError):
            method.estimate(reference, frame)

    def test_low_score_rejected(self):
        """Genuinely dissimilar content, found within the search window,
        must be rejected on score -- distinct from the search-margin-exceeded
        (edge-of-window) rejection above."""
        reference = _textured_frame(seed=3)
        dissimilar_frame = _textured_frame(seed=99)  # unrelated content
        method = MaskedTemplateCorrelation(
            _default_mask_rect(), search_margin=30, min_score=0.5
        )
        with self.assertRaisesRegex(RegistrationError, "min_score"):
            method.estimate(reference, dissimilar_frame)

    def test_translation_happy_path_with_rotated_mask_rect(self):
        """A non-zero ``mask_rect.angle`` must not corrupt the recovered
        translation: the search window has to be cropped with the same
        rotation as the template (see registration.py's P1 fix)."""
        reference = _textured_frame(seed=3)
        dx_true, dy_true = 6.0, -3.0
        frame = _warp(reference, dx_true, dy_true, 0.0)

        mask_rect = RotatedCropSpec(center=(100.0, 100.0), size=(60, 60), angle=15.0)
        method = MaskedTemplateCorrelation(mask_rect, search_margin=30)
        transform = method.estimate(reference, frame)

        self.assertAlmostEqual(transform.dx, dx_true, delta=0.5)
        self.assertAlmostEqual(transform.dy, dy_true, delta=0.5)
        self.assertEqual(transform.theta, 0.0)


class TestHoughLineRigidFit(unittest.TestCase):
    """Rigid recovery via straight-edge (Hough line) matching."""

    def test_rigid_happy_path(self):
        reference = _grid_frame(seed=4)
        dx_true, dy_true, theta_true = 4.0, -2.5, 3.0
        frame = _warp(reference, dx_true, dy_true, theta_true)

        transform = HoughLineRigidFit().estimate(reference, frame)

        self.assertAlmostEqual(transform.dx, dx_true, delta=0.5)
        self.assertAlmostEqual(transform.dy, dy_true, delta=0.5)
        self.assertAlmostEqual(transform.theta, theta_true, delta=0.5)

    def test_zero_motion(self):
        reference = _grid_frame(seed=4)
        transform = HoughLineRigidFit().estimate(reference, reference)
        self.assertAlmostEqual(transform.dx, 0.0, delta=0.5)
        self.assertAlmostEqual(transform.dy, 0.0, delta=0.5)
        self.assertAlmostEqual(transform.theta, 0.0, delta=0.5)

    def test_blank_frame_raises(self):
        blank = _blank_frame()
        with self.assertRaises(RegistrationError):
            HoughLineRigidFit().estimate(blank, blank)

    def test_no_lines_detected_raises(self):
        # a gentle intensity ramp has no straight edges strong enough for Canny
        ramp = np.tile(np.linspace(0, 40, SIZE), (SIZE, 1)).astype(np.uint8)
        with self.assertRaises(RegistrationError):
            HoughLineRigidFit().estimate(ramp, ramp)

    def test_ambiguous_periodic_drift_raises(self):
        """A periodic (evenly-spaced) grid, warped by drift large enough to
        make several lines' nearest-reference-candidate genuinely ambiguous,
        must raise rather than silently lock onto the wrong line.

        A pure axis-aligned translation on this specific (small, 4-line)
        grid always leaves one boundary line as an unambiguous anchor (it
        has no reference candidate beyond the grid's edge to be confused
        with), so the fit degrades gracefully to the correct answer even
        past half the grid spacing -- a good outcome, not a bug. Adding a
        modest rotation on top (still a physically plausible "translation +
        slight rotation" drift) removes that incidental single-anchor
        rescue and reliably exercises the *genuinely* ambiguous case the
        ambiguity/reuse-conflict guards in ``_match_lines`` exist for.
        """
        reference = _grid_frame(seed=4)
        # drift far exceeds half the 45px grid spacing on both axes
        frame = _warp(reference, -31.0, -12.0, 8.0)
        with self.assertRaises(RegistrationError):
            HoughLineRigidFit().estimate(reference, frame)

    def test_fewer_than_two_matched_pairs_raises(self):
        """A scene with exactly one clean matchable line -- as opposed to
        the all-(near-)parallel-lines rank-deficiency case -- cannot supply
        the two required pairs at all."""
        reference = _single_line_frame(seed=5)
        frame = _warp(reference, 3.0, 0.0, 0.0)
        with self.assertRaisesRegex(
            RegistrationError, "fewer than 2 matched line pairs"
        ):
            HoughLineRigidFit().estimate(reference, frame)


class TestFeatureRANSACEuclidean(unittest.TestCase):
    """Rigid recovery via ORB features + RANSAC-fit Euclidean transform."""

    def test_rigid_happy_path(self):
        reference = _structured_frame(seed=1)
        dx_true, dy_true, theta_true = 5.0, -3.0, 4.0
        frame = _warp(reference, dx_true, dy_true, theta_true)

        transform = FeatureRANSACEuclidean().estimate(reference, frame)

        self.assertAlmostEqual(transform.dx, dx_true, delta=0.5)
        self.assertAlmostEqual(transform.dy, dy_true, delta=0.5)
        self.assertAlmostEqual(transform.theta, theta_true, delta=0.5)

    def test_zero_motion(self):
        reference = _structured_frame(seed=1)
        transform = FeatureRANSACEuclidean().estimate(reference, reference)
        self.assertAlmostEqual(transform.dx, 0.0, delta=0.5)
        self.assertAlmostEqual(transform.dy, 0.0, delta=0.5)
        self.assertAlmostEqual(transform.theta, 0.0, delta=0.5)

    def test_blank_frame_raises(self):
        blank = _blank_frame()
        with self.assertRaises(RegistrationError):
            FeatureRANSACEuclidean().estimate(blank, blank)

    def test_too_few_matches_raises(self):
        # a structured reference against a featureless blank frame: ORB finds
        # (at best) a handful of keypoints in one image and none in the
        # other, so far fewer than 3 correspondences can survive.
        reference = _structured_frame(seed=1)
        blank = _blank_frame()
        with self.assertRaises(RegistrationError):
            FeatureRANSACEuclidean().estimate(reference, blank)

    def test_multi_channel_input_matches_grayscale_accuracy(self):
        reference_gray = _structured_frame(seed=1)
        dx_true, dy_true, theta_true = 5.0, -3.0, 4.0
        frame_gray = _warp(reference_gray, dx_true, dy_true, theta_true)

        for channels in (1, 3, 4):
            with self.subTest(channels=channels):
                reference = np.stack([reference_gray] * channels, axis=-1)
                frame = np.stack([frame_gray] * channels, axis=-1)
                transform = FeatureRANSACEuclidean().estimate(reference, frame)
                self.assertAlmostEqual(transform.dx, dx_true, delta=0.5)
                self.assertAlmostEqual(transform.dy, dy_true, delta=0.5)
                self.assertAlmostEqual(transform.theta, theta_true, delta=0.5)

    def test_estimate_affine_partial2d_returns_no_model_raises(self):
        """``cv2.estimateAffinePartial2D`` returning no model (``None``) is
        handled explicitly rather than propagating a ``TypeError``/``None``
        downstream.

        This branch is defensive: with the existing ratio-test and
        ``min_inliers`` guards already in place, realistic synthetic input
        that reaches ``estimateAffinePartial2D`` with >= ``min_inliers``
        correspondences essentially always yields *some* model. Rather than
        force a contrived degenerate point configuration, this test
        monkeypatches ``cv2.estimateAffinePartial2D`` directly -- mirroring
        the same pattern already used below for
        ``GradientECC``'s non-convergence branch -- to verify the
        ``matrix is None`` branch itself behaves correctly when it does
        fire.
        """
        reference = _structured_frame(seed=1)
        frame = _warp(reference, 5.0, -3.0, 4.0)

        def _no_model(*args, **kwargs):
            return None, None

        original = cv2.estimateAffinePartial2D
        cv2.estimateAffinePartial2D = _no_model
        try:
            with self.assertRaises(RegistrationError):
                FeatureRANSACEuclidean().estimate(reference, frame)
        finally:
            cv2.estimateAffinePartial2D = original


class TestGradientECC(unittest.TestCase):
    """Rigid recovery via Enhanced Correlation Coefficient (ECC) maximization."""

    def test_rigid_happy_path(self):
        reference = _structured_frame(seed=2)
        dx_true, dy_true, theta_true = 3.0, 2.0, 2.5
        frame = _warp(reference, dx_true, dy_true, theta_true)

        transform = GradientECC().estimate(reference, frame)

        self.assertAlmostEqual(transform.dx, dx_true, delta=0.5)
        self.assertAlmostEqual(transform.dy, dy_true, delta=0.5)
        self.assertAlmostEqual(transform.theta, theta_true, delta=0.5)

    def test_zero_motion(self):
        reference = _structured_frame(seed=2)
        transform = GradientECC().estimate(reference, reference)
        self.assertAlmostEqual(transform.dx, 0.0, delta=0.5)
        self.assertAlmostEqual(transform.dy, 0.0, delta=0.5)
        self.assertAlmostEqual(transform.theta, 0.0, delta=0.5)

    def test_blank_frame_raises(self):
        blank = _blank_frame()
        with self.assertRaises(RegistrationError):
            GradientECC().estimate(blank, blank)

    def test_non_convergence_reraised_as_registration_error(self):
        """``cv2.error`` from ``findTransformECC`` is caught and re-raised."""
        reference = _structured_frame(seed=2)
        frame = _warp(reference, 3.0, 2.0, 2.5)

        def _raise_cv2_error(*args, **kwargs):
            raise cv2.error("synthetic non-convergence for testing")

        original = cv2.findTransformECC
        cv2.findTransformECC = _raise_cv2_error
        try:
            with self.assertRaises(RegistrationError):
                GradientECC().estimate(reference, frame)
        finally:
            cv2.findTransformECC = original

    def test_nan_frame_raises(self):
        """Non-finite input must not silently bypass the ``min_gradient_std``
        guard (``std()`` of a NaN-containing array is itself NaN, which
        compares False against any threshold)."""
        reference = _structured_frame(seed=2)
        frame = _warp(reference, 3.0, 2.0, 2.5).astype(np.float32)
        frame[50, 50] = np.nan
        with self.assertRaises(RegistrationError):
            GradientECC().estimate(reference, frame)


class TestShapeMismatchGuard(unittest.TestCase):
    """Reference/frame shape mismatch must raise a clear ``RegistrationError``
    -- one shared test for the 4 methods that lacked this guard
    (:class:`~acia.registration.HoughLineRigidFit` already had one)."""

    def test_shape_mismatch_raises(self):
        reference = _textured_frame(seed=0, size=SIZE)
        wrong_shape_frame = _textured_frame(seed=0, size=SIZE + 10)

        methods = [
            PhaseCorrelationHighpass(),
            MaskedTemplateCorrelation(_default_mask_rect(), search_margin=30),
            FeatureRANSACEuclidean(),
            GradientECC(),
        ]
        for method in methods:
            with (
                self.subTest(method=type(method).__name__),
                self.assertRaises(RegistrationError),
            ):
                method.estimate(reference, wrong_shape_frame)


class TestParabolicRefine(unittest.TestCase):
    """Direct unit tests of the shared subpixel-peak-offset helper."""

    def test_clamped_to_valid_range(self):
        # A strongly asymmetric/degenerate 3-sample peak: the raw parabolic
        # formula (0.5 * (c_minus - c_plus) / (c_minus - 2*c_zero + c_plus))
        # here evaluates to 0.5*(0-1)/(0-1.8+1) = 0.625, well outside
        # [-0.5, 0.5]; the clamp must bring it back in range.
        clamped = _parabolic_refine(0.0, 0.9, 1.0)
        self.assertLessEqual(clamped, 0.5)
        self.assertGreaterEqual(clamped, -0.5)
        self.assertAlmostEqual(clamped, 0.5)

    def test_non_finite_samples_return_zero(self):
        self.assertEqual(_parabolic_refine(float("nan"), 1.0, 0.5), 0.0)
        self.assertEqual(_parabolic_refine(0.5, float("inf"), 1.0), 0.0)
        self.assertEqual(_parabolic_refine(0.5, 1.0, float("-inf")), 0.0)


class TestApplyCorrection(unittest.TestCase):
    """Direct unit tests for the shared inverse-warp helper -- the single
    place :class:`~acia.base.RegisteredSequenceSource`, the verify view, and
    batch-apply all correct a frame."""

    def test_undoes_a_known_drift(self):
        reference = _textured_frame(seed=0, size=SIZE)
        drifted = _warp(reference, dx=4.0, dy=-3.0, theta=2.0)
        transform = FrameTransform(dx=4.0, dy=-3.0, theta=2.0)

        corrected = apply_correction(drifted, transform)

        self.assertLess(
            np.abs(corrected.astype(np.float32) - reference.astype(np.float32)).mean(),
            np.abs(drifted.astype(np.float32) - reference.astype(np.float32)).mean(),
        )

    def test_grayscale_2d_shape_preserved(self):
        frame = _textured_frame(seed=0, size=SIZE)
        self.assertEqual(frame.ndim, 2)

        corrected = apply_correction(frame, FrameTransform(dx=1.0, dy=1.0, theta=0.0))

        self.assertEqual(corrected.shape, frame.shape)

    def test_single_channel_axis_not_collapsed(self):
        # Regression test: cv2.warpAffine silently drops a trailing (H, W, 1)
        # channel axis, turning it into a 2D (H, W) array -- the same bug
        # RotatedCropSequenceSource._warp already guards against. A
        # RegisteredSequenceSource wrapping single-channel microscopy data
        # must not silently violate this codebase's (H, W, C) convention.
        frame = _textured_frame(seed=0, size=SIZE)[..., None]
        self.assertEqual(frame.shape, (SIZE, SIZE, 1))

        corrected = apply_correction(frame, FrameTransform(dx=1.0, dy=1.0, theta=0.0))

        self.assertEqual(corrected.shape, (SIZE, SIZE, 1))

    def test_multi_channel_shape_preserved(self):
        gray = _textured_frame(seed=0, size=SIZE)
        frame = np.stack([gray, gray, gray], axis=-1)

        corrected = apply_correction(frame, FrameTransform(dx=2.0, dy=0.0, theta=5.0))

        self.assertEqual(corrected.shape, frame.shape)

    def test_more_than_four_channels_shape_preserved(self):
        gray = _textured_frame(seed=0, size=SIZE)
        frame = np.stack([gray] * 6, axis=-1)

        corrected = apply_correction(frame, FrameTransform(dx=1.0, dy=-1.0, theta=3.0))

        self.assertEqual(corrected.shape, frame.shape)

    def test_zero_transform_is_near_identity(self):
        frame = _textured_frame(seed=0, size=SIZE)

        corrected = apply_correction(frame, FrameTransform(dx=0.0, dy=0.0, theta=0.0))

        np.testing.assert_allclose(
            corrected.astype(np.float32), frame.astype(np.float32), atol=1.0
        )


class TestRunComparisonProgress(unittest.TestCase):
    """``run_comparison``'s optional ``on_progress`` callback -- shared by the
    comparison notebooks (unaffected, defaults to ``None``) and
    ``RegistrationDashboard._run_verify``, which wires a callback to report
    per-frame progress to the widget."""

    def test_on_progress_called_once_per_frame_in_order(self):
        reference = _textured_frame(seed=0, size=SIZE)
        frames = {
            0: _warp(reference, dx=1.0, dy=0.0, theta=0.0),
            1: _warp(reference, dx=2.0, dy=0.0, theta=0.0),
            2: _warp(reference, dx=3.0, dy=0.0, theta=0.0),
        }
        calls = []
        run_comparison(
            {"GradientECC": GradientECC()},
            reference,
            lambda t: frames[t],
            [0, 1, 2],
            on_progress=lambda i, total: calls.append((i, total)),
        )
        self.assertEqual(calls, [(0, 3), (1, 3), (2, 3)])

    def test_on_progress_fires_after_every_method_for_that_frame(self):
        """The callback for frame i fires only once *all* methods have been
        run against ``frame_indices[i]`` -- not interleaved per-method."""
        reference = _textured_frame(seed=0, size=SIZE)
        frame = _warp(reference, dx=1.0, dy=0.0, theta=0.0)

        class _CountingMethod(GradientECC):
            def __init__(self):
                super().__init__()
                self.count = 0

            def estimate(self, reference, frame):
                self.count += 1
                return super().estimate(reference, frame)

        m1, m2 = _CountingMethod(), _CountingMethod()
        counts_at_progress = []

        def on_progress(i, total):
            counts_at_progress.append((m1.count, m2.count))

        run_comparison(
            {"a": m1, "b": m2},
            reference,
            lambda t: frame,
            [0],
            on_progress=on_progress,
        )
        self.assertEqual(counts_at_progress, [(1, 1)])

    def test_on_progress_defaults_to_none_and_is_backward_compatible(self):
        """Existing callers (both notebooks) pass no ``on_progress`` at all --
        must behave exactly as before."""
        reference = _textured_frame(seed=0, size=SIZE)
        frame = _warp(reference, dx=1.0, dy=0.0, theta=0.0)
        results = run_comparison(
            {"GradientECC": GradientECC()}, reference, lambda t: frame, [0]
        )
        self.assertEqual(len(results["GradientECC"]), 1)

    def test_on_progress_failure_does_not_abort_the_comparison(self):
        """A broken ``on_progress`` callback (e.g. a dashboard's ``send`` on a
        closed comm channel) must be isolated the same way a per-method
        failure is -- it must not abort the rest of the comparison, and every
        frame must still get a result."""
        reference = _textured_frame(seed=0, size=SIZE)
        frames = {
            0: _warp(reference, dx=1.0, dy=0.0, theta=0.0),
            1: _warp(reference, dx=2.0, dy=0.0, theta=0.0),
            2: _warp(reference, dx=3.0, dy=0.0, theta=0.0),
        }
        calls = []

        def on_progress(i, total):
            calls.append((i, total))
            raise RuntimeError("simulated broken callback")

        results = run_comparison(
            {"GradientECC": GradientECC()},
            reference,
            lambda t: frames[t],
            [0, 1, 2],
            on_progress=on_progress,
        )
        # the callback still fired for every frame (it just failed each time)
        self.assertEqual(calls, [(0, 3), (1, 3), (2, 3)])
        # every frame still produced a result entry -- not aborted partway
        # through (a ragged/short list would mean the callback's exception
        # propagated out of the loop).
        self.assertEqual(len(results["GradientECC"]), 3)


if __name__ == "__main__":
    unittest.main()
