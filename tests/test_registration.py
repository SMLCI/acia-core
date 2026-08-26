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
import warnings
from itertools import pairwise

import cv2
import numpy as np

from acia.base import ArrayImage, ImageSequenceSource, RotatedCropSpec
from acia.registration import (
    FeatureRANSACEuclidean,
    FrameTransform,
    GradientECC,
    HoughLineRigidFit,
    LowConfidenceWarning,
    MaskedTemplateCorrelation,
    PhaseCorrelationHighpass,
    ReanchoringReference,
    RegistrationError,
    _build_gray_pyramid,
    _parabolic_refine,
    _rect_mask,
    apply_correction,
    apply_correction_to_spec,
    compose,
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


DEVICE_SIZE = 512

# The two "growth channels": static rectangles whose *interiors* fill with
# cells over time, while their borders (the channel walls) stay put.
DEVICE_ROIS = ((120, 100, 90, 260), (300, 100, 90, 260))


def _device_frame(seed: int = 11, size: int = DEVICE_SIZE) -> np.ndarray:
    """Static microfluidic-like geometry: noise + irregularly spaced walls.

    The wall spacing is deliberately irregular. An evenly-spaced grid aliases
    under coarse-to-fine downsampling -- :class:`GradientECC`'s documented
    wrong-by-one-period limitation -- which would make these tests flaky for a
    reason that has nothing to do with what they are checking.
    """
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 25, (size, size)).astype(np.float32)
    for x in (40, 205, 250, 470):
        cv2.line(frame, (x, 0), (x, size), 220, 2)
    for y in (30, 480):
        cv2.line(frame, (0, y), (size, y), 220, 2)
    for x, y, w, h in DEVICE_ROIS:
        cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 3)
    return cv2.GaussianBlur(frame, (0, 0), 1.0)


def _device_roi_specs() -> list[RotatedCropSpec]:
    """:data:`DEVICE_ROIS` as the crop specs a caller would already have."""
    return [
        RotatedCropSpec(
            center=(x + w / 2.0, y + h / 2.0), size=(int(w), int(h)), angle=0.0
        )
        for x, y, w, h in DEVICE_ROIS
    ]


def _growing_colony_series(
    n_frames: int = 40, drift_per_frame: tuple[float, float] = (0.06, -0.04)
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Frames with known linear drift *and* monotonically growing content.

    Reproduces the field failure this module's ``exclude_rects`` support
    exists for: cells accumulate inside :data:`DEVICE_ROIS` as time advances,
    so a fixed-reference correlation coefficient decays with elapsed content
    change even though the drift stays perfectly recoverable.

    Returns:
        tuple: ``(frames, ground_truth)`` where ``ground_truth[t]`` is the
            ``(dx, dy)`` frame ``t`` was displaced by.
    """
    base = _device_frame()
    frames, truth = [], []
    for t in range(n_frames):
        frame = base.copy()
        # Same RNG stream every frame, so frame t's cells are a superset of
        # frame t-1's -- growth, not a reshuffle.
        rng = np.random.default_rng(100)
        for x, y, w, h in DEVICE_ROIS:
            for _ in range(int(t * 1.2)):
                cx = int(rng.integers(x + 6, x + w - 6))
                cy = int(rng.integers(y + 6, y + h - 6))
                cv2.circle(frame, (cx, cy), int(rng.integers(3, 6)), 200, -1)
        frame = cv2.GaussianBlur(frame, (0, 0), 1.0)
        dx, dy = drift_per_frame[0] * t, drift_per_frame[1] * t
        frames.append(
            cv2.warpAffine(
                frame,
                np.float32([[1, 0, dx], [0, 1, dy]]),
                (DEVICE_SIZE, DEVICE_SIZE),
                borderMode=cv2.BORDER_REFLECT,
            )
        )
        truth.append((dx, dy))
    return frames, truth


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
        method = MaskedTemplateCorrelation(
            _default_mask_rect(), search_margin=8, on_low_confidence="reject"
        )
        with self.assertRaises(RegistrationError):
            method.estimate(reference, frame)

    def test_shift_beyond_search_margin_is_kept_but_flagged_by_default(self):
        """The default policy keeps this fit -- it is exactly the case
        ``confidence`` exists to flag, since the returned shift is wrong."""
        reference = _textured_frame(seed=3)
        frame = _warp(reference, 50.0, 0.0, 0.0)
        method = MaskedTemplateCorrelation(_default_mask_rect(), search_margin=8)
        with self.assertWarns(LowConfidenceWarning):
            transform = method.estimate(reference, frame)
        self.assertLess(transform.confidence, 0.5)

    def test_low_score_rejected(self):
        """Genuinely dissimilar content, found within the search window,
        must be rejected on score -- distinct from the search-margin-exceeded
        (edge-of-window) rejection above."""
        reference = _textured_frame(seed=3)
        dissimilar_frame = _textured_frame(seed=99)  # unrelated content
        method = MaskedTemplateCorrelation(
            _default_mask_rect(),
            search_margin=30,
            min_score=0.5,
            on_low_confidence="reject",
        )
        with self.assertRaisesRegex(RegistrationError, "min_score"):
            method.estimate(reference, dissimilar_frame)

    def test_low_score_kept_and_warned_by_default(self):
        reference = _textured_frame(seed=3)
        dissimilar_frame = _textured_frame(seed=99)
        method = MaskedTemplateCorrelation(
            _default_mask_rect(), search_margin=30, min_score=0.5
        )
        with self.assertWarnsRegex(LowConfidenceWarning, "min_score"):
            transform = method.estimate(reference, dissimilar_frame)
        self.assertLess(transform.confidence, 0.5)

    def test_kept_estimate_equals_the_ungated_one(self):
        """Gate-motion regression: the refinement now runs *before* the gate,
        so what "keep" returns must be byte-identical to what the same method
        with the gate disabled returns."""
        reference = _textured_frame(seed=3)
        dissimilar_frame = _textured_frame(seed=99)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LowConfidenceWarning)
            kept = MaskedTemplateCorrelation(
                _default_mask_rect(), search_margin=30, min_score=0.5
            ).estimate(reference, dissimilar_frame)
        ungated = MaskedTemplateCorrelation(
            _default_mask_rect(), search_margin=30, min_score=0.0
        ).estimate(reference, dissimilar_frame)
        self.assertEqual(kept, ungated)

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

    def test_too_few_matches_raises_whatever_the_policy(self):
        """The pre-RANSAC match count is a hard floor, not a confidence gate:
        with too few correspondences no model is ever built, so there is no
        estimate for "keep" to keep."""
        reference = _structured_frame(seed=1)
        blank = _blank_frame()
        with self.assertRaises(RegistrationError):
            FeatureRANSACEuclidean(on_low_confidence="keep").estimate(reference, blank)

    def test_low_inlier_count_is_kept_and_warned_by_default(self):
        reference = _structured_frame(seed=1)
        dx_true, dy_true, theta_true = 5.0, -3.0, 4.0
        frame = _warp(reference, dx_true, dy_true, theta_true)
        # 80 sits in the window between this fixture's RANSAC inlier count
        # (~67) and its ratio-test match count (~101): high enough to miss the
        # confidence gate, low enough to clear the hard pre-RANSAC floor that
        # shares the same setting. Nothing about the fit itself changes.
        method = FeatureRANSACEuclidean(min_inliers=80)
        with self.assertWarnsRegex(LowConfidenceWarning, "inlier"):
            transform = method.estimate(reference, frame)
        self.assertAlmostEqual(transform.dx, dx_true, delta=0.5)
        self.assertEqual(transform, FeatureRANSACEuclidean().estimate(reference, frame))

    def test_low_inlier_count_rejected_under_the_reject_policy(self):
        reference = _structured_frame(seed=1)
        frame = _warp(reference, 5.0, -3.0, 4.0)
        method = FeatureRANSACEuclidean(min_inliers=80, on_low_confidence="reject")
        with self.assertRaisesRegex(RegistrationError, "inlier"):
            method.estimate(reference, frame)

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

    def test_rigid_happy_path_multi_level_pyramid(self):
        """Same happy-path recovery, but on a large enough frame (800x800)
        that the default ``max_pyramid_levels``/``min_pyramid_size`` produce
        multiple coarse-to-fine levels (800 -> 400 -> 200, stopped short of
        a 4th by ``min_pyramid_size=128``) -- exercising the pyramid loop
        end-to-end, not just its single-level degenerate case."""
        reference = _structured_frame(seed=2, size=800)
        dx_true, dy_true, theta_true = 3.0, 2.0, 2.5
        frame = _warp(reference, dx_true, dy_true, theta_true)

        transform = GradientECC().estimate(reference, frame)

        self.assertAlmostEqual(transform.dx, dx_true, delta=0.5)
        self.assertAlmostEqual(transform.dy, dy_true, delta=0.5)
        self.assertAlmostEqual(transform.theta, theta_true, delta=0.5)

    def test_translation_only_recovers_translation_forces_zero_theta(self):
        """``translation_only=True`` fits ``cv2.MOTION_TRANSLATION`` -- no
        rotation parameter exists to fit, so ``theta`` must come back exactly
        ``0.0`` (not just close), matching the module's convention for
        translation-only methods (``PhaseCorrelationHighpass`` etc.)."""
        reference = _structured_frame(seed=2)
        dx_true, dy_true = 3.0, 2.0
        frame = _warp(reference, dx_true, dy_true, 0.0)

        transform = GradientECC(translation_only=True).estimate(reference, frame)

        self.assertAlmostEqual(transform.dx, dx_true, delta=0.5)
        self.assertAlmostEqual(transform.dy, dy_true, delta=0.5)
        self.assertEqual(transform.theta, 0.0)

    def test_early_stop_delta_px_stops_before_full_resolution(self):
        """A generous ``early_stop_delta_px`` should stop the 800x800
        fixture's 3-level pyramid (800 -> 400 -> 200) after the coarsest
        level -- fewer ``cv2.findTransformECC`` calls than the full pyramid,
        while the returned transform still meets the same tolerance."""
        reference = _structured_frame(seed=2, size=800)
        dx_true, dy_true, theta_true = 3.0, 2.0, 2.5
        frame = _warp(reference, dx_true, dy_true, theta_true)

        call_count = 0
        original = cv2.findTransformECC

        def _counting_ecc(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        cv2.findTransformECC = _counting_ecc
        try:
            full_call_count = call_count = 0
            GradientECC().estimate(reference, frame)
            full_call_count = call_count

            call_count = 0
            transform = GradientECC(early_stop_delta_px=5.0).estimate(reference, frame)
            early_stop_call_count = call_count
        finally:
            cv2.findTransformECC = original

        self.assertLess(early_stop_call_count, full_call_count)
        self.assertAlmostEqual(transform.dx, dx_true, delta=0.5)
        self.assertAlmostEqual(transform.dy, dy_true, delta=0.5)
        self.assertAlmostEqual(transform.theta, theta_true, delta=0.5)

    def test_early_stop_delta_px_none_matches_prior_behavior(self):
        """``early_stop_delta_px=None`` (the default) must run every pyramid
        level, identically to a `GradientECC()` with no early-stop kwarg --
        i.e. this feature is opt-in, not a behavior change by default."""
        reference = _structured_frame(seed=2, size=800)
        dx_true, dy_true, theta_true = 3.0, 2.0, 2.5
        frame = _warp(reference, dx_true, dy_true, theta_true)

        transform = GradientECC(early_stop_delta_px=None).estimate(reference, frame)

        self.assertAlmostEqual(transform.dx, dx_true, delta=0.5)
        self.assertAlmostEqual(transform.dy, dy_true, delta=0.5)
        self.assertAlmostEqual(transform.theta, theta_true, delta=0.5)


class TestBuildGrayPyramid(unittest.TestCase):
    """Coarse-to-fine level construction shared by :class:`GradientECC`."""

    def test_finest_level_is_input_array(self):
        gray = np.zeros((200, 200), dtype=np.float32)
        levels = _build_gray_pyramid(gray, max_levels=4, min_size=128)
        self.assertIs(levels[0], gray)

    def test_stops_on_min_size_before_reaching_max_levels(self):
        """The default ``min_pyramid_size=128`` used by ``GradientECC`` must
        keep the existing 200x200 test fixtures single-level (a candidate
        100x100 next level is rejected as < 128) -- i.e. no existing
        ``GradientECC`` test's behavior changes because of the pyramid."""
        gray = np.zeros((SIZE, SIZE), dtype=np.float32)
        levels = _build_gray_pyramid(gray, max_levels=4, min_size=128)
        self.assertEqual(len(levels), 1)

    def test_stops_on_max_levels_before_reaching_min_size(self):
        gray = np.zeros((4096, 4096), dtype=np.float32)
        levels = _build_gray_pyramid(gray, max_levels=3, min_size=16)
        self.assertEqual(len(levels), 3)
        for level, expected_size in zip(levels, (4096, 2048, 1024), strict=True):
            self.assertEqual(level.shape, (expected_size, expected_size))

    def test_handles_non_square_odd_dimensions(self):
        gray = np.zeros((201, 151), dtype=np.float32)
        levels = _build_gray_pyramid(gray, max_levels=4, min_size=16)
        self.assertEqual(levels[0].shape, (201, 151))
        for prev, cur in pairwise(levels):
            self.assertEqual(
                cur.shape, ((prev.shape[0] + 1) // 2, (prev.shape[1] + 1) // 2)
            )


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


class _GraySource(ImageSequenceSource):
    """Minimal (T, H, W) grayscale source (mirrors ``tests/test_base.py``)."""

    def __init__(self, stack: np.ndarray):
        self.stack = stack

    def get_frame(self, frame: int) -> ArrayImage:
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


class TestApplyCorrectionToSpec(unittest.TestCase):
    """Moving crop geometry across the same warp `apply_correction` applies
    to pixels -- so an ROI drawn on a late frame still crops the region it was
    drawn around once the sequence is registered onto frame 0."""

    SPEC = RotatedCropSpec(center=(50.0, 40.0), size=(30, 20), angle=12.0)

    def test_none_transform_returns_the_spec_itself(self):
        self.assertIs(
            apply_correction_to_spec(self.SPEC, None, shape=(SIZE, SIZE)), self.SPEC
        )

    def test_zero_transform_is_exactly_a_no_op(self):
        mapped = apply_correction_to_spec(
            self.SPEC, FrameTransform(0.0, 0.0, 0.0), shape=(SIZE, SIZE)
        )
        self.assertEqual(mapped, self.SPEC)

    def test_translation_moves_the_center_the_other_way(self):
        """A frame that drifted by (+7, -3) has its content sitting 7 px right
        of where the reference had it, so a box drawn around that content maps
        back by the same amount in the opposite direction."""
        mapped = apply_correction_to_spec(
            self.SPEC, FrameTransform(dx=7.0, dy=-3.0, theta=0.0), shape=(SIZE, SIZE)
        )
        self.assertAlmostEqual(mapped.center[0], 43.0, places=5)
        self.assertAlmostEqual(mapped.center[1], 43.0, places=5)
        self.assertAlmostEqual(mapped.angle, 12.0, places=5)

    def test_rotation_adds_to_the_angle(self):
        mapped = apply_correction_to_spec(
            self.SPEC, FrameTransform(0.0, 0.0, 5.0), shape=(SIZE, SIZE)
        )
        self.assertAlmostEqual(mapped.angle, 17.0, places=5)

    def test_size_is_preserved_as_positive_ints(self):
        """RotatedCropSpec rejects a non-int size, so the tuple must be passed
        through rather than recomputed."""
        mapped = apply_correction_to_spec(
            self.SPEC, FrameTransform(3.3, -1.7, 2.5), shape=(SIZE, SIZE)
        )
        self.assertEqual(mapped.size, (30, 20))
        for value in mapped.size:
            self.assertIsInstance(value, int)

    def _registered_pair(self):
        """A 2-frame source: frame 0 the reference, frame 1 drifted by a known
        transform. Returns (source, transform, spec drawn on frame 1)."""
        reference = _structured_frame(seed=1)
        transform = FrameTransform(dx=6.0, dy=-4.0, theta=3.0)
        frame_n = _warp(reference, dx=6.0, dy=-4.0, theta=3.0)
        source = _GraySource(np.stack([reference, frame_n]))
        # Well inside the frame, so no zero-border creeps into either crop.
        spec = RotatedCropSpec(center=(100.0, 96.0), size=(40, 24), angle=15.0)
        return source, transform, spec

    @staticmethod
    def _mean_abs(a, b):
        return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())

    def test_registered_crop_matches_the_crop_from_the_anchor_frame(self):
        source, transform, spec = self._registered_pair()
        want = source.crop_rotated(spec).get_frame(1).raw

        mapped = apply_correction_to_spec(
            spec, transform, shape=(source.size_h, source.size_w)
        )
        registered = source.register({0: FrameTransform(0.0, 0.0, 0.0), 1: transform})
        got = registered.crop_rotated(mapped).get_frame(1).raw

        self.assertEqual(got.shape, want.shape)
        # Interpolation runs twice on the `got` side (correct, then crop), so
        # compare on mean absolute error rather than exactly.
        self.assertLess(self._mean_abs(got, want), 12.0)

    def test_the_unmapped_spec_is_measurably_worse(self):
        """The negative control: without it the test above would still pass if
        `apply_correction_to_spec` were `return spec`."""
        source, transform, spec = self._registered_pair()
        want = source.crop_rotated(spec).get_frame(1).raw
        registered = source.register({0: FrameTransform(0.0, 0.0, 0.0), 1: transform})

        mapped = apply_correction_to_spec(
            spec, transform, shape=(source.size_h, source.size_w)
        )
        mapped_error = self._mean_abs(
            registered.crop_rotated(mapped).get_frame(1).raw, want
        )
        unmapped_error = self._mean_abs(
            registered.crop_rotated(spec).get_frame(1).raw, want
        )

        self.assertGreater(unmapped_error, 3.0 * mapped_error)


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


class TestFrameTransformConfidence(unittest.TestCase):
    """The optional per-transform goodness-of-fit score."""

    def test_defaults_to_none_and_is_omitted_from_to_dict(self):
        transform = FrameTransform(dx=1.0, dy=2.0, theta=3.0)
        self.assertIsNone(transform.confidence)
        # Serialization of a score-less transform is unchanged, so manifests
        # written by earlier versions stay byte-identical.
        self.assertEqual(transform.to_dict(), {"dx": 1.0, "dy": 2.0, "theta": 3.0})

    def test_roundtrips_when_set(self):
        transform = FrameTransform(dx=1.0, dy=2.0, theta=3.0, confidence=0.87)
        self.assertEqual(transform.to_dict()["confidence"], 0.87)
        self.assertEqual(FrameTransform.from_dict(transform.to_dict()), transform)

    def test_from_dict_without_confidence_yields_none(self):
        transform = FrameTransform.from_dict({"dx": 1.0, "dy": 2.0, "theta": 0.0})
        self.assertIsNone(transform.confidence)

    def test_gradient_ecc_reports_a_confidence(self):
        reference = _structured_frame()
        moved = _warp(reference, dx=3.0, dy=-2.0, theta=0.0)
        transform = GradientECC().estimate(reference, moved)
        self.assertIsNotNone(transform.confidence)
        self.assertGreater(transform.confidence, 0.95)

    def test_phase_correlation_reports_no_confidence(self):
        """A method with no honest scalar to report leaves it None."""
        reference = _textured_frame()
        moved = _warp(reference, dx=3.0, dy=-2.0, theta=0.0)
        transform = PhaseCorrelationHighpass().estimate(reference, moved)
        self.assertIsNone(transform.confidence)


class TestCompose(unittest.TestCase):
    """:func:`compose` against the matrix product it stands in for."""

    @staticmethod
    def _matrix(transform: FrameTransform, center: tuple[float, float]) -> np.ndarray:
        matrix = cv2.getRotationMatrix2D(center, transform.theta, 1.0)
        matrix[0, 2] += transform.dx
        matrix[1, 2] += transform.dy
        return np.vstack([matrix, [0.0, 0.0, 1.0]])

    def _assert_close(self, actual: FrameTransform, expected: FrameTransform):
        self.assertAlmostEqual(actual.dx, expected.dx, places=9)
        self.assertAlmostEqual(actual.dy, expected.dy, places=9)
        self.assertAlmostEqual(actual.theta, expected.theta, places=9)

    def test_matches_matrix_product_for_any_pivot(self):
        first = FrameTransform(dx=3.0, dy=-2.0, theta=4.0)
        second = FrameTransform(dx=-1.5, dy=0.5, theta=-7.0)
        composed = compose(first, second)
        # Center-independence is the property that lets compose() take no
        # frame shape: assert it against three different pivots.
        for center in ((0.0, 0.0), (137.0, 42.0), (-11.0, 300.0)):
            product = self._matrix(second, center) @ self._matrix(first, center)
            np.testing.assert_allclose(
                self._matrix(composed, center), product, atol=1e-9
            )

    def test_identity_is_a_left_and_right_unit(self):
        identity = FrameTransform(dx=0.0, dy=0.0, theta=0.0)
        transform = FrameTransform(dx=3.0, dy=-2.0, theta=4.0)
        self._assert_close(compose(identity, transform), transform)
        self._assert_close(compose(transform, identity), transform)

    def test_is_associative(self):
        a = FrameTransform(dx=3.0, dy=-2.0, theta=4.0)
        b = FrameTransform(dx=-1.5, dy=0.5, theta=-7.0)
        c = FrameTransform(dx=0.7, dy=9.0, theta=2.5)
        self._assert_close(compose(compose(a, b), c), compose(a, compose(b, c)))

    def test_confidence_is_the_weakest_link(self):
        a = FrameTransform(dx=0.0, dy=0.0, theta=0.0, confidence=0.9)
        b = FrameTransform(dx=0.0, dy=0.0, theta=0.0, confidence=0.7)
        self.assertEqual(compose(a, b).confidence, 0.7)

    def test_confidence_is_none_when_either_input_lacks_one(self):
        scored = FrameTransform(dx=0.0, dy=0.0, theta=0.0, confidence=0.9)
        unscored = FrameTransform(dx=0.0, dy=0.0, theta=0.0)
        self.assertIsNone(compose(scored, unscored).confidence)
        self.assertIsNone(compose(unscored, scored).confidence)


class TestRectMask(unittest.TestCase):
    """The ECC validity mask built from crop specs."""

    def test_no_rects_yields_no_mask(self):
        self.assertIsNone(_rect_mask((64, 64), []))

    def test_excludes_the_rect_interior_and_keeps_everything_else(self):
        spec = RotatedCropSpec(center=(50.0, 50.0), size=(20, 20), angle=0.0)
        mask = _rect_mask((100, 100), [spec])
        self.assertEqual(mask[50, 50], 0)  # inside
        self.assertEqual(mask[10, 10], 255)  # outside

    def test_shrink_keeps_a_band_just_inside_the_border(self):
        """The border band is what makes the mask worth using -- a channel
        wall's sharp edge is exactly the static feature to register on."""
        spec = RotatedCropSpec(center=(50.0, 50.0), size=(40, 40), angle=0.0)
        tight = _rect_mask((100, 100), [spec], shrink_px=0.0)
        shrunk = _rect_mask((100, 100), [spec], shrink_px=5.0)
        # A pixel 3 px inside the border: excluded without shrink, kept with.
        self.assertEqual(tight[50, 33], 0)
        self.assertEqual(shrunk[50, 33], 255)
        self.assertEqual(shrunk[50, 50], 0)  # the interior is still excluded

    def test_shrinking_a_rect_away_entirely_excludes_nothing(self):
        spec = RotatedCropSpec(center=(50.0, 50.0), size=(10, 10), angle=0.0)
        self.assertIsNone(_rect_mask((100, 100), [spec], shrink_px=20.0))

    def test_scale_maps_the_rect_onto_a_pyramid_level(self):
        spec = RotatedCropSpec(center=(100.0, 100.0), size=(40, 40), angle=0.0)
        half = _rect_mask((100, 100), [spec], scale=0.5)
        self.assertEqual(half[50, 50], 0)  # the rect center, halved
        self.assertEqual(half[95, 95], 255)

    def test_rotated_rect_is_masked_where_the_crop_would_be_taken(self):
        """The mask must agree with RotatedCropSequenceSource's own geometry,
        or an excluded region would drift away from the ROI it stands for."""
        spec = RotatedCropSpec(center=(60.0, 50.0), size=(40, 16), angle=30.0)
        mask = _rect_mask((120, 120), [spec])

        # Reproduce the crop the same way acia.base does, and check the mask
        # is zero exactly over the pixels that crop would have covered.
        cx, cy = spec.center
        w, h = spec.size
        matrix = cv2.getRotationMatrix2D((cx, cy), spec.angle, 1.0)
        matrix[0, 2] += w / 2 - cx
        matrix[1, 2] += h / 2 - cy
        cropped_mask = cv2.warpAffine(
            mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=255
        )
        # Allow a 1px rounding skin around the polygon edge.
        self.assertLess(cropped_mask[2:-2, 2:-2].max(), 1)

    def test_covering_the_whole_frame_raises(self):
        spec = RotatedCropSpec(center=(50.0, 50.0), size=(400, 400), angle=0.0)
        with self.assertRaises(RegistrationError) as ctx:
            _rect_mask((100, 100), [spec])
        self.assertIn("entire", str(ctx.exception))


class TestSceneDriftFalseFailure(unittest.TestCase):
    """The reported bug: content change, not misalignment, trips the gate.

    Against a fixed reference, ECC's correlation coefficient measures scene
    similarity. When the imaged content evolves, it decays with elapsed time
    even though the fit stays accurate -- so ``min_confidence`` degrades into
    a "how far into the run are we" gate and rejects good fits in a
    contiguous tail of late frames.
    """

    @classmethod
    def setUpClass(cls):
        cls.frames, cls.truth = _growing_colony_series(n_frames=40)

    def test_late_frames_are_rejected_against_a_fixed_reference(self):
        method = GradientECC(on_low_confidence="reject")  # min_confidence=0.9
        with self.assertRaises(RegistrationError) as ctx:
            method.estimate(self.frames[0], self.frames[-1])
        self.assertIn("correlation coefficient", str(ctx.exception))

    def test_late_frames_are_kept_and_warned_about_by_default(self):
        """The whole point of ``on_low_confidence="keep"`` in one assertion:
        the frame the gate distrusts still gets a transform, that transform is
        in fact correct, and its low ``confidence`` is what says "check me"."""
        method = GradientECC()  # min_confidence=0.9, on_low_confidence="keep"
        with self.assertWarnsRegex(LowConfidenceWarning, "min_confidence"):
            transform = method.estimate(self.frames[0], self.frames[-1])
        dx, dy = self.truth[-1]
        self.assertAlmostEqual(transform.dx, dx, delta=0.5)
        self.assertAlmostEqual(transform.dy, dy, delta=0.5)
        self.assertLess(transform.confidence, 0.9)

    def test_kept_estimate_equals_the_ungated_one(self):
        """Gate-motion regression: decomposition now runs before the gate."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LowConfidenceWarning)
            kept = GradientECC().estimate(self.frames[0], self.frames[-1])
        ungated = GradientECC(min_confidence=0.0).estimate(
            self.frames[0], self.frames[-1]
        )
        self.assertEqual(kept, ungated)

    def test_the_rejected_fits_were_actually_correct(self):
        """With the gate off, the very frames it rejected land on target."""
        method = GradientECC(min_confidence=0.0)
        for t in (20, 30, 39):
            transform = method.estimate(self.frames[0], self.frames[t])
            dx, dy = self.truth[t]
            self.assertAlmostEqual(transform.dx, dx, delta=0.5)
            self.assertAlmostEqual(transform.dy, dy, delta=0.5)
            self.assertAlmostEqual(transform.theta, 0.0, delta=0.5)

    def test_confidence_decays_monotonically_with_elapsed_content_change(self):
        method = GradientECC(min_confidence=0.0)
        scores = [
            method.estimate(self.frames[0], self.frames[t]).confidence
            for t in range(0, 40, 8)
        ]
        for earlier, later in pairwise(scores):
            self.assertLess(later, earlier)


class TestGradientECCExcludeRects(unittest.TestCase):
    """The fix: leave the regions whose content changes out of the objective."""

    @classmethod
    def setUpClass(cls):
        cls.frames, cls.truth = _growing_colony_series(n_frames=40)
        cls.specs = _device_roi_specs()

    def test_every_frame_passes_the_default_gate(self):
        """What the bug above breaks, this restores -- at the default
        min_confidence, with no tolerance loosened anywhere."""
        method = GradientECC(exclude_rects=self.specs, exclude_shrink_px=10.0)
        for t in range(1, 40):
            transform = method.estimate(self.frames[0], self.frames[t])
            dx, dy = self.truth[t]
            self.assertAlmostEqual(transform.dx, dx, delta=0.5)
            self.assertAlmostEqual(transform.dy, dy, delta=0.5)

    def test_confidence_stays_high_across_the_sequence(self):
        method = GradientECC(exclude_rects=self.specs, exclude_shrink_px=10.0)
        for t in (20, 30, 39):
            confidence = method.estimate(self.frames[0], self.frames[t]).confidence
            self.assertGreater(confidence, 0.9)

    def test_no_exclusion_takes_the_unmasked_path(self):
        """Passing no rects must not perturb an existing calibration.

        Checked where the guarantee actually lives -- an empty rect list yields
        no mask, so both callers hand ``cv2.findTransformECC`` the same
        ``inputMask=None`` and it runs the identical code. Comparing the two
        results *bit* for bit looks stronger but is not: ``confidence`` is the
        final correlation coefficient of an iterative optimiser, and OpenCV does
        not promise two invocations on identical input agree in the last bits
        (observed on CI, differing at the 9th decimal while dx/dy/theta matched
        exactly). So the geometry is compared exactly and the score to a
        precision far below anything that could perturb a calibration.
        """
        self.assertIsNone(_rect_mask((64, 64), ()))

        reference = _structured_frame()
        moved = _warp(reference, dx=3.0, dy=-2.0, theta=1.0)
        plain = GradientECC(min_confidence=0.0).estimate(reference, moved)
        empty = GradientECC(min_confidence=0.0, exclude_rects=[]).estimate(
            reference, moved
        )

        self.assertEqual(plain.dx, empty.dx)
        self.assertEqual(plain.dy, empty.dy)
        self.assertEqual(plain.theta, empty.theta)
        self.assertAlmostEqual(plain.confidence, empty.confidence, places=6)

    def test_excluding_everything_raises(self):
        huge = [RotatedCropSpec(center=(256.0, 256.0), size=(2000, 2000), angle=0.0)]
        method = GradientECC(exclude_rects=huge)
        with self.assertRaises(RegistrationError):
            method.estimate(self.frames[0], self.frames[10])


class _RecordingMethod:
    """A stub method capturing which reference each estimate was made against.

    Frames are single-pixel arrays holding their own index, so a recorded
    reference identifies the anchor unambiguously.
    """

    def __init__(self, fail_on=(), fail_against_reference=()):
        self.fail_on = set(fail_on)
        self.fail_against_reference = set(fail_against_reference)
        self.calls: list[tuple[int, int]] = []

    def estimate(self, reference: np.ndarray, frame: np.ndarray) -> FrameTransform:
        ref_id, frame_id = int(reference[0, 0]), int(frame[0, 0])
        self.calls.append((ref_id, frame_id))
        if frame_id in self.fail_on:
            raise RegistrationError("stub: unconditional failure")
        if ref_id in self.fail_against_reference:
            raise RegistrationError("stub: cannot estimate against this reference")
        # A transform that encodes the *step* taken, so composition is checkable.
        return FrameTransform(dx=float(frame_id - ref_id), dy=0.0, theta=0.0)


def _id_frame(index: int) -> np.ndarray:
    return np.full((1, 1), index, dtype=np.int32)


class TestReanchoringReference(unittest.TestCase):
    """The fallback reference policy."""

    def test_rejects_an_unknown_mode(self):
        with self.assertRaises(ValueError):
            ReanchoringReference(_RecordingMethod(), _id_frame(0), mode="sideways")

    def test_fixed_mode_always_uses_the_original_reference(self):
        method = _RecordingMethod()
        tracker = ReanchoringReference(method, _id_frame(0), mode="fixed")
        for t in range(1, 5):
            tracker.estimate(t, _id_frame(t))
        self.assertEqual(method.calls, [(0, 1), (0, 2), (0, 3), (0, 4)])
        self.assertEqual(tracker.reanchor_events, [])
        self.assertEqual(tracker.anchors_used, {})

    def test_fixed_mode_propagates_failures_without_retrying(self):
        method = _RecordingMethod(fail_on={3})
        tracker = ReanchoringReference(method, _id_frame(0), mode="fixed")
        tracker.estimate(1, _id_frame(1))
        with self.assertRaises(RegistrationError):
            tracker.estimate(3, _id_frame(3))
        self.assertEqual(method.calls, [(0, 1), (0, 3)])

    def test_reanchor_only_fires_after_a_failure(self):
        """A frame that succeeds against the reference must take the same path
        it always did -- that is what makes this safe to enable by default."""
        method = _RecordingMethod()
        tracker = ReanchoringReference(method, _id_frame(0), mode="reanchor")
        for t in range(1, 5):
            tracker.estimate(t, _id_frame(t))
        self.assertEqual(method.calls, [(0, 1), (0, 2), (0, 3), (0, 4)])
        self.assertEqual(tracker.reanchor_events, [])

    def test_reanchor_retries_against_the_last_good_frame_and_composes(self):
        method = _RecordingMethod(fail_against_reference={0})
        tracker = ReanchoringReference(method, _id_frame(0), mode="reanchor")
        # Frame 1 has no fallback yet (last-good is still the reference).
        with self.assertRaises(RegistrationError):
            tracker.estimate(1, _id_frame(1))

        method.fail_against_reference = set()
        tracker.estimate(2, _id_frame(2))  # succeeds, becomes last-good
        method.fail_against_reference = {0}

        transform = tracker.estimate(5, _id_frame(5))
        # 0->2 (dx=2) composed with 2->5 (dx=3) == 0->5 (dx=5), absolute.
        self.assertAlmostEqual(transform.dx, 5.0)
        self.assertEqual(method.calls[-2:], [(0, 5), (2, 5)])
        self.assertEqual(tracker.reanchor_events, [(5, 2)])
        self.assertEqual(tracker.anchors_used, {5: 2})

    def test_an_unrecoverable_frame_stays_an_isolated_failure(self):
        """An isolated bad frame must not poison the frames after it."""
        method = _RecordingMethod(fail_on={3})
        tracker = ReanchoringReference(method, _id_frame(0), mode="reanchor")
        tracker.estimate(1, _id_frame(1))
        tracker.estimate(2, _id_frame(2))
        with self.assertRaises(RegistrationError):
            tracker.estimate(3, _id_frame(3))
        # The anchor is untouched, so frame 4 resolves normally.
        self.assertEqual(tracker.estimate(4, _id_frame(4)).dx, 4.0)

    def test_chained_mode_always_steps_from_the_previous_frame(self):
        method = _RecordingMethod()
        tracker = ReanchoringReference(method, _id_frame(0), mode="chained")
        for t in range(1, 5):
            transform = tracker.estimate(t, _id_frame(t))
        self.assertEqual(method.calls, [(0, 1), (1, 2), (2, 3), (3, 4)])
        # Composition still yields an absolute reference->frame transform.
        self.assertAlmostEqual(transform.dx, 4.0)

    def test_seed_restores_the_chain_for_a_resumed_run(self):
        method = _RecordingMethod(fail_against_reference={0})
        tracker = ReanchoringReference(method, _id_frame(0), mode="reanchor")
        tracker.seed(7, _id_frame(7), FrameTransform(dx=7.0, dy=0.0, theta=0.0))
        transform = tracker.estimate(8, _id_frame(8))
        self.assertAlmostEqual(transform.dx, 8.0)
        self.assertEqual(tracker.reanchor_events, [(8, 7)])


class TestReanchoringOnGrowingColony(unittest.TestCase):
    """The policy on the fixture that motivated it, with a real method."""

    @classmethod
    def setUpClass(cls):
        cls.frames, cls.truth = _growing_colony_series(n_frames=40)

    def _run(self, mode, on_low_confidence="reject"):
        # "reject" on purpose: re-anchoring is driven by RegistrationError, so
        # the default "keep" policy is precisely what stops it from firing (see
        # test_keeping_low_confidence_fits_supersedes_reanchoring below). These
        # cases are about the fallback itself, so they opt back into the gate.
        tracker = ReanchoringReference(
            GradientECC(on_low_confidence=on_low_confidence),
            self.frames[0],
            mode=mode,
        )
        results, failures = {}, []
        for t in range(1, len(self.frames)):
            try:
                results[t] = tracker.estimate(t, self.frames[t])
            except RegistrationError:
                failures.append(t)
        return tracker, results, failures

    def test_fixed_mode_reproduces_the_failure_tail(self):
        _tracker, _results, failures = self._run("fixed")
        self.assertTrue(failures, "expected the fixed reference to fail late")
        # Contiguous tail, not scattered one-offs -- the signature of content
        # change rather than of individual bad frames.
        self.assertEqual(failures, list(range(failures[0], len(self.frames))))

    def test_reanchor_mode_registers_every_frame_accurately(self):
        tracker, results, failures = self._run("reanchor")
        self.assertEqual(failures, [])
        self.assertTrue(tracker.reanchor_events)
        for t, transform in results.items():
            dx, dy = self.truth[t]
            self.assertAlmostEqual(transform.dx, dx, delta=0.5)
            self.assertAlmostEqual(transform.dy, dy, delta=0.5)

    def test_keeping_low_confidence_fits_supersedes_reanchoring(self):
        """With the default policy an unconfident fit is a *success*, so the
        re-anchor fallback never sees it. Every frame still lands on target --
        by a different route, and flagged by a low confidence instead."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LowConfidenceWarning)
            tracker, results, failures = self._run("reanchor", "keep")
        self.assertEqual(failures, [])
        self.assertEqual(tracker.reanchor_events, [])
        self.assertEqual(len(results), len(self.frames) - 1)
        for t, transform in results.items():
            dx, dy = self.truth[t]
            self.assertAlmostEqual(transform.dx, dx, delta=0.5)
            self.assertAlmostEqual(transform.dy, dy, delta=0.5)
        self.assertLess(min(t.confidence for t in results.values()), 0.9)

    def test_chained_mode_accumulation_stays_bounded(self):
        """Chaining every frame works but accumulates composition error --
        measured at ~0.006 px/frame here, hence the preference for reanchor."""
        _tracker, results, failures = self._run("chained")
        self.assertEqual(failures, [])
        worst = max(
            abs(transform.dx - self.truth[t][0]) for t, transform in results.items()
        )
        self.assertLess(worst, 1.0)


class TestLowConfidencePolicy(unittest.TestCase):
    """The knob itself, independent of any one method's gate."""

    def test_unknown_policy_is_rejected_at_construction(self):
        for factory in (
            lambda p: GradientECC(on_low_confidence=p),
            lambda p: FeatureRANSACEuclidean(on_low_confidence=p),
            lambda p: MaskedTemplateCorrelation(
                _default_mask_rect(), on_low_confidence=p
            ),
        ):
            with self.assertRaisesRegex(ValueError, "on_low_confidence"):
                factory("warn")

    def test_keep_is_the_default(self):
        self.assertEqual(GradientECC().on_low_confidence, "keep")
        self.assertEqual(FeatureRANSACEuclidean().on_low_confidence, "keep")
        self.assertEqual(
            MaskedTemplateCorrelation(_default_mask_rect()).on_low_confidence,
            "keep",
        )

    def test_the_warning_reports_the_score_and_the_threshold(self):
        frames, _truth = _growing_colony_series(n_frames=40)
        with self.assertWarns(LowConfidenceWarning) as ctx:
            GradientECC().estimate(frames[0], frames[-1])
        message = str(ctx.warning)
        self.assertIn("min_confidence=0.9", message)
        # The measured coefficient, not just the threshold it missed.
        self.assertRegex(message, r"coefficient 0\.\d+")

    def test_ungated_methods_take_no_policy(self):
        """PhaseCorrelationHighpass/HoughLineRigidFit report no confidence, so
        there is nothing to gate -- they must not silently accept the knob."""
        for cls in (PhaseCorrelationHighpass, HoughLineRigidFit):
            with self.assertRaises(TypeError):
                cls(on_low_confidence="keep")


if __name__ == "__main__":
    unittest.main()
