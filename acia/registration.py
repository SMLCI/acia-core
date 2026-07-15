"""Pluggable frame-to-frame registration methods for drift correction.

This module provides a small, dependency-free (beyond ``opencv-python-headless``
and ``scikit-image``, both already core dependencies) abstraction for estimating
the rigid drift between a reference frame and a later frame of a time-lapse
sequence: :class:`RegistrationMethod`. Five concrete implementations are
provided, each exploiting a different signal in the image pair. They are
peers -- there is no "winner", no central registry, and no dispatch logic.
Callers construct the concrete class they want and call :meth:`estimate`
directly, exactly like :class:`~acia.segm.filter.CellFilter` or
``PropertyExtractor``.

All methods operate on plain ``np.ndarray`` frame pairs (grayscale ``(H, W)``
or multi-channel ``(H, W, C)``) -- there is no :class:`~acia.base.ImageSequenceSource`
plumbing, no calibration, and no pint units here; that is deliberately out of
scope for this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

from acia.base import RotatedCropSpec


class RegistrationError(Exception):
    """Raised when a :class:`RegistrationMethod` cannot produce a confident estimate.

    Every method raises this instead of silently returning an identity/zero
    transform: a silent zero-transform on failure would be indistinguishable
    from "genuinely no drift" and would corrupt any downstream comparison.
    """


@dataclass(frozen=True)
class FrameTransform:
    """A rigid (translation + rotation) transform between two frames.

    ``theta`` follows the same convention as :attr:`~acia.base.RotatedCropSpec.angle`:
    degrees, counter-clockwise, matching OpenCV's ``getRotationMatrix2D``. The
    rotation is understood to pivot about the frame's geometric center,
    followed by a translation of ``(dx, dy)`` pixels.

    Attributes:
        dx: Translation along the image x-axis, in pixels.
        dy: Translation along the image y-axis, in pixels.
        theta: Rotation angle in degrees, counter-clockwise. Defaults to
            ``0.0`` for translation-only methods; always set explicitly
            (never omitted) so a translation-only result is unambiguous.
    """

    dx: float
    dy: float
    theta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-friendly dict representation.

        Returns:
            dict: ``{"dx": dx, "dy": dy, "theta": theta}``.
        """
        return {"dx": self.dx, "dy": self.dy, "theta": self.theta}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameTransform:
        """Build a :class:`FrameTransform` from a plain dict.

        Args:
            data: A mapping as produced by :meth:`to_dict`. ``theta`` may be
                omitted, defaulting to ``0.0``.

        Returns:
            FrameTransform: The reconstructed transform.
        """
        return cls(
            dx=float(data["dx"]),
            dy=float(data["dy"]),
            theta=float(data.get("theta", 0.0)),
        )


def apply_correction(frame: np.ndarray, transform: FrameTransform) -> np.ndarray:
    """Undo an estimated drift by inverting and applying its forward transform.

    ``transform`` is the :class:`FrameTransform` estimated as reference->frame;
    inverting it and warping ``frame`` maps it back onto the reference frame's
    coordinate system. Same convention as the rotation-about-center + explicit
    translation used throughout this module (and mirrored by
    :class:`~acia.base.RotatedCropSequenceSource`'s own warp matrix).

    This is the single place this warp math lives -- the verify view, the
    batch-apply step, and :class:`~acia.base.RegisteredSequenceSource` all call
    this one implementation.

    Args:
        frame: The frame to correct, grayscale ``(H, W)`` or multi-channel
            ``(H, W, C)``.
        transform: The ``(dx, dy, theta)`` estimated as reference->``frame``.

    Returns:
        np.ndarray: ``frame`` warped back onto the reference frame's coordinate
            system, same shape as ``frame``.
    """
    h, w = frame.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, transform.theta, 1.0)
    matrix[0, 2] += transform.dx
    matrix[1, 2] += transform.dy
    inverse = cv2.invertAffineTransform(matrix)

    if frame.ndim == 2:
        # grayscale (H, W): warpAffine handles directly
        return cv2.warpAffine(frame, inverse, (w, h), flags=cv2.INTER_LINEAR)

    if frame.shape[2] <= 4:
        # up to 4 channels: warpAffine handles directly, but it collapses a
        # trailing singleton channel -- restore the (H, W, C) axis so a
        # (H, W, 1) frame does not silently become a 2D frame (same fix as
        # RotatedCropSequenceSource._warp in acia.base).
        out = cv2.warpAffine(frame, inverse, (w, h), flags=cv2.INTER_LINEAR)
        return out if out.ndim == 3 else out[..., None]

    # cv2.warpAffine only supports <= 4 channels: warp per-channel and re-stack
    channels = [
        cv2.warpAffine(frame[..., c], inverse, (w, h), flags=cv2.INTER_LINEAR)
        for c in range(frame.shape[2])
    ]
    return np.stack(channels, axis=-1)


class RegistrationMethod(ABC):
    """Base class for a pluggable frame-to-frame registration method.

    Mirrors the ``CellFilter``/``PropertyExtractor`` extension style: each
    concrete method is a standalone class implementing :meth:`estimate` with
    the same signature. Instances are used directly -- there is no registry
    and no central dispatch. Adding a new method is just a new subclass.
    """

    @abstractmethod
    def estimate(self, reference: np.ndarray, frame: np.ndarray) -> FrameTransform:
        """Estimate the rigid transform that maps ``reference`` to ``frame``.

        Args:
            reference: The reference frame, grayscale ``(H, W)`` or
                multi-channel ``(H, W, C)``.
            frame: The frame to register against ``reference``, same shape
                convention as ``reference``.

        Returns:
            FrameTransform: The estimated ``(dx, dy, theta)``.

        Raises:
            RegistrationError: If no confident estimate can be produced.
        """


def build_sample_frame_indices(
    size_t: int, reference_index: int, n_sample_frames: int
) -> list[int]:
    """Evenly-spaced comparison-frame indices across ``[0, size_t)``.

    Shared by the synthetic and real-data sections of the comparison notebook
    and by the dashboard's verify step -- a bug here is caught once, before
    real data is ever touched.

    - Raises ``ValueError`` if ``n_sample_frames < 1``: a silently empty/
      degenerate sample list would be worse than a loud, immediate failure.
    - Deduplicates (``sorted(set(...))``) since ``np.linspace(...).round()``
      can produce repeated indices once ``n_sample_frames`` approaches or
      exceeds ``size_t``; if that dedup drops below the requested count, a
      one-line note is printed so the shortfall isn't silently misleading.
    - Excludes ``reference_index`` where present: comparing the reference
      frame against itself is a trivial identity case that wastes a sample
      and inflates success/drift-magnitude statistics -- unless doing so
      would leave zero frames, in which case ``reference_index`` is kept and
      a note is printed that it's a trivial self-comparison.

    Args:
        size_t: Number of frames in the sequence.
        reference_index: The reference frame index to exclude where possible.
        n_sample_frames: Requested number of sample indices (``>= 1``).

    Returns:
        list[int]: Sorted, deduplicated sample frame indices.

    Raises:
        ValueError: If ``n_sample_frames < 1``.
    """
    if n_sample_frames < 1:
        raise ValueError(f"n_sample_frames must be >= 1, got {n_sample_frames}.")

    indices = sorted(
        {int(t) for t in np.linspace(0, size_t - 1, n_sample_frames).round()}
    )
    if len(indices) < n_sample_frames:
        print(
            f"note: only {len(indices)} unique frame indices sampled "
            f"(requested {n_sample_frames}) -- np.linspace produced duplicates "
            f"given size_t={size_t}."
        )

    without_reference = [t for t in indices if t != reference_index]
    if not without_reference:
        print(
            f"note: reference_index={reference_index} is the only sampled "
            "frame -- keeping it, but estimate(reference, reference) is a "
            "trivial self-comparison (identity transform expected)."
        )
        return indices

    return without_reference


def run_comparison(
    methods: Mapping[str, RegistrationMethod],
    reference_frame: np.ndarray,
    get_frame: Callable[[int], np.ndarray],
    frame_indices: list[int],
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[FrameTransform | None]]:
    """Run every method in ``methods`` against every frame in ``frame_indices``.

    ``get_frame(t)`` fetches the comparison frame for index ``t`` (a real
    ``source.get_frame(t).raw`` or a synthetic in-memory lookup -- the same
    function drives both the synthetic and real-data sections, and the
    dashboard's verify step). A failure in one ``(method, frame)`` pair is
    caught and recorded as ``None`` without ever stopping the other methods or
    frames: broadened from :class:`RegistrationError` to ``Exception`` so a
    genuinely unanticipated error can't silently abort the whole run, and
    every ``results[name]`` list ends up with exactly one entry per entry in
    ``frame_indices`` (never ragged).

    Args:
        methods: Mapping of method name -> :class:`RegistrationMethod`
            instance.
        reference_frame: The reference frame passed to every ``estimate`` call.
        get_frame: Callable returning the comparison frame for a given frame
            index.
        frame_indices: Frame indices to compare against ``reference_frame``.
        on_progress: Optional callback invoked as ``on_progress(i, total)``
            after every ``frame_indices[i]`` has been compared against every
            method (``total = len(frame_indices)``) -- lets a caller (e.g. the
            dashboard's verify step) report per-frame progress without this
            function knowing anything about UI. Defaults to ``None`` so
            existing callers are unaffected. A failure raised by the callback
            itself is caught and printed the same way a per-method failure
            is, rather than aborting the rest of the comparison.

    Returns:
        dict[str, list]: method name -> list of ``FrameTransform | None``, one
            entry per ``frame_indices`` entry, in the same order.
    """
    results: dict[str, list[FrameTransform | None]] = {name: [] for name in methods}
    total = len(frame_indices)
    for i, t in enumerate(frame_indices):
        comparison_frame = get_frame(t)
        for name, method in methods.items():
            try:
                transform: FrameTransform | None = method.estimate(
                    reference_frame, comparison_frame
                )
            except Exception as e:  # noqa: BLE001 -- deliberately broad, see docstring
                print(f"frame {t}: {name} failed -- {type(e).__name__}: {e}")
                transform = None
            results[name].append(transform)
        if on_progress is not None:
            try:
                on_progress(i, total)
            except Exception as e:  # noqa: BLE001 -- isolate: a broken progress
                # callback (e.g. RegistrationDashboard.send on a closed comm
                # channel) must not abort the rest of the comparison.
                print(f"frame {t}: on_progress failed -- {type(e).__name__}: {e}")
    return results


def _to_grayscale_f32(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale or multi-channel frame to a ``float32`` grayscale array.

    Accepts ``(H, W)`` grayscale or ``(H, W, C)`` multi-channel input. A
    3-channel input is converted via ``cv2.cvtColor`` (BGR-style
    coefficients); any other channel count (including 1) falls back to a
    plain mean over the channel axis.

    Args:
        image: Input frame, ``(H, W)`` or ``(H, W, C)``.

    Returns:
        np.ndarray: ``(H, W)`` grayscale array, dtype ``float32``.

    Raises:
        RegistrationError: If ``image`` has neither 2 nor 3 dimensions.
    """
    arr = np.asarray(image).astype(np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] == 3:
            gray: np.ndarray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            return gray
        mean_gray: np.ndarray = arr.mean(axis=-1).astype(np.float32)
        return mean_gray
    raise RegistrationError(
        f"Unsupported frame shape {arr.shape}; expected (H, W) or (H, W, C)."
    )


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient-magnitude ("high-pass") of a grayscale frame.

    Args:
        gray: ``(H, W)`` grayscale array, dtype ``float32``.

    Returns:
        np.ndarray: ``(H, W)`` gradient-magnitude array, dtype ``float32``.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _grayscale_gradient(image: np.ndarray) -> np.ndarray:
    """Shared preprocessing for the two gradient-domain methods.

    Converts ``image`` to grayscale (see :func:`_to_grayscale_f32`) and then
    to a gradient-magnitude ("edge-emphasized") representation (see
    :func:`_gradient_magnitude`). Used by both :class:`PhaseCorrelationHighpass`
    and :class:`GradientECC`.

    Args:
        image: Input frame, ``(H, W)`` or ``(H, W, C)``.

    Returns:
        np.ndarray: ``(H, W)`` gradient-magnitude array, dtype ``float32``.
    """
    return _gradient_magnitude(_to_grayscale_f32(image))


def _build_gray_pyramid(
    gray: np.ndarray, max_levels: int, min_size: int
) -> list[np.ndarray]:
    """Finest-to-coarsest Gaussian pyramid of a grayscale frame.

    Repeatedly halves ``gray`` via ``cv2.pyrDown`` (Gaussian blur + downsample,
    avoids the aliasing a naive strided subsample would introduce), stopping
    once ``max_levels`` is reached or the next candidate level would fall
    below ``min_size`` on its shorter side.

    Args:
        gray: ``(H, W)`` grayscale array, dtype ``float32``.
        max_levels: Hard cap on the number of levels returned.
        min_size: Minimum allowed length, in pixels, of a level's shorter side.

    Returns:
        list[np.ndarray]: Levels ordered finest-to-coarsest; ``levels[0] is gray``.
    """
    levels = [gray]
    while len(levels) < max_levels:
        candidate = cv2.pyrDown(levels[-1])
        if min(candidate.shape[:2]) < min_size:
            break
        levels.append(candidate)
    return levels


def _decompose_similarity(
    matrix: np.ndarray, center: tuple[float, float]
) -> FrameTransform:
    """Decompose a 2x3 similarity matrix into a center-pivot :class:`FrameTransform`.

    ``matrix`` is assumed to map ``reference`` pixel coordinates to ``frame``
    pixel coordinates (forward warp, the same direction cv2's
    ``estimateAffinePartial2D(src=reference_pts, dst=frame_pts)`` and
    ``findTransformECC(templateImage=reference, inputImage=frame, ...)``
    both produce). The decomposition assumes the similarity represents a
    rotation about ``center`` by ``theta`` degrees (CCW, matching
    ``cv2.getRotationMatrix2D``) followed by a translation of ``(dx, dy)``.

    Args:
        matrix: 2x3 similarity (translation + rotation [+ uniform scale])
            matrix, ``[[a, b, tx], [c, d, ty]]``.
        center: The ``(cx, cy)`` pivot point used to interpret ``theta``.

    Returns:
        FrameTransform: The decomposed ``(dx, dy, theta)``.

    Raises:
        RegistrationError: If ``matrix`` contains any non-finite (NaN/Inf)
            value -- decomposing it would silently propagate a bogus
            ``FrameTransform`` rather than failing loudly.
    """
    if not np.isfinite(matrix).all():
        raise RegistrationError(
            "_decompose_similarity: input matrix contains non-finite "
            "(NaN/Inf) values; refusing to decompose."
        )
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    cx, cy = center
    theta = float(np.degrees(np.arctan2(b - c, a + d)))
    dx = float(tx - cx + a * cx + b * cy)
    dy = float(ty - cy + c * cx + d * cy)
    return FrameTransform(dx=dx, dy=dy, theta=theta)


class PhaseCorrelationHighpass(RegistrationMethod):
    """Translation-only registration via high-pass phase correlation.

    Both frames are converted to a Sobel gradient-magnitude representation
    (see :func:`_grayscale_gradient`) -- a cheap high-pass filter that
    suppresses slow-varying content (e.g. a growing colony's interior) and
    emphasizes sharp structure (e.g. device edges) -- and then registered
    with :func:`skimage.registration.phase_cross_correlation`, which recovers
    a subpixel translation via FFT cross-correlation with upsampling.

    This method is translation-only: ``theta`` is always ``0.0``.

    Limitation (not a bug -- no test required): like any FFT-based phase
    correlation, the implicit search range is bounded by the frame size, and
    accuracy degrades as the true shift approaches a large fraction of the
    frame dimensions (periodic wraparound).

    Attributes:
        upsample_factor: Subpixel upsampling factor passed to
            ``phase_cross_correlation`` (higher = finer subpixel precision,
            at increased compute cost).
        min_gradient_std: Minimum standard deviation of the gradient-magnitude
            image required to consider a frame to have detectable signal;
            below this, the frame is treated as blank/textureless.
    """

    def __init__(self, upsample_factor: int = 20, min_gradient_std: float = 1e-3):
        self.upsample_factor = upsample_factor
        self.min_gradient_std = min_gradient_std

    def estimate(self, reference: np.ndarray, frame: np.ndarray) -> FrameTransform:
        """See :meth:`RegistrationMethod.estimate`."""
        if reference.shape[:2] != frame.shape[:2]:
            raise RegistrationError(
                "PhaseCorrelationHighpass: reference and frame must have the "
                f"same (H, W) shape, got {reference.shape[:2]} and "
                f"{frame.shape[:2]}."
            )
        ref_grad = _grayscale_gradient(reference)
        frm_grad = _grayscale_gradient(frame)

        if (
            not np.isfinite(ref_grad).all()
            or not np.isfinite(frm_grad).all()
            or float(ref_grad.std()) < self.min_gradient_std
            or float(frm_grad.std()) < self.min_gradient_std
        ):
            raise RegistrationError(
                "PhaseCorrelationHighpass: no detectable gradient signal in "
                "reference or frame (blank/textureless or non-finite input)."
            )

        shift, _error, _phasediff = phase_cross_correlation(
            ref_grad, frm_grad, upsample_factor=self.upsample_factor
        )
        # skimage returns the shift (row, col) that would need to be applied
        # to `frame` (via e.g. scipy.ndimage.shift) to align it back onto
        # `reference`; the drift we report is the opposite of that.
        dy, dx = shift
        return FrameTransform(dx=float(-dx), dy=float(-dy), theta=0.0)


class MaskedTemplateCorrelation(RegistrationMethod):
    """Translation-only registration via masked normalized template matching.

    A rectangular region of interest (``mask_rect``, e.g. drawn around a
    static microfluidic channel wall or trap chamber) is cropped from the
    reference frame and matched against a padded search window in the target
    frame via ``cv2.matchTemplate(..., TM_CCOEFF_NORMED)``, with the best
    match refined to subpixel precision by parabolic interpolation of the
    correlation surface around its peak.

    ``mask_rect`` is expected to have ``angle == 0`` in typical use (this is
    a comparison-only method, not a production pipeline), but a non-zero
    angle is still honored for forward-compatibility, using the same
    rotate-then-crop math as :class:`~acia.base.RotatedCropSequenceSource`.
    ``RotatedCropSpec`` is reused here purely as an inert data container --
    no other coupling to :mod:`acia.base` is introduced.

    This method is translation-only: ``theta`` is always ``0.0``.

    Limitation (not a bug -- no test required): the true shift must fall
    within ``search_margin`` pixels of the template's nominal location, or
    the match is rejected outright (see the "masked search exceeds window"
    row of the I/O matrix) rather than returning a wrong answer.

    Attributes:
        mask_rect: The rectangle to crop from the reference frame as the
            template.
        search_margin: Extra pixels of search radius (in each spatial
            direction) added around ``mask_rect``'s nominal location in the
            target frame.
        min_score: Minimum acceptable ``TM_CCOEFF_NORMED`` peak score;
            matches below this are rejected as unconfident.
    """

    def __init__(
        self,
        mask_rect: RotatedCropSpec,
        search_margin: int = 30,
        min_score: float = 0.5,
    ):
        self.mask_rect = mask_rect
        self.search_margin = search_margin
        self.min_score = min_score

    def estimate(self, reference: np.ndarray, frame: np.ndarray) -> FrameTransform:
        """See :meth:`RegistrationMethod.estimate`."""
        if reference.shape[:2] != frame.shape[:2]:
            raise RegistrationError(
                "MaskedTemplateCorrelation: reference and frame must have the "
                f"same (H, W) shape, got {reference.shape[:2]} and "
                f"{frame.shape[:2]}."
            )
        gray_ref = _to_grayscale_f32(reference)
        gray_frm = _to_grayscale_f32(frame)

        cx, cy = self.mask_rect.center
        w, h = self.mask_rect.size

        crop_matrix = cv2.getRotationMatrix2D((cx, cy), self.mask_rect.angle, 1.0)
        crop_matrix[0, 2] += w / 2 - cx
        crop_matrix[1, 2] += h / 2 - cy
        template = cv2.warpAffine(gray_ref, crop_matrix, (w, h), flags=cv2.INTER_LINEAR)

        # The search window must be cropped with the SAME rotation as the
        # template (just re-centered for the larger padded canvas) -- using a
        # pure axis-aligned crop here (as before) put the template and search
        # patches in inconsistent orientations for any non-zero
        # ``mask_rect.angle``, degrading (and even sign-flipping) the
        # recovered correlation peak.
        margin = self.search_margin
        search_w, search_h = w + 2 * margin, h + 2 * margin
        search_matrix = cv2.getRotationMatrix2D((cx, cy), self.mask_rect.angle, 1.0)
        search_matrix[0, 2] += search_w / 2 - cx
        search_matrix[1, 2] += search_h / 2 - cy
        search = cv2.warpAffine(
            gray_frm, search_matrix, (search_w, search_h), flags=cv2.INTER_LINEAR
        )

        if float(template.std()) < 1e-3 or float(search.std()) < 1e-3:
            raise RegistrationError(
                "MaskedTemplateCorrelation: template or search window has no "
                "detectable texture (blank/textureless input)."
            )

        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
        mx, my = max_loc
        res_h, res_w = result.shape

        if not np.isfinite(max_val):
            raise RegistrationError(
                "MaskedTemplateCorrelation: correlation peak is non-finite "
                "(NaN/Inf); refusing to trust the match."
            )
        if mx <= 0 or my <= 0 or mx >= res_w - 1 or my >= res_h - 1:
            raise RegistrationError(
                "MaskedTemplateCorrelation: best match sits on the search-window "
                "edge; the true shift likely exceeds search_margin="
                f"{self.search_margin}."
            )
        if max_val < self.min_score:
            raise RegistrationError(
                f"MaskedTemplateCorrelation: best correlation score {max_val:.3f} "
                f"is below min_score={self.min_score}."
            )

        dx_sub = _parabolic_refine(
            result[my, mx - 1], result[my, mx], result[my, mx + 1]
        )
        dy_sub = _parabolic_refine(
            result[my - 1, mx], result[my, mx], result[my + 1, mx]
        )

        # `(mx + dx_sub, my + dy_sub)` is the matched offset within the
        # search canvas, measured relative to the template's own canvas --
        # i.e. `margin + R @ (dx, dy)` where `R` is the 2x2 rotation applied
        # by `crop_matrix`/`search_matrix` (identity when `mask_rect.angle`
        # is `0.0`). Un-rotate by `R^-1` (`= R^T`, since `R` is orthonormal)
        # to recover the true image-axis-aligned `(dx, dy)`.
        raw_dx = float(mx + dx_sub - margin)
        raw_dy = float(my + dy_sub - margin)
        alpha, beta = float(crop_matrix[0, 0]), float(crop_matrix[0, 1])
        dx = alpha * raw_dx - beta * raw_dy
        dy = beta * raw_dx + alpha * raw_dy

        return FrameTransform(dx=dx, dy=dy, theta=0.0)


def _parabolic_refine(c_minus: float, c_zero: float, c_plus: float) -> float:
    """Subpixel peak offset via parabolic interpolation of three samples.

    Args:
        c_minus: Correlation value one sample before the peak.
        c_zero: Correlation value at the (integer) peak.
        c_plus: Correlation value one sample after the peak.

    Returns:
        float: The subpixel offset from the integer peak, clamped to
            ``[-0.5, 0.5]`` (``0.0`` if the samples are degenerate or any
            sample is non-finite).
    """
    if not (np.isfinite(c_minus) and np.isfinite(c_zero) and np.isfinite(c_plus)):
        return 0.0
    denom = c_minus - 2 * c_zero + c_plus
    if abs(denom) < 1e-12:
        return 0.0
    offset = 0.5 * (c_minus - c_plus) / denom
    return float(np.clip(offset, -0.5, 0.5))


class HoughLineRigidFit(RegistrationMethod):
    """Rigid registration via straight-edge (Hough line) matching.

    Targets the microfluidic device's rigid, static geometry (channel walls,
    trap chamber edges) rather than the growing/dividing cell colony, which
    should show up as non-line clutter that this method simply ignores.

    Pipeline: Canny edge detection, then ``cv2.HoughLinesP`` to find straight
    segments; near-duplicate detections of the same physical edge (e.g. both
    sides of a thin stroke's Canny response) are merged by clustering on
    (angle, perpendicular offset); the ``top_n`` longest survivors are kept
    and refined to subpixel position via an intensity-centroid profile
    (robust to which side of a thin edge Canny happens to trace). Frame lines
    are matched to reference lines by nearest angle then nearest
    perpendicular offset; the median angle shift across matched pairs gives
    ``theta``, and a small (2 DOF) least-squares solve over each matched
    pair's own perpendicular-offset equation gives ``(dx, dy)``.

    Limitation (not a bug -- no test required): needs at least two matched
    lines of sufficiently different orientation (e.g. a horizontal and a
    vertical device edge) to solve for translation; a scene with only
    parallel lines cannot constrain the perpendicular-to-those-lines
    translation component and will raise :class:`RegistrationError`. Also,
    ``theta`` recovery is only meaningful up to the scene's own rotational
    symmetry period: for a scene with periodic structure (e.g. a square
    grid, symmetric under 90-degree rotation), a true rotation near/at that
    period aliases to a much smaller apparent angle (empirically, a true 80
    degree rotation of a square grid was reported as roughly -10 degrees).
    This is not a bug -- the spec's acceptance criterion only requires
    correct recovery for ``theta`` up to 5 degrees, well inside any
    realistic scene's symmetry period, so it already holds; it is simply not
    safe to extrapolate this method's ``theta`` output to large rotations.

    The line-to-line correspondence across frames is the highest-risk part
    of this method: an ambiguous or absent match honestly raises
    :class:`RegistrationError` rather than guessing. Concretely,
    :meth:`_match_lines` requires the best-matching reference line to be
    decisively closer (by perpendicular offset) than the second-best
    candidate, and resolves any reference line claimed by more than one
    frame line in favor of the closer match -- see :meth:`_match_lines` for
    the exact thresholds.

    Attributes:
        top_n: Number of longest (deduplicated) line segments to keep per
            frame.
        canny_thresholds: ``(low, high)`` thresholds for ``cv2.Canny``.
        hough_threshold: Accumulator threshold for ``cv2.HoughLinesP``.
        min_line_length: Minimum segment length for ``cv2.HoughLinesP``.
        max_line_gap: Maximum gap to bridge when merging collinear segments
            in ``cv2.HoughLinesP``.
        angle_tolerance: Maximum angle difference (degrees) for a frame line
            to be considered a candidate match of a reference line.
    """

    def __init__(
        self,
        top_n: int = 10,
        canny_thresholds: tuple[int, int] = (50, 150),
        hough_threshold: int = 60,
        min_line_length: int = 60,
        max_line_gap: int = 15,
        angle_tolerance: float = 20.0,
    ):
        self.top_n = top_n
        self.canny_thresholds = canny_thresholds
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.angle_tolerance = angle_tolerance

    def estimate(self, reference: np.ndarray, frame: np.ndarray) -> FrameTransform:
        """See :meth:`RegistrationMethod.estimate`."""
        ref_gray = _to_grayscale_f32(reference)
        frm_gray = _to_grayscale_f32(frame)
        if ref_gray.shape != frm_gray.shape:
            raise RegistrationError(
                "HoughLineRigidFit: reference and frame must have the same "
                f"shape, got {ref_gray.shape} and {frm_gray.shape}."
            )
        h, w = ref_gray.shape
        center = np.array([w / 2.0, h / 2.0])

        ref_lines = self._detect_lines(ref_gray, center)
        frm_lines = self._detect_lines(frm_gray, center)
        if not ref_lines or not frm_lines:
            raise RegistrationError(
                "HoughLineRigidFit: no straight edges detected in reference "
                "and/or frame."
            )

        pairs = self._match_lines(frm_lines, ref_lines, center)
        if len(pairs) < 2:
            raise RegistrationError(
                "HoughLineRigidFit: fewer than 2 matched line pairs; cannot "
                "constrain a rigid transform."
            )

        angle_diffs = [_angle_diff(frm.angle, ref.angle) for frm, ref in pairs]
        theta = -float(np.median(angle_diffs))

        rows = []
        rhs = []
        for frm, ref in pairs:
            n_frame = _line_normal(frm.angle)
            n_ref = _line_normal(ref.angle)
            if n_frame @ n_ref < 0:
                n_ref = -n_ref
            rho_frame = float(n_frame @ (np.array([frm.x, frm.y]) - center))
            rho_ref = float(n_ref @ (np.array([ref.x, ref.y]) - center))
            rows.append(n_frame)
            rhs.append(rho_frame - rho_ref)

        design = np.array(rows)
        target = np.array(rhs)
        if np.linalg.matrix_rank(design) < 2:
            raise RegistrationError(
                "HoughLineRigidFit: matched lines are all (near-)parallel; "
                "cannot solve for a unique translation."
            )

        translation, *_ = np.linalg.lstsq(design, target, rcond=None)
        dx, dy = translation
        return FrameTransform(dx=float(dx), dy=float(dy), theta=theta)

    def _detect_lines(
        self, gray: np.ndarray, center: np.ndarray
    ) -> list[_DetectedLine]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        low, high = self.canny_thresholds
        normalized = cv2.normalize(  # type: ignore[call-overload]
            blurred, None, 0, 255, cv2.NORM_MINMAX
        )
        blurred_u8 = normalized.astype(np.uint8)
        edges = cv2.Canny(blurred_u8, low, high)
        raw = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        if raw is None:
            return []

        candidates = []
        for x1, y1, x2, y2 in raw[:, 0, :]:
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = _wrap_angle(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            candidates.append((angle, length, mx, my))
        candidates.sort(key=lambda t: -t[1])

        deduped = _dedupe_lines(candidates, center)
        deduped = deduped[: self.top_n]

        return [
            self._refine_line(blurred, angle, length, mx, my)
            for angle, length, mx, my in deduped
        ]

    def _refine_line(
        self, blurred: np.ndarray, angle: float, length: float, mx: float, my: float
    ) -> _DetectedLine:
        """Subpixel line-position refine via an intensity-centroid profile.

        Rotates the whole (blurred) frame so the candidate line becomes
        horizontal, then computes the intensity-weighted centroid row across
        a narrow perpendicular band spanning the line's length. This is
        robust to which side of a thin, anti-aliased edge Canny happens to
        trace (a plain fit over raw Canny edge points is not: it can
        systematically favor one side over the other by about a pixel,
        depending on noise).
        """
        h, w = blurred.shape
        img_center = (w / 2.0, h / 2.0)
        rot_matrix = cv2.getRotationMatrix2D(img_center, angle, 1.0)
        rotated = cv2.warpAffine(blurred, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)

        rx, ry = rot_matrix @ np.array([mx, my, 1.0])
        y0 = int(round(ry))
        half_band = 6
        x_lo = max(0, int(rx - length / 2))
        x_hi = min(w, int(rx + length / 2))
        y_lo = max(0, y0 - half_band)
        y_hi = min(h, y0 + half_band + 1)

        if x_hi - x_lo < 10 or y_hi - y_lo < 3:
            return _DetectedLine(angle=angle, length=length, x=mx, y=my)

        band = rotated[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
        profile = band.mean(axis=1)
        baseline = float(np.percentile(profile, 20))
        weights = np.clip(profile - baseline, 0, None)
        if weights.sum() < 1e-6:
            return _DetectedLine(angle=angle, length=length, x=mx, y=my)

        rows = np.arange(y_lo, y_hi)
        centroid_row = float((weights * rows).sum() / weights.sum())

        inv_matrix = cv2.invertAffineTransform(rot_matrix)
        ox, oy = inv_matrix @ np.array([rx, centroid_row, 1.0])
        return _DetectedLine(angle=angle, length=length, x=float(ox), y=float(oy))

    def _match_lines(
        self,
        frame_lines: list[_DetectedLine],
        ref_lines: list[_DetectedLine],
        center: np.ndarray,
    ) -> list[tuple[_DetectedLine, _DetectedLine]]:
        """Match each frame line to its most likely reference-line counterpart.

        Two disambiguation guards protect against silently locking onto the
        wrong physical line -- the dominant failure mode for periodic or
        repeated structure (e.g. an evenly-spaced grid of parallel channel
        walls, once true drift exceeds roughly half the line spacing):

        1. Ambiguity check: a frame line's best (closest perpendicular
           offset) reference candidate must be decisively closer than the
           second-best. "Decisively" means the second-best distance is at
           least twice the best distance (using ``max(best, 1.0)`` px as the
           floor for the comparison, so a near-zero best distance still
           requires the runner-up to be at least ~2 px away rather than
           trivially satisfying a `2x` ratio against ~0). If this fails, the
           frame line is left unmatched rather than guessing.
        2. Reuse check: if two different frame lines both end up claiming
           the same reference line (each individually passing guard 1), only
           the closer of the two is kept; the other is treated as unmatched.

        If ambiguity/reuse resolution leaves fewer than 2 matched pairs
        overall, :meth:`estimate` raises :class:`RegistrationError` (cannot
        constrain a rigid transform).
        """
        tentative: list[tuple[_DetectedLine, _DetectedLine, float]] = []
        for frm in frame_lines:
            candidates = [
                ref
                for ref in ref_lines
                if abs(_angle_diff(frm.angle, ref.angle)) < self.angle_tolerance
            ]
            if not candidates:
                continue
            n_frame = _line_normal(frm.angle)
            rho_frame = float(n_frame @ (np.array([frm.x, frm.y]) - center))
            ranked = sorted(
                (
                    abs(n_frame @ (np.array([ref.x, ref.y]) - center) - rho_frame),
                    i,
                    ref,
                )
                for i, ref in enumerate(candidates)
            )
            best_dist, _best_i, best_ref = ranked[0]
            if len(ranked) > 1:
                second_dist, _second_i, _second_ref = ranked[1]
                # A small multiplicative margin (2.05x rather than an exact
                # 2.0x) keeps this robust to sub-pixel numerical jitter in
                # the refined line positions landing exactly on the 2x
                # boundary -- an exact "<" or "<=" comparison there is a
                # coin flip in practice.
                if second_dist <= 2.05 * max(best_dist, 1.0):
                    # Ambiguous: the runner-up reference candidate is not
                    # decisively farther away than the best one -- e.g. two
                    # adjacent grid lines are both plausible. Leave this
                    # frame line unmatched rather than risk a confidently
                    # wrong pairing.
                    continue
            tentative.append((frm, best_ref, best_dist))

        # Resolve reference-line reuse: keep only the closest match for any
        # reference line claimed by more than one frame line.
        best_by_ref: dict[int, tuple[_DetectedLine, _DetectedLine, float]] = {}
        for frm, ref, dist in tentative:
            key = id(ref)
            existing = best_by_ref.get(key)
            if existing is None or dist < existing[2]:
                best_by_ref[key] = (frm, ref, dist)

        return [(frm, ref) for frm, ref, _dist in best_by_ref.values()]


@dataclass(frozen=True)
class _DetectedLine:
    """A detected/refined straight-line candidate used by :class:`HoughLineRigidFit`."""

    angle: float
    length: float
    x: float
    y: float


def _wrap_angle(angle: float) -> float:
    """Wrap a line-direction angle (degrees) into ``(-90, 90]``."""
    if angle <= -90:
        return angle + 180
    if angle > 90:
        return angle - 180
    return angle


def _angle_diff(a: float, b: float) -> float:
    """Signed difference ``a - b`` for line-direction angles, wrapped mod 180."""
    return ((a - b + 90) % 180) - 90


def _line_normal(angle_deg: float) -> np.ndarray:
    """Unit normal vector of a line with the given direction angle (degrees)."""
    theta_rad = np.radians(angle_deg)
    return np.array([-np.sin(theta_rad), np.cos(theta_rad)])


def _dedupe_lines(
    candidates: list[tuple[float, float, float, float]], center: np.ndarray
) -> list[tuple[float, float, float, float]]:
    """Merge near-duplicate detections of the same physical line.

    A thin stroke's Canny response often yields two nearly-parallel,
    nearly-collinear segments (one per edge of the stroke); this clusters
    candidates by ``(angle, perpendicular offset)`` and keeps only the
    longest per cluster.

    Args:
        candidates: ``(angle, length, mx, my)`` tuples, longest first.
        center: The ``(cx, cy)`` point used to compute each candidate's
            perpendicular offset (``rho``).

    Returns:
        list: Deduplicated ``(angle, length, mx, my)`` tuples, longest first.
    """
    deduped: list[tuple[float, float, float, float, float]] = []
    for angle, length, mx, my in candidates:
        normal = _line_normal(angle)
        rho = float(normal @ (np.array([mx, my]) - center))
        merged = False
        for i, (d_angle, d_length, _d_mx, _d_my, d_rho) in enumerate(deduped):
            if abs(_angle_diff(angle, d_angle)) < 5.0 and abs(rho - d_rho) < 6.0:
                merged = True
                if length > d_length:
                    deduped[i] = (angle, length, mx, my, rho)
                break
        if not merged:
            deduped.append((angle, length, mx, my, rho))
    deduped.sort(key=lambda t: -t[1])
    return [(angle, length, mx, my) for angle, length, mx, my, _rho in deduped]


class FeatureRANSACEuclidean(RegistrationMethod):
    """Rigid registration via ORB features + RANSAC-fit Euclidean transform.

    ORB keypoints/descriptors are detected in both frames, matched with a
    Hamming-distance ``cv2.BFMatcher`` plus Lowe's ratio test, and the
    surviving correspondences are fit with ``cv2.estimateAffinePartial2D``
    under ``cv2.RANSAC`` (robust to outlier matches, e.g. spurious features
    on the growing/dividing cell colony). The resulting 2x3 similarity
    matrix is decomposed into ``(dx, dy, theta)``.

    Limitation (not a bug -- no test required): needs enough distinctive,
    repeatable ORB keypoints; a scene dominated by smooth, low-contrast
    content with few corners (or a very large motion that exceeds ORB's
    matching range) will not produce enough inliers.

    Attributes:
        n_features: Maximum number of ORB features to detect per frame.
        ratio_thresh: Lowe's ratio-test threshold (lower = stricter).
        ransac_thresh: RANSAC reprojection-error threshold, in pixels.
        min_inliers: Minimum number of RANSAC inlier correspondences required
            to accept the fit.
    """

    def __init__(
        self,
        n_features: int = 500,
        ratio_thresh: float = 0.75,
        ransac_thresh: float = 3.0,
        min_inliers: int = 3,
    ):
        self.n_features = n_features
        self.ratio_thresh = ratio_thresh
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers

    def estimate(self, reference: np.ndarray, frame: np.ndarray) -> FrameTransform:
        """See :meth:`RegistrationMethod.estimate`."""
        if reference.shape[:2] != frame.shape[:2]:
            raise RegistrationError(
                "FeatureRANSACEuclidean: reference and frame must have the "
                f"same (H, W) shape, got {reference.shape[:2]} and "
                f"{frame.shape[:2]}."
            )
        ref_gray = _to_grayscale_f32(reference)
        frm_gray = _to_grayscale_f32(frame)
        ref_norm = cv2.normalize(  # type: ignore[call-overload]
            ref_gray, None, 0, 255, cv2.NORM_MINMAX
        )
        frm_norm = cv2.normalize(  # type: ignore[call-overload]
            frm_gray, None, 0, 255, cv2.NORM_MINMAX
        )
        ref_u8 = ref_norm.astype(np.uint8)
        frm_u8 = frm_norm.astype(np.uint8)

        orb = cv2.ORB_create(self.n_features)  # type: ignore[attr-defined]
        kp1, des1 = orb.detectAndCompute(ref_u8, None)
        kp2, des2 = orb.detectAndCompute(frm_u8, None)
        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            raise RegistrationError(
                "FeatureRANSACEuclidean: too few ORB keypoints detected "
                "(blank/textureless input?)."
            )

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn_matches = matcher.knnMatch(des1, des2, k=2)
        good = [
            pair[0]
            for pair in knn_matches
            if len(pair) == 2
            and pair[0].distance < self.ratio_thresh * pair[1].distance
        ]
        if len(good) < self.min_inliers:
            raise RegistrationError(
                f"FeatureRANSACEuclidean: only {len(good)} good matches survived "
                f"the ratio test; need >= {self.min_inliers}."
            )

        src_pts = np.array(
            [kp1[m.queryIdx].pt for m in good], dtype=np.float32
        ).reshape(-1, 1, 2)
        dst_pts = np.array(
            [kp2[m.trainIdx].pt for m in good], dtype=np.float32
        ).reshape(-1, 1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
        )
        if matrix is None:
            raise RegistrationError(
                "FeatureRANSACEuclidean: estimateAffinePartial2D found no model."
            )
        n_inliers = int(inliers.sum()) if inliers is not None else 0
        if n_inliers < self.min_inliers:
            raise RegistrationError(
                f"FeatureRANSACEuclidean: only {n_inliers} RANSAC inlier "
                f"correspondences; need >= {self.min_inliers}."
            )

        h, w = ref_gray.shape
        return _decompose_similarity(matrix, (w / 2.0, h / 2.0))


class GradientECC(RegistrationMethod):
    """Rigid registration via Enhanced Correlation Coefficient (ECC) maximization.

    Both frames are converted to a Sobel gradient-magnitude representation
    (see :func:`_grayscale_gradient`) -- emphasizing sharp device edges over
    the colony's smoother interior -- and ``cv2.findTransformECC`` iteratively
    refines a Euclidean (rotation + translation) warp maximizing the
    correlation coefficient between them. The resulting 2x3 matrix is
    decomposed into ``(dx, dy, theta)``.

    Limitation (not a bug -- no test required): ECC is a local optimizer;
    it needs a reasonable initial overlap and can fail to converge (raising
    :class:`RegistrationError`) for large motions well beyond its capture
    range. To improve convergence reliability within the intended
    happy-path envelope (translation up to roughly a dozen pixels, small
    rotation), estimation runs coarse-to-fine over a small image pyramid
    (see :func:`_build_gray_pyramid`): the coarsest level is seeded with a
    translation-only phase-correlation estimate (on that level's gradient
    images) rather than starting from the identity transform, and each
    finer level is seeded from the previous level's converged warp,
    scaled to that level's resolution; if the coarse phase-correlation
    pre-pass itself fails, the coarsest level simply falls back to an
    identity-seeded warp. This also substantially reduces the number of
    full-resolution ECC iterations needed on large frames, since the seed
    arriving at full resolution is already close to the true optimum. One
    accepted limitation shared with :class:`HoughLineRigidFit`: on device
    geometry with several closely-spaced parallel channel walls,
    downsampling can blur two walls into one feature at a coarse level,
    risking a wrong-by-one-period coarse seed.

    Attributes:
        n_iterations: Maximum ECC iterations, applied at every pyramid level.
        epsilon: ECC convergence threshold, applied at every pyramid level.
        min_gradient_std: Minimum standard deviation of the gradient-magnitude
            image required to consider a frame to have detectable signal;
            below this, the frame is treated as blank/textureless.
        min_confidence: Minimum acceptable ``cv2.findTransformECC`` final
            correlation coefficient; below this, the fit is rejected as
            unconfident rather than returned. Empirically (randomized
            translation/rotation within this class's happy-path envelope,
            on structured synthetic data), correctly-converged fits cluster
            tightly around ``0.98``-``0.99``, while silently-wrong fits
            (converged to the wrong local optimum) top out around ``0.87``
            -- ``0.9`` cleanly separates the two with margin on both sides.
        max_pyramid_levels: Hard cap on the number of coarse-to-fine levels
            (see :func:`_build_gray_pyramid`); the finest level is always
            the full-resolution frame.
        min_pyramid_size: Minimum shorter-side length, in pixels, a level
            must have to be included; smaller frames simply get fewer
            levels (possibly just one, i.e. today's single-resolution
            behavior).
    """

    def __init__(
        self,
        n_iterations: int = 200,
        epsilon: float = 1e-6,
        min_gradient_std: float = 1e-3,
        min_confidence: float = 0.9,
        max_pyramid_levels: int = 4,
        min_pyramid_size: int = 128,
    ):
        self.n_iterations = n_iterations
        self.epsilon = epsilon
        self.min_gradient_std = min_gradient_std
        self.min_confidence = min_confidence
        self.max_pyramid_levels = max_pyramid_levels
        self.min_pyramid_size = min_pyramid_size

    def estimate(self, reference: np.ndarray, frame: np.ndarray) -> FrameTransform:
        """See :meth:`RegistrationMethod.estimate`."""
        if reference.shape[:2] != frame.shape[:2]:
            raise RegistrationError(
                "GradientECC: reference and frame must have the same (H, W) "
                f"shape, got {reference.shape[:2]} and {frame.shape[:2]}."
            )
        ref_gray = _to_grayscale_f32(reference)
        frm_gray = _to_grayscale_f32(frame)
        ref_grad = _gradient_magnitude(ref_gray)
        frm_grad = _gradient_magnitude(frm_gray)

        if (
            not np.isfinite(ref_grad).all()
            or not np.isfinite(frm_grad).all()
            or float(ref_grad.std()) < self.min_gradient_std
            or float(frm_grad.std()) < self.min_gradient_std
        ):
            raise RegistrationError(
                "GradientECC: no detectable gradient signal in reference or "
                "frame (blank/textureless or non-finite input)."
            )

        ref_pyr = _build_gray_pyramid(
            ref_gray, self.max_pyramid_levels, self.min_pyramid_size
        )
        frm_pyr = _build_gray_pyramid(
            frm_gray, self.max_pyramid_levels, self.min_pyramid_size
        )
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.n_iterations,
            self.epsilon,
        )

        warp_matrix: np.ndarray = np.eye(2, 3, dtype=np.float32)
        cc: float = 0.0
        for level_idx in range(len(ref_pyr) - 1, -1, -1):
            level_ref_grad = _gradient_magnitude(ref_pyr[level_idx])
            level_frm_grad = _gradient_magnitude(frm_pyr[level_idx])
            level_ref_norm = cv2.normalize(  # type: ignore[call-overload]
                level_ref_grad, None, 0, 1, cv2.NORM_MINMAX
            )
            level_frm_norm = cv2.normalize(  # type: ignore[call-overload]
                level_frm_grad, None, 0, 1, cv2.NORM_MINMAX
            )

            if level_idx == len(ref_pyr) - 1:
                # Coarsest level: seed the ECC warp guess with a coarse
                # translation-only phase-correlation estimate (on this
                # level's gradient images) instead of the identity
                # transform. A failure here is non-fatal (falls back to
                # an identity-seeded warp).
                dx0, dy0 = 0.0, 0.0
                try:
                    shift, _error, _phasediff = phase_cross_correlation(
                        level_ref_grad, level_frm_grad, upsample_factor=1
                    )
                    dy_coarse, dx_coarse = shift
                    dx0, dy0 = float(-dx_coarse), float(-dy_coarse)
                except Exception:  # pre-pass is best-effort only; any
                    # failure here simply falls back to an identity-seeded
                    # warp below.
                    dx0, dy0 = 0.0, 0.0
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                warp_matrix[0, 2] = dx0
                warp_matrix[1, 2] = dy0
            else:
                # Finer level: seed from the previous (coarser) level's
                # converged warp. Rotation is scale-invariant and carries
                # over unchanged; translation is scaled by the exact
                # per-axis ratio between the two levels' shapes (not a
                # hardcoded x2 -- cv2.pyrDown halves via (w+1)//2, so real
                # dimensions aren't guaranteed evenly divisible at every
                # level).
                prev_h, prev_w = ref_pyr[level_idx + 1].shape[:2]
                cur_h, cur_w = ref_pyr[level_idx].shape[:2]
                warp_matrix[0, 2] *= cur_w / prev_w
                warp_matrix[1, 2] *= cur_h / prev_h

            try:
                cc, warp_matrix = cv2.findTransformECC(
                    level_ref_norm,
                    level_frm_norm,
                    warp_matrix,
                    cv2.MOTION_EUCLIDEAN,
                    criteria,
                )
            except cv2.error as exc:
                raise RegistrationError(
                    "GradientECC: cv2.findTransformECC failed to converge "
                    f"at pyramid level {level_idx} (shape "
                    f"{level_ref_norm.shape}): {exc}"
                ) from exc

            if not np.isfinite(cc) or not np.isfinite(warp_matrix).all():
                raise RegistrationError(
                    "GradientECC: cv2.findTransformECC produced a "
                    f"non-finite result at pyramid level {level_idx} "
                    f"(shape {level_ref_norm.shape})."
                )

        if cc < self.min_confidence:
            raise RegistrationError(
                f"GradientECC: final correlation coefficient {cc:.3f} is "
                f"below min_confidence={self.min_confidence}; rejecting a "
                "low-confidence fit rather than returning it."
            )

        h, w = ref_pyr[0].shape
        return _decompose_similarity(warp_matrix, (w / 2.0, h / 2.0))
