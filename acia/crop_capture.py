"""Persist manual crops as reloadable specs plus training-data captures.

This module supports the "capture-as-you-go flywheel": each capture saves the
full (uncropped) source frame as a normalized 8-bit grayscale PNG together with
a sidecar ``*.json`` holding a :class:`~acia.base.RotatedCropSpec` as an
oriented (rotated) box label, provenance, and the image shape. Captures are
auto-enumerated into a dataset directory (``0000.png``/``0000.json``,
``0001.*``, ...). A loader reconstructs the :class:`RotatedCropSpec` so a
parameterized/batch run can re-crop without any widget.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from acia.base import ImageSequenceSource, RotatedCropSpec

__all__ = ["save_crop_capture", "load_crop_spec"]


def _normalize_uint8(
    frame: np.ndarray,
    channel: int | None = None,
    clip_percentiles: tuple[float, float] | None = None,
) -> np.ndarray:
    """Reduce a frame to a 2D 8-bit grayscale array for PNG export.

    A multi-channel ``(H, W, C)`` frame is reduced to a single channel
    (``channel``, defaulting to ``0``). ``(H, W)`` and ``(H, W, 1)`` frames are
    treated as grayscale. An optional percentile clip is applied before a
    per-image min-max scaling to ``uint8``. A flat frame (``max == min``) yields
    all zeros (no divide-by-zero). A ``uint8`` frame with no clip requested is
    passed through unchanged (after channel reduction).

    Args:
        frame: Source frame as ``(H, W)``, ``(H, W, 1)`` or ``(H, W, C)``.
        channel: Channel index to select when the frame is multi-channel.
            Defaults to ``0`` when ``None``.
        clip_percentiles: Optional ``(low, high)`` percentiles in ``[0, 100]``.
            When given, intensities are clipped to those percentiles before
            scaling.

    Returns:
        np.ndarray: A 2D ``uint8`` array suitable for a grayscale PNG.
    """
    arr = np.asarray(frame)

    # Reduce to 2D grayscale.
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr[..., 0 if channel is None else channel]

    # uint8 pass-through when no clip is requested.
    if arr.dtype == np.uint8 and clip_percentiles is None:
        return np.ascontiguousarray(arr)

    arr = arr.astype(np.float64)

    if not np.isfinite(arr).all():
        raise ValueError(
            "Frame contains non-finite values (NaN/inf); cannot normalize to uint8."
        )

    if clip_percentiles is not None:
        low, high = np.percentile(arr, clip_percentiles)
        arr = np.clip(arr, low, high)

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    if max_val > min_val:
        scaled = (arr - min_val) / (max_val - min_val) * 255.0
        return np.rint(scaled).astype(np.uint8)

    return np.zeros(arr.shape, dtype=np.uint8)


def _next_index(dataset_dir: Path) -> int:
    """Return the next free zero-padded capture index in ``dataset_dir``.

    Scans existing ``NNNN.json`` files and returns ``max + 1`` (or ``0`` when
    none exist).

    Args:
        dataset_dir: Directory that holds the dataset captures.

    Returns:
        int: The next free index.
    """
    indices: list[int] = []
    for json_path in dataset_dir.glob("*.json"):
        stem = json_path.stem
        if stem.isdigit():
            indices.append(int(stem))
    return max(indices) + 1 if indices else 0


def save_crop_capture(
    source: ImageSequenceSource,
    spec: RotatedCropSpec,
    dataset_dir: str | Path,
    *,
    frame: int = 0,
    channel: int | None = None,
    clip_percentiles: tuple[float, float] | None = None,
    source_ref: str | None = None,
) -> dict[str, object]:
    """Save a full-frame training image plus a rotated-box crop spec.

    The full (uncropped) source frame is rendered, normalized to an 8-bit
    grayscale PNG, and written next to a JSON sidecar describing the crop as an
    oriented (rotated) box label with provenance and image shape. Files are
    auto-enumerated with a 4-digit zero-padded index.

    Args:
        source: The image sequence source to render the full frame from.
        spec: The rotated crop specification to persist as the label.
        dataset_dir: Directory the capture is written to (created if missing).
        frame: Index of the frame to render. Defaults to ``0``.
        channel: Channel to select for grayscale rendering when the frame is
            multi-channel. Defaults to channel ``0`` when ``None``.
        clip_percentiles: Optional ``(low, high)`` percentiles for clipping
            before normalization.
        source_ref: Explicit provenance string. When ``None``, provenance is
            auto-detected from ``source.filename`` then ``source.imageId``;
            otherwise stored as ``null``.

    Returns:
        dict: ``{"index": int, "image": Path, "json": Path}`` for the capture.
    """
    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)

    idx = _next_index(dataset_path)
    stem = f"{idx:04d}"
    image_path = dataset_path / f"{stem}.png"
    json_path = dataset_path / f"{stem}.json"

    full_frame = np.asarray(source.get_frame(frame).raw)
    height, width = int(full_frame.shape[0]), int(full_frame.shape[1])

    normalized = _normalize_uint8(
        full_frame, channel=channel, clip_percentiles=clip_percentiles
    )
    if not cv2.imwrite(str(image_path), normalized):
        raise OSError(f"cv2.imwrite failed to write {image_path}")

    if source_ref is not None:
        provenance: str | None = str(source_ref)
    else:
        origin = getattr(source, "filename", None)
        if origin is None:
            origin = getattr(source, "imageId", None)
        provenance = str(origin) if origin is not None else None

    metadata = {
        "crop": spec.to_dict(),
        "box_type": "rotated",
        "source": provenance,
        "frame": frame,
        "image": f"{stem}.png",
        "image_shape": [height, width],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f)

    return {"index": idx, "image": image_path, "json": json_path}


def load_crop_spec(json_path: str | Path) -> RotatedCropSpec:
    """Reconstruct a :class:`RotatedCropSpec` from a capture's JSON sidecar.

    Args:
        json_path: Path to a ``*.json`` capture sidecar.

    Returns:
        RotatedCropSpec: The reconstructed crop specification.

    Raises:
        FileNotFoundError: If ``json_path`` does not exist.
        KeyError: If the JSON has no ``"crop"`` entry.
    """
    with Path(json_path).open(encoding="utf-8") as f:
        data = json.load(f)
    return RotatedCropSpec.from_dict(data["crop"])
