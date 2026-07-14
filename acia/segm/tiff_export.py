"""Optional lazy TIFF export of a (cropped) image sequence.

:func:`save_tiff_stack` streams any :class:`~acia.base.ImageSequenceSource`
(e.g. a crop returned by :func:`acia.selection.load_selection`) to a TIFF stack
**one frame at a time**, so peak memory stays at a single frame regardless of
sequence length. Physical calibration (pixel size, frame interval) is written into
the TIFF resolution + ImageJ metadata so downstream tools keep µm/px and timing.

This is the *optional* export leg of the curation workflow — the manifest is the
primary output, not a materialized TIFF.
"""

from __future__ import annotations

import os

import numpy as np
import tifffile


def save_tiff_stack(
    source,
    path: str | os.PathLike,
    *,
    imagej: bool = True,
    dtype=None,
) -> str:
    """Write ``source`` to a TIFF stack lazily (one frame at a time).

    Args:
        source: A lazy ``ImageSequenceSource``; iterated frame-by-frame.
        path: Output ``.tif`` path (parent dirs are created).
        imagej: Write an ImageJ hyperstack (default) — matches how
            ``acia-workflows`` consumes TIFFs. Calibration goes into ImageJ
            metadata + TIFF resolution tags.
        dtype: Optional numpy dtype to cast each frame to (default: keep source
            dtype).

    Returns:
        The written path.
    """
    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    n = int(source.size_t)
    if n == 0:
        raise ValueError("Cannot export an empty source (size_t == 0).")

    resolution, resunit, ij_meta = _calibration_tags(source)

    # Peek frame 0 to fix the per-frame shape/axes without holding the whole stack.
    first = _prepare_frame(source.get_frame(0).raw, dtype)
    if first.ndim == 2:  # (H, W)
        axes = "TYX"
    else:  # (C, H, W) — channel-first for an ImageJ-canonical hyperstack
        axes = "TCYX"
    shape = (n, *first.shape)
    est_bytes = n * first.nbytes

    def _frames():
        yield first
        for i in range(1, n):
            yield _prepare_frame(source.get_frame(i).raw, dtype)

    metadata = {**ij_meta, "axes": axes} if imagej else None

    if imagej:
        writer = tifffile.TiffWriter(path, imagej=True)
    else:
        writer = tifffile.TiffWriter(path, bigtiff=est_bytes > 3_900_000_000)

    with writer as tw:
        # A generator + explicit shape lets tifffile pull one frame at a time,
        # so peak memory stays at a single frame regardless of n.
        tw.write(
            _frames(),
            shape=shape,
            dtype=first.dtype,
            metadata=metadata,
            resolution=resolution,
            resolutionunit=resunit,
        )
    return path


def _prepare_frame(raw: np.ndarray, dtype) -> np.ndarray:
    """Return a ``(H, W)`` (single-channel) or ``(C, H, W)`` frame, optionally cast."""
    frame = np.asarray(raw)
    if frame.ndim == 3:
        if frame.shape[2] == 1:
            frame = frame[..., 0]  # squeeze singleton channel -> (H, W)
        else:
            frame = np.moveaxis(frame, -1, 0)  # (H, W, C) -> (C, H, W)
    if dtype is not None:
        frame = frame.astype(dtype)
    return frame


def _calibration_tags(source):
    """Build (resolution, resolutionunit, imagej_metadata) from source calibration.

    Missing calibration is simply omitted (no fabricated tags).
    """
    resolution = None
    resunit = None
    ij_meta: dict = {}

    pixel_size = getattr(source, "pixel_size", None)
    if pixel_size is not None:
        try:
            um = float(pixel_size.to("micrometer").magnitude)
            if um > 0:
                ppu = 1.0 / um  # pixels per micrometer
                resolution = (ppu, ppu)
                resunit = "MICROMETER"
                ij_meta["unit"] = "um"
        except Exception:  # noqa: BLE001 - calibration is best-effort
            pass

    interval_s = _frame_interval_seconds(source)
    if interval_s is not None:
        ij_meta["finterval"] = interval_s
        if interval_s > 0:
            ij_meta["fps"] = 1.0 / interval_s

    return resolution, resunit, ij_meta


def _frame_interval_seconds(source):
    """Derive the seconds-per-frame from the source's timepoints, if any."""
    try:
        tps = source.timepoints
    except Exception:  # noqa: BLE001
        return None
    if tps is None or len(tps) < 2:
        return None
    try:
        return float((tps[1] - tps[0]).to("second").magnitude)
    except Exception:  # noqa: BLE001
        return None
