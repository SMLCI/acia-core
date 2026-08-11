"""Best-effort calibration reader for OME-TIFF / ImageJ-hyperstack metadata.

This is the inverse of :mod:`acia.segm.tiff_export`'s calibration tags: it reads
back the ``PhysicalSizeX``/``TimeIncrement`` (OME) or ``unit``/``finterval``
(ImageJ) tags that module writes, so acia's own OME/ImageJ TIFF exports round
-trip through it, and any scope-native OME-TIFF or ImageJ hyperstack benefits
too. Reads headers only -- never the pixel array. Missing calibration is
simply omitted, never fabricated (same discipline as the writer side).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import fsspec
import numpy as np
import tifffile

from acia import Q_
from acia.config import resolve_storage_options


@dataclass(frozen=True)
class TiffCalibration:
    """Best-effort physical calibration read from a TIFF's own metadata."""

    pixel_size: object | None  # pint Quantity (length/px) or None
    frame_interval: object | None  # pint Quantity (time) or None
    timepoints: object | None  # pint Quantity array (time, per-frame) or None
    source: str | None  # "ome" | "imagej" | None (where the values came from)


_NO_CALIBRATION = TiffCalibration(
    pixel_size=None, frame_interval=None, timepoints=None, source=None
)


def read_tiff_calibration(
    path: str | os.PathLike, storage_options: dict | None = None
) -> TiffCalibration:
    """Read pixel size / frame timing from a TIFF's OME-XML or ImageJ metadata.

    Reads only the file's headers/metadata (no pixel array). OME-XML is tried
    first; if the file is not an OME-TIFF (or nothing usable is found there),
    ImageJ hyperstack tags are tried next. If neither is present, every field
    is ``None``.

    Args:
        path: Path to the ``.tif``/``.tiff`` file. May be a plain local path or
            any fsspec-supported URL (e.g. ``smb://``).
        storage_options: extra fsspec storage options (e.g. credentials),
            resolved the same way as :class:`~acia.segm.local.LocalSequenceSource`.

    Returns:
        TiffCalibration: best-effort calibration; missing values are ``None``.
    """
    path = os.fspath(path)
    opts = resolve_storage_options(path, storage_options)
    with fsspec.open(path, mode="rb", **opts) as f, tifffile.TiffFile(f) as tf:
        if tf.is_ome:
            cal = _read_ome_calibration(tf)
            if cal is not None:
                return cal
        if tf.is_imagej:
            cal = _read_imagej_calibration(tf)
            if cal is not None:
                return cal
    return _NO_CALIBRATION


def _to_quantity(value, unit: str | None):
    """Best-effort ``Q_(value, unit)``; ``None`` if the value/unit is unusable."""
    if value is None or unit is None:
        return None
    try:
        return Q_(float(value), unit)
    except Exception:  # noqa: BLE001 - calibration is best-effort, never fatal
        return None


def _first(value):
    """OME dicts collapse repeated elements to a bare dict; multiple ones to a
    list. Normalize both to "the first entry" for the single-image case."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _read_ome_calibration(tf) -> TiffCalibration | None:
    """Parse OME-XML for ``PhysicalSizeX``/``TimeIncrement`` (+ per-plane ``DeltaT``)."""
    try:
        meta = tifffile.xml2dict(tf.ome_metadata)
        image = _first(meta["OME"]["Image"])
        pixels = image["Pixels"]
    except Exception:  # noqa: BLE001 - unparsable/unexpected OME-XML shape
        return None

    pixel_size = _to_quantity(
        pixels.get("PhysicalSizeX"), pixels.get("PhysicalSizeXUnit")
    )
    frame_interval = _to_quantity(
        pixels.get("TimeIncrement"), pixels.get("TimeIncrementUnit")
    )
    timepoints = _read_ome_plane_timepoints(pixels)

    if pixel_size is None and frame_interval is None and timepoints is None:
        return None

    return TiffCalibration(
        pixel_size=pixel_size,
        frame_interval=frame_interval,
        timepoints=timepoints,
        source="ome",
    )


def _read_ome_plane_timepoints(pixels):
    """Best-effort per-frame ``DeltaT`` array from ``<Plane>`` elements, else ``None``."""
    planes = pixels.get("Plane")
    if not planes:
        return None
    if isinstance(planes, dict):
        planes = [planes]

    size_t = pixels.get("SizeT")
    by_frame: dict[int, float] = {}
    unit = None
    for plane in planes:
        delta_t = plane.get("DeltaT")
        the_t = plane.get("TheT")
        if delta_t is None or the_t is None:
            continue
        # channel/z planes repeat the same T -- keep the first DeltaT seen per frame
        by_frame.setdefault(int(the_t), float(delta_t))
        unit = unit or plane.get("DeltaTUnit")

    if not by_frame or unit is None:
        return None
    n = int(size_t) if size_t is not None else max(by_frame) + 1
    if len(by_frame) != n or any(t not in by_frame for t in range(n)):
        return None  # incomplete -- don't fabricate the missing frames

    try:
        return Q_(np.array([by_frame[t] for t in range(n)]), unit)
    except Exception:  # noqa: BLE001 - calibration is best-effort, never fatal
        return None


def _read_imagej_calibration(tf) -> TiffCalibration | None:
    """Parse ImageJ hyperstack ``unit``/``finterval`` + resolution tags."""
    ij_meta = tf.imagej_metadata or {}
    unit = ij_meta.get("unit")

    pixel_size = None
    try:
        tags = tf.pages[0].tags
        num, den = tags["XResolution"].value
        if num > 0 and den > 0:
            pixels_per_unit = num / den
            if pixels_per_unit > 0:
                pixel_size = _to_quantity(1.0 / pixels_per_unit, unit)
    except Exception:  # noqa: BLE001 - resolution tag is optional/best-effort
        pass

    frame_interval = _to_quantity(ij_meta.get("finterval"), "second")

    if pixel_size is None and frame_interval is None:
        return None

    return TiffCalibration(
        pixel_size=pixel_size,
        frame_interval=frame_interval,
        timepoints=None,
        source="imagej",
    )
