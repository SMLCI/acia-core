"""Format-agnostic lazy opener for multi-position acquisitions.

:func:`open_sequence` dispatches by filename suffix to a per-format backend and
returns a :class:`SequenceFile` — a lazy handle that exposes unified metadata and
a per-position time series without loading pixels. A "position" is the ND2 ``P``
axis and the CZI ``S`` (scene) axis, unified here so the notebook and dashboard
never branch on format.

Reading :attr:`SequenceFile.metadata` / :attr:`SequenceFile.positions` performs
no pixel reads for ND2/CZI; :meth:`SequenceFile.thumbnail` reads exactly one
(downscaled) frame. (The TIFF backend wraps the existing local stack loader,
which reads frame 0 to determine its shape — TIFF stacks are small, unlike the
hundred-gigabyte ND2/CZI files this module is built for.)
"""

from __future__ import annotations

import contextlib
import io
import os
from dataclasses import dataclass

import numpy as np

from acia.notebook import normalize_to_uint8

# filename suffix -> internal format key
_SUFFIX_FORMAT = {"nd2": "nd2", "czi": "czi", "tif": "tiff", "tiff": "tiff"}


@dataclass(frozen=True)
class SequenceMetadata:
    """File-level metadata, unified across formats. JSON-safe via :meth:`to_dict`."""

    sizes: dict[str, int]
    pixel_size: object | None  # pint Quantity or None
    frame_interval: object | None  # pint Quantity or None
    channels: list[str]
    dtype: str
    num_positions: int
    num_timepoints: int

    def to_dict(self) -> dict:
        """Emit the manifest ``source`` metadata block (plain floats + strings)."""

        def _um(v):
            return None if v is None else float(v.to("micrometer").magnitude)

        def _s(v):
            return None if v is None else float(v.to("second").magnitude)

        return {
            "sizes": dict(self.sizes),
            "pixel_size_um": _um(self.pixel_size),
            "frame_interval_s": _s(self.frame_interval),
            "channels": list(self.channels),
            "dtype": self.dtype,
            "num_positions": self.num_positions,
            "num_timepoints": self.num_timepoints,
        }


@dataclass(frozen=True)
class PositionInfo:
    """Lightweight per-position descriptor for the gallery (no pixel reads)."""

    index: int
    name: str | None = None
    stage_xy: tuple[float, float] | None = None


class SequenceFile:
    """Lazy handle over a multi-position acquisition (format-agnostic).

    Construct via :func:`open_sequence`. Reading metadata/positions performs no
    pixel reads for ND2/CZI. Per-position sources are created on demand and cached
    (one source per position).
    """

    def __init__(
        self,
        path: str | os.PathLike,
        fmt: str,
        *,
        pixel_size=None,
        frame_interval=None,
    ) -> None:
        self.path = str(path)
        self.format = fmt
        self._pixel_size = pixel_size
        self._frame_interval = frame_interval
        self._pos_cache: dict[int, object] = {}
        self._meta: SequenceMetadata | None = None
        self._positions: list[PositionInfo] | None = None

    # --- per-position sources --------------------------------------------------

    def _make_source(self, index: int):
        """Construct the per-position ``ImageSequenceSource`` for ``index``."""
        if self.format == "nd2":
            from acia.segm.nd2_source import ND2SequenceSource

            return ND2SequenceSource(
                self.path,
                position=index,
                pixel_size=self._pixel_size,
                frame_interval=self._frame_interval,
            )
        if self.format == "czi":
            from acia.segm.czi_source import CZISequenceSource

            return CZISequenceSource(
                self.path,
                position=index,
                pixel_size=self._pixel_size,
                frame_interval=self._frame_interval,
            )
        if self.format == "tiff":
            if index != 0:
                raise ValueError(
                    f"TIFF is single-position; position {index} is invalid."
                )
            from acia.segm.local import LocalSequenceSource

            return LocalSequenceSource(
                self.path,
                normalize_image=False,
                pixel_size=self._pixel_size,
                frame_interval=self._frame_interval,
            )
        raise ValueError(f"unknown format {self.format!r}")  # pragma: no cover

    def _probe(self):
        """Return (and cache) the position-0 source used to read metadata."""
        if 0 not in self._pos_cache:
            self._pos_cache[0] = self._make_source(0)
        return self._pos_cache[0]

    def position(self, index: int):
        """Return the per-position lazy ``ImageSequenceSource`` for ``index``."""
        n = self.num_positions
        if not 0 <= index < n:
            raise ValueError(
                f"position {index} out of range for {n} position(s) in {self.path!r}"
            )
        if index not in self._pos_cache:
            self._pos_cache[index] = self._make_source(index)
        return self._pos_cache[index]

    # --- metadata --------------------------------------------------------------

    @property
    def metadata(self) -> SequenceMetadata:
        """Unified file-level metadata (no pixel reads for ND2/CZI)."""
        if self._meta is None:
            self._meta = self._read_metadata()
        return self._meta

    @property
    def num_positions(self) -> int:
        """Number of positions (ND2 ``P`` / CZI ``S`` / 1 for TIFF)."""
        return self.metadata.num_positions

    @property
    def positions(self) -> list[PositionInfo]:
        """Per-position descriptors, lazily built (no pixel reads)."""
        if self._positions is None:
            names = self._position_names()
            self._positions = [
                PositionInfo(index=i, name=names.get(i))
                for i in range(self.num_positions)
            ]
        return self._positions

    def _read_metadata(self) -> SequenceMetadata:
        probe = self._probe()
        if self.format == "tiff":
            frame0 = probe.get_frame(0).raw  # small stack: reading is acceptable
            h, w = int(frame0.shape[0]), int(frame0.shape[1])
            c = int(frame0.shape[2]) if frame0.ndim == 3 else 1
            sizes = {"T": int(probe.size_t), "Y": h, "X": w, "C": c}
            return SequenceMetadata(
                sizes=sizes,
                pixel_size=probe.pixel_size,
                frame_interval=self._frame_interval,
                channels=[f"ch{i}" for i in range(c)],
                dtype=str(frame0.dtype),
                num_positions=1,
                num_timepoints=int(probe.size_t),
            )

        sizes = dict(probe.sizes)
        if self.format == "nd2":
            num_positions = int(sizes.get("P", 1))
        else:  # czi
            num_positions = int(getattr(probe, "n_scenes", sizes.get("S", 1)))
        channels = list(getattr(probe, "channel_names", []) or [])
        if not channels:
            channels = [f"ch{i}" for i in range(int(probe.size_c))]
        return SequenceMetadata(
            sizes=sizes,
            pixel_size=probe.pixel_size,
            frame_interval=self._frame_interval,
            channels=channels,
            dtype=str(getattr(probe, "dtype", "") or ""),
            num_positions=num_positions,
            num_timepoints=int(probe.size_t),
        )

    def _position_names(self) -> dict[int, str | None]:
        if self.format == "czi":
            return dict(getattr(self._probe(), "scene_names", {}) or {})
        return {}

    # --- thumbnails ------------------------------------------------------------

    def thumbnail(
        self, index: int, *, downscale: int = 8, frame: int = 0
    ) -> np.ndarray:
        """A downscaled ``(h, w, 3)`` uint8 preview of one frame (lazy).

        Reads exactly one frame of ``index``, takes the display channel (0),
        min-max normalizes, strides by ``downscale``, and returns an RGB preview.
        """
        raw = self.position(index).get_frame(frame).raw  # (H, W, C) or (H, W)
        plane = raw[..., 0] if raw.ndim == 3 else raw
        small = plane[::downscale, ::downscale]
        u8 = normalize_to_uint8(small)
        return np.stack([u8, u8, u8], axis=-1)

    def thumbnail_png(self, index: int, *, downscale: int = 8, frame: int = 0) -> bytes:
        """PNG-encoded bytes of :meth:`thumbnail` (for the widget's byte channel)."""
        from PIL import Image

        rgb = self.thumbnail(index, downscale=downscale, frame=frame)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return buf.getvalue()

    # --- cleanup ---------------------------------------------------------------

    def close(self) -> None:
        """Release cached per-position readers."""
        for src in self._pos_cache.values():
            close = getattr(src, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
        self._pos_cache.clear()
        self._positions = None

    def __enter__(self) -> SequenceFile:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_sequence(path, *, pixel_size=None, frame_interval=None) -> SequenceFile:
    """Open a multi-position acquisition by filename suffix (ND2/CZI/TIFF).

    Returns a lazy :class:`SequenceFile`; no file is opened until metadata or a
    position is accessed. User ``pixel_size``/``frame_interval`` override any file
    metadata and propagate to every per-position source.

    Raises:
        ValueError: If the suffix maps to no supported reader.
    """
    text = str(path)
    suffix = text.rsplit(".", 1)[-1].lower() if "." in os.path.basename(text) else ""
    fmt = _SUFFIX_FORMAT.get(suffix)
    if fmt is None:
        supported = ", ".join(sorted(set(_SUFFIX_FORMAT)))
        raise ValueError(
            f"open_sequence: no reader for '.{suffix}' in {text!r} "
            f"(supported suffixes: {supported})"
        )
    return SequenceFile(text, fmt, pixel_size=pixel_size, frame_interval=frame_interval)
