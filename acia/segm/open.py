"""Format-agnostic lazy opener for multi-position acquisitions.

:func:`open_sequence` dispatches by filename suffix to a per-format backend and
returns a :class:`SequenceFile` — a lazy handle that exposes unified metadata and
a per-position time series without loading pixels. A "position" is the ND2 ``P``
axis and the CZI ``S`` (scene) axis, unified here so the notebook and dashboard
never branch on format.

A path that is a *directory* is opened as a folder of per-timepoint TIFFs (one
file per frame); a folder whose immediate subfolders hold the TIFFs exposes one
position per subfolder, so folder trees, ND2 and CZI all reach the notebook
through the same :meth:`SequenceFile.position` surface.

Reading :attr:`SequenceFile.metadata` / :attr:`SequenceFile.positions` performs
no pixel reads for ND2/CZI; :meth:`SequenceFile.thumbnail` reads exactly one
(downscaled) frame. (The TIFF backends read frame 0 to determine its shape — a
TIFF stack or a single per-timepoint file is small, unlike the hundred-gigabyte
ND2/CZI files this module is built for.)
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


def _scalar_frame_interval(source):
    """Derive the scalar time-between-frames from a source's timepoints, if any.

    There is no public per-scalar ``frame_interval`` on ``ImageSequenceSource``
    (only ``timepoints``, the canonical per-frame calibration), so this mirrors
    :func:`acia.segm.tiff_export._frame_interval_seconds`, returning a pint
    ``Quantity`` instead of a bare seconds float.
    """
    tps = source.timepoints
    if tps is None or len(tps) < 2:
        return None
    return tps[1] - tps[0]


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
        pattern=None,
    ) -> None:
        self.path = str(path)
        self.format = fmt
        self._pixel_size = pixel_size
        self._frame_interval = frame_interval
        self._pattern = pattern
        self._pos_cache: dict[int, object] = {}
        self._meta: SequenceMetadata | None = None
        self._positions: list[PositionInfo] | None = None
        self._layout: tuple[list[str], bool] | None = None

    # --- folder layout (tiff_folder only) --------------------------------------

    def _folders(self) -> tuple[list[str], bool]:
        """Resolve (once) the folder's position folders and whether it is nested."""
        if self._layout is None:
            from acia.segm.folder_source import resolve_layout

            self._layout = resolve_layout(self.path, pattern=self._pattern)
        return self._layout

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
        if self.format == "tiff_folder":
            from acia.segm.folder_source import FolderSequenceSource

            folders, _ = self._folders()
            return FolderSequenceSource(
                folders[index],
                pattern=self._pattern,
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
        """Number of positions (ND2 ``P`` / CZI ``S`` / folder subfolders / 1)."""
        if self.format == "tiff_folder":
            # answerable from the directory listing alone -- don't force the
            # metadata read (which decodes frame 0) just to count positions
            return len(self._folders()[0])
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
        if self.format in ("tiff", "tiff_folder"):
            # one small read: a stack's frame 0, or one per-timepoint file. `size_t`
            # of a folder is a directory listing, so T stays decode-free.
            frame0 = probe.get_frame(0).raw
            h, w = int(frame0.shape[0]), int(frame0.shape[1])
            c = int(frame0.shape[2]) if frame0.ndim == 3 else 1
            sizes = {"T": int(probe.size_t), "Y": h, "X": w, "C": c}
            num_positions = 1 if self.format == "tiff" else len(self._folders()[0])
            return SequenceMetadata(
                sizes=sizes,
                pixel_size=probe.pixel_size,
                frame_interval=_scalar_frame_interval(probe),
                channels=[f"ch{i}" for i in range(c)],
                dtype=str(frame0.dtype),
                num_positions=num_positions,
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
        if self.format == "tiff_folder":
            folders, nested = self._folders()
            if not nested:  # the folder is the movie, not a named position
                return {}
            return {i: os.path.basename(f.rstrip("/")) for i, f in enumerate(folders)}
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


def _is_directory(text: str) -> bool:
    """Whether ``text`` points at a directory (locally or on an fsspec remote).

    Never raises: a probe that cannot answer (unreachable host, missing
    credentials, unknown protocol) means "not a directory", so the caller still
    reports the plain "no reader for this suffix" ``ValueError`` rather than a
    connection error from a filesystem the user never asked to open.
    """
    if "://" not in text:
        return os.path.isdir(text)

    import fsspec

    from acia.config import resolve_storage_options

    try:
        fs, root = fsspec.core.url_to_fs(text, **resolve_storage_options(text))
        return bool(fs.isdir(root))
    except Exception:  # noqa: BLE001 - any probe failure means "not a directory"
        return False


def open_sequence(
    path, *, pixel_size=None, frame_interval=None, pattern=None
) -> SequenceFile:
    """Open an acquisition by filename suffix (ND2/CZI/TIFF) or as a folder.

    A path that *is a directory* is opened as a folder of per-timepoint TIFFs --
    one file per frame, or one subfolder per position (see
    :func:`acia.segm.folder_source.resolve_layout`). ND2/CZI dispatch purely on
    suffix and stay strictly IO-free, so opening a hundred-gigabyte acquisition
    costs nothing until metadata or a position is touched; only an unknown or
    TIFF-like name is probed with a stat.

    Returns a lazy :class:`SequenceFile`. User ``pixel_size``/``frame_interval``
    override any file metadata and propagate to every per-position source.

    Args:
        path: file to open, or a directory of per-timepoint TIFFs.
        pixel_size: physical pixel size overriding the file's own metadata.
        frame_interval: scalar time between frames, likewise overriding.
        pattern: frame-filename glob, folders only (``None`` -> ``.tif``/``.tiff``).

    Raises:
        ValueError: If the suffix maps to no supported reader and the path is not
            a directory, or if ``pattern`` is given for a non-folder path.
    """
    text = str(path)
    suffix = text.rsplit(".", 1)[-1].lower() if "." in os.path.basename(text) else ""
    fmt = _SUFFIX_FORMAT.get(suffix)
    if fmt in (None, "tiff") and _is_directory(text):
        # No suffix match, or a *.tif/*.tiff name that is actually a directory --
        # per-timepoint folders are routinely named like the stack they replace
        # (pos001_roi002.tiff/). ND2/CZI never reach this probe, so opening a
        # hundred-gigabyte acquisition stays strictly IO-free.
        fmt = "tiff_folder"
    if fmt is None:
        supported = ", ".join(sorted(set(_SUFFIX_FORMAT)))
        raise ValueError(
            f"open_sequence: no reader for '.{suffix}' in {text!r} "
            f"(supported suffixes: {supported}; a directory is read as a folder "
            "of per-timepoint TIFFs)"
        )
    if pattern is not None and fmt != "tiff_folder":
        raise ValueError(
            f"open_sequence: pattern={pattern!r} only applies to a folder of "
            f"per-timepoint TIFFs, but {text!r} is a {fmt} file"
        )
    return SequenceFile(
        text,
        fmt,
        pixel_size=pixel_size,
        frame_interval=frame_interval,
        pattern=pattern,
    )
