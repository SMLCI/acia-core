"""Lazy, memory-bounded image source for Nikon ND2 files.

An ND2 file may contain several stage positions; the ROI workflow processes a
single position, which itself is a time series. :class:`ND2SequenceSource`
selects one position and exposes that position's ``(T, H, W, C)`` time series
through the standard :class:`~acia.base.ImageSequenceSource` contract.

The ND2 file is opened lazily on first access and read **one frame at a time**
via the ``nd2`` dask interface, so peak memory stays at a single frame
regardless of file size (ND2 files may be tens of gigabytes). ``nd2`` is an
optional dependency, imported lazily inside :meth:`ND2SequenceSource._ensure_reader`.
"""

from __future__ import annotations

import numpy as np

from acia import Q_
from acia.base import BaseImage, ImageSequenceSource
from acia.notebook import JupyterVisualizationMixin
from acia.segm.local import LocalImage

_ND2_IMPORT_HINT = "ND2 support requires the optional dependency: pip install acia[nd2]"


class ND2SequenceSource(ImageSequenceSource, JupyterVisualizationMixin):
    """A single ND2 position exposed as a lazy ``(T, H, W, C)`` time series.

    Axis mapping is done by *name* (using the file's ``sizes`` dict), never by a
    fixed axis order. For each frame the source selects ``P == position`` and
    ``T == frame``, moves the channel axis last, and adds a trailing channel axis
    for grayscale planes so every frame is shaped ``(H, W, C)``.

    The reader is opened lazily and only a single frame is materialized at a time
    via ``nd2``'s dask interface; the whole array is never loaded.

    Args:
        path: Path to the ``.nd2`` file.
        position: Index of the stage position to expose. A file with no ``P``
            axis is treated as a single position (only ``0`` is valid).
        pixel_size: Optional pint length per pixel that overrides the file's
            voxel size. Strings like ``"0.5 um"`` are accepted.
        frame_interval: Optional scalar time between frames (pint or string)
            that overrides metadata-derived timing.
        timepoints: Optional explicit per-frame pint ``Quantity`` array that
            overrides metadata-derived timing.
    """

    def __init__(
        self,
        path: str,
        position: int = 0,
        *,
        pixel_size=None,
        frame_interval=None,
        timepoints=None,
    ) -> None:
        self.path = path
        self.position = position
        # user-supplied calibration overrides (metadata is used otherwise)
        self._user_pixel_size = pixel_size
        self._user_frame_interval = frame_interval
        self._user_timepoints = timepoints

        # lazy state, populated by _ensure_reader()
        self._reader = None
        self._sizes: dict[str, int] | None = None
        self._dask = None
        self._axes: list[str] = []

    # --- lazy reader / metadata ------------------------------------------------

    def _ensure_reader(self) -> None:
        """Open the ND2 file once and resolve sizes, dask handle and calibration.

        Opening the file and reading metadata must not materialize pixel data.
        Validates the requested ``position`` and rejects Z-stacks (``Z > 1``).

        Raises:
            ImportError: If the optional ``nd2`` dependency is not installed.
            ValueError: If ``position`` is out of range or a ``Z`` axis with
                size > 1 is present (3D/Z support is deferred).
        """
        if self._reader is not None:
            return

        try:
            import nd2  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - trivial branch
            raise ImportError(_ND2_IMPORT_HINT) from exc

        reader = nd2.ND2File(self.path)
        try:
            sizes = dict(reader.sizes)
            axes = list(sizes.keys())

            # Only known axes are supported. An RGB/sample ("S") or any other
            # unexpected axis would be silently mis-mapped, so reject it loudly.
            unknown = [a for a in axes if a not in ("P", "T", "Z", "C", "Y", "X")]
            if unknown:
                raise ValueError(
                    f"ND2 file {self.path!r} has unsupported axes {unknown}; only "
                    "P/T/Z/C/Y/X are supported (RGB/'S' / extra axes are deferred)."
                )

            # validate position against the P axis (no P axis -> single position 0)
            n_positions = sizes.get("P", 1)
            if not 0 <= self.position < n_positions:
                raise ValueError(
                    f"position {self.position} out of range for {n_positions} "
                    f"position(s) in {self.path!r}"
                )

            # Z-stacks are deferred; size-1 Z is squeezed implicitly via indexing
            if sizes.get("Z", 1) > 1:
                raise ValueError(
                    f"ND2 file {self.path!r} has a Z axis of size {sizes['Z']}; "
                    "3D/Z-stack support is not implemented (Z deferred)."
                )

            dask = reader.to_dask()
            # Axis mapping is by name and assumes to_dask()'s axis order matches
            # the `sizes` key order; verify the shapes agree so a mismatch fails
            # loudly instead of silently reading the wrong plane.
            if tuple(dask.shape) != tuple(sizes.values()):
                raise ValueError(
                    f"ND2 dask shape {tuple(dask.shape)} does not match sizes "
                    f"{tuple(sizes.values())} for {self.path!r} (axis-order mismatch)."
                )

            pixel_size = self._resolve_pixel_size(reader)
            frame_interval = self._resolve_frame_interval(reader)
        except Exception:
            reader.close()
            raise

        # Assign instance state only after everything above succeeded, so a
        # partial failure never leaves a half-open, wedged source.
        self._reader = reader
        self._sizes = sizes
        self._axes = axes
        self._dask = dask
        self._init_calibration(
            frame_interval=frame_interval,
            timepoints=self._user_timepoints,
            pixel_size=pixel_size,
        )

    def _resolve_pixel_size(self, reader):
        """Resolve the pixel size: user override > voxel-size metadata > None."""
        if self._user_pixel_size is not None:
            return self._user_pixel_size
        try:
            voxel = reader.voxel_size()
            x = float(voxel.x)
            if not np.isfinite(x) or x <= 0:
                return None  # missing/zero calibration -> uncalibrated, not 0 µm
            return Q_(x, "micrometer")
        except Exception:  # noqa: BLE001 - metadata is best-effort
            return None

    def _resolve_frame_interval(self, reader):
        """Resolve the frame interval from the user override, else ``None``.

        ND2 per-frame timing varies by file and ``nd2`` version, so v1 does NOT
        read it from metadata: if the user did not pass ``frame_interval``/
        ``timepoints``, timing is left ``None``. (Deriving the interval from ND2
        metadata is a tracked follow-up.)
        """
        return self._user_frame_interval

    @property
    def sizes(self) -> dict[str, int]:
        """The ND2 axis-size mapping (e.g. ``{'P': 2, 'T': 5, ...}``)."""
        self._ensure_reader()
        assert self._sizes is not None
        return self._sizes

    @property
    def pixel_size(self):
        """Pint length per pixel, resolved from the file (or override)."""
        self._ensure_reader()
        return ImageSequenceSource.pixel_size.fget(self)

    @property
    def timepoints(self):
        """Per-frame pint timepoints, or ``None`` if uncalibrated."""
        self._ensure_reader()
        return ImageSequenceSource.timepoints.fget(self)

    @property
    def dtype(self) -> str:
        """Numpy dtype string of the pixel data (from metadata, no pixel read)."""
        self._ensure_reader()
        assert self._reader is not None
        return str(np.dtype(self._reader.dtype))

    @property
    def channel_names(self) -> list[str]:
        """Channel names from ND2 metadata (best-effort; empty if unavailable)."""
        self._ensure_reader()
        assert self._reader is not None
        try:
            channels = self._reader.metadata.channels or []
            return [c.channel.name for c in channels]
        except Exception:  # noqa: BLE001 - metadata is best-effort
            return []

    # --- ImageSequenceSource contract -----------------------------------------

    @property
    def size_t(self) -> int:
        """Number of frames (``T``); 1 if the file has no ``T`` axis."""
        return int(self.sizes.get("T", 1))

    @property
    def size_h(self) -> int:
        """Image height (``Y``)."""
        return int(self.sizes["Y"])

    @property
    def size_w(self) -> int:
        """Image width (``X``)."""
        return int(self.sizes["X"])

    @property
    def size_c(self) -> int:
        """Number of channels (``C``); 1 if the file has no ``C`` axis."""
        return int(self.sizes.get("C", 1))

    @property
    def num_channels(self) -> int:
        """Number of channels (alias of :attr:`size_c`)."""
        return self.size_c

    def get_frame(self, frame: int) -> BaseImage:
        """Read a single ``(H, W, C)`` plane for ``(position, frame)``.

        Only the requested frame is materialized (via the dask interface); the
        whole array is never loaded.

        Args:
            frame: Time index of the frame to read.

        Returns:
            BaseImage: A :class:`~acia.segm.local.LocalImage` wrapping the
            ``(H, W, C)`` plane.
        """
        self._ensure_reader()
        assert self._dask is not None

        # Validate bounds: a negative frame would silently wrap to the last
        # frame, and a file with no T axis would otherwise ignore `frame`.
        if not 0 <= frame < self.size_t:
            raise IndexError(f"frame {frame} out of range for size_t={self.size_t}")

        indexer = self._build_indexer(frame)
        plane = np.asarray(self._dask[indexer])
        plane = self._to_hwc(plane)
        return LocalImage(plane, frame=frame)

    # --- axis-name mapping -----------------------------------------------------

    def _build_indexer(self, frame: int) -> tuple:
        """Build a name-based index tuple selecting one ``(position, frame)`` plane.

        Selects ``P == position`` and ``T == frame`` (when those axes exist),
        squeezes a size-1 ``Z`` axis, and keeps ``Y``, ``X`` and ``C``. The order
        of entries matches ``self._axes`` (the dask array axis order).
        """
        index: list = []
        for axis in self._axes:
            if axis == "P":
                index.append(self.position)
            elif axis == "T":
                index.append(frame)
            elif axis == "Z":
                index.append(0)  # size-1 Z, squeezed
            else:
                # Y, X, C: keep the full axis
                index.append(slice(None))
        return tuple(index)

    def _to_hwc(self, plane: np.ndarray) -> np.ndarray:
        """Arrange a materialized plane to ``(H, W, C)``.

        The plane's remaining axes are the non-indexed axes of ``self._axes`` in
        order (some subset of ``Y``, ``X``, ``C``). Channel is moved last; a
        grayscale plane (no ``C`` axis) gains a trailing channel axis.
        """
        remaining = [a for a in self._axes if a not in ("P", "T", "Z")]
        if "C" in remaining:
            c_pos = remaining.index("C")
            plane = np.moveaxis(plane, c_pos, -1)
        else:
            plane = plane[..., np.newaxis]
        return plane

    # --- cleanup ---------------------------------------------------------------

    def __del__(self) -> None:
        """Best-effort close of the underlying ND2 file handle."""
        try:
            if self._reader is not None:
                self._reader.close()
        except Exception:  # noqa: BLE001 - cleanup must never raise
            pass
