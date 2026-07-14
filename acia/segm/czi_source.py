"""Lazy, memory-bounded image source for Zeiss CZI files.

A CZI file may contain several scenes (stage positions); the ROI workflow
processes a single scene, which itself is a time series. :class:`CZISequenceSource`
selects one scene and exposes that scene's ``(T, H, W, C)`` time series through
the standard :class:`~acia.base.ImageSequenceSource` contract.

The CZI file is opened lazily on first access and read **one plane at a time**
via ``aicspylibczi``'s ``read_image`` (which reads only the addressed subblocks),
so peak memory stays at a single frame regardless of file size (CZI files may be
hundreds of gigabytes). ``aicspylibczi`` is an optional dependency, imported
lazily inside :meth:`CZISequenceSource._ensure_reader`.

This mirrors :mod:`acia.segm.nd2_source`: a "position" is the CZI scene (``S``)
axis, the exact analogue of ND2's ``P``.
"""

from __future__ import annotations

import numpy as np

from acia import Q_
from acia.base import BaseImage, ImageSequenceSource
from acia.notebook import JupyterVisualizationMixin
from acia.segm.local import LocalImage

_CZI_IMPORT_HINT = "CZI support requires the optional dependency: pip install acia[czi]"

# CZI axes understood in v1. A "position" is the scene axis ``S``. Any other axis
# (e.g. block ``B``, phase ``H``, illumination ``I``) would be silently mis-mapped,
# so it is rejected loudly.
_KNOWN_AXES = ("S", "T", "C", "Z", "M", "Y", "X")
_SPATIAL_AXES = ("Y", "X", "C")

# CZI pixel-type name -> numpy dtype string (best-effort; extended as needed).
_CZI_PIXEL_TYPE = {
    "gray8": "uint8",
    "gray16": "uint16",
    "gray32float": "float32",
    "bgr24": "uint8",
    "bgr48": "uint16",
    "bgra32": "uint8",
}


class CZISequenceSource(ImageSequenceSource, JupyterVisualizationMixin):
    """A single CZI scene exposed as a lazy ``(T, H, W, C)`` time series.

    Axis mapping is done by *name* (using the reader's ``dims`` string), never by
    a fixed axis order. For each frame the source selects ``S == position`` and
    ``T == frame``, moves the channel axis last, and adds a trailing channel axis
    for grayscale planes so every frame is shaped ``(H, W, C)``.

    The reader is opened lazily and only a single plane is materialized at a time
    via ``read_image``; the whole scene/file is never loaded.

    Args:
        path: Path to the ``.czi`` file.
        position: Index of the scene to expose. A file with no ``S`` axis is
            treated as a single scene (only ``0`` is valid).
        pixel_size: Optional pint length per pixel that overrides the file's
            scaling metadata. Strings like ``"0.5 um"`` are accepted.
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
        self._dims: str = ""
        self._sizes: dict[str, int] | None = None
        self._scene_names: dict[int, str | None] = {}

    # --- lazy reader / metadata ------------------------------------------------

    def _ensure_reader(self) -> None:
        """Open the CZI file once and resolve dims, sizes and calibration.

        Opening the file and reading metadata must not materialize pixel data.
        Validates the requested ``position`` and rejects mosaic files and
        Z-stacks (``Z > 1``).

        Raises:
            ImportError: If the optional ``aicspylibczi`` dependency is missing.
            ValueError: On unsupported axes, a mosaic, a ``Z`` axis with size > 1,
                a missing ``Y``/``X`` axis, or a ``position`` out of range.
        """
        if self._reader is not None:
            return

        try:
            import aicspylibczi  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - trivial branch
            raise ImportError(_CZI_IMPORT_HINT) from exc

        reader = aicspylibczi.CziFile(self.path)
        try:
            dims = str(reader.dims)
            size = tuple(int(s) for s in reader.size)
            if len(dims) != len(size):
                raise ValueError(
                    f"CZI file {self.path!r} dims {dims!r} and size {size} disagree."
                )
            sizes = dict(zip(dims, size, strict=True))

            unknown = [a for a in dims if a not in _KNOWN_AXES]
            if unknown:
                raise ValueError(
                    f"CZI file {self.path!r} has unsupported axes {unknown}; only "
                    f"{'/'.join(_KNOWN_AXES)} are supported (mosaic/other axes deferred)."
                )

            if "Y" not in sizes or "X" not in sizes:
                raise ValueError(
                    f"CZI file {self.path!r} lacks a Y/X image plane (dims {dims!r})."
                )

            # Mosaic tiling is deferred (stitching not implemented).
            try:
                is_mosaic = bool(reader.is_mosaic())
            except Exception:  # noqa: BLE001 - fall back to the M axis
                is_mosaic = sizes.get("M", 1) > 1
            if is_mosaic or sizes.get("M", 1) > 1:
                raise ValueError(
                    f"CZI file {self.path!r} is a mosaic; mosaic stitching is not "
                    "implemented (deferred)."
                )

            # Z-stacks are deferred; a size-1 Z is squeezed implicitly via indexing.
            if sizes.get("Z", 1) > 1:
                raise ValueError(
                    f"CZI file {self.path!r} has a Z axis of size {sizes['Z']}; "
                    "3D/Z-stack support is not implemented (Z deferred)."
                )

            # validate scene/position (no S axis -> single scene 0)
            n_scenes = sizes.get("S", 1)
            if not 0 <= self.position < n_scenes:
                raise ValueError(
                    f"position {self.position} out of range for {n_scenes} "
                    f"scene(s) in {self.path!r}"
                )

            pixel_size = self._resolve_pixel_size(reader)
            frame_interval = self._resolve_frame_interval(reader)
            scene_names = self._resolve_scene_names(reader)
        except Exception:
            self._safe_close(reader)
            raise

        # Assign instance state only after everything above succeeded, so a
        # partial failure never leaves a half-open, wedged source.
        self._reader = reader
        self._dims = dims
        self._sizes = sizes
        self._scene_names = scene_names
        self._init_calibration(
            frame_interval=frame_interval,
            timepoints=self._user_timepoints,
            pixel_size=pixel_size,
        )

    def _resolve_pixel_size(self, reader):
        """Resolve the pixel size: user override > CZI scaling metadata > None."""
        if self._user_pixel_size is not None:
            return self._user_pixel_size
        try:
            meta = reader.meta
            el = meta.find(".//Scaling/Items/Distance[@Id='X']/Value")
            if el is None or not el.text:
                return None
            x_m = float(el.text)  # CZI scaling is in metres
            if not np.isfinite(x_m) or x_m <= 0:
                return None  # missing/zero calibration -> uncalibrated, not 0 µm
            return Q_(x_m, "meter").to("micrometer")
        except Exception:  # noqa: BLE001 - metadata is best-effort
            return None

    def _resolve_frame_interval(self, reader):
        """Resolve the frame interval from the user override, else ``None``.

        CZI per-frame timing lives in acquisition metadata and varies by file, so
        v1 does NOT read it from metadata: if the user did not pass
        ``frame_interval``/``timepoints``, timing is left ``None``. (Deriving the
        interval from CZI metadata is a tracked follow-up.)
        """
        return self._user_frame_interval

    def _resolve_scene_names(self, reader) -> dict[int, str | None]:
        """Read the scene index -> name mapping from metadata (best-effort)."""
        names: dict[int, str | None] = {}
        try:
            meta = reader.meta
            for scene in meta.findall(".//Scenes/Scene"):
                idx = scene.get("Index")
                if idx is None:
                    continue
                names[int(idx)] = scene.get("Name")
        except Exception:  # noqa: BLE001 - metadata is best-effort
            pass
        return names

    @property
    def sizes(self) -> dict[str, int]:
        """The CZI axis-size mapping (e.g. ``{'S': 107, 'T': 294, ...}``)."""
        self._ensure_reader()
        assert self._sizes is not None
        return self._sizes

    @property
    def n_scenes(self) -> int:
        """Number of scenes (positions) in the file; 1 if there is no ``S`` axis."""
        return int(self.sizes.get("S", 1))

    @property
    def scene_names(self) -> dict[int, str | None]:
        """Mapping of scene index -> name (e.g. ``{0: 'P1', 1: 'P2', ...}``)."""
        self._ensure_reader()
        return self._scene_names

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
        """Numpy dtype string mapped from the CZI pixel type (best-effort)."""
        self._ensure_reader()
        assert self._reader is not None
        try:
            pt = str(self._reader.pixel_type).lower()
            return _CZI_PIXEL_TYPE.get(pt, "")
        except Exception:  # noqa: BLE001 - metadata is best-effort
            return ""

    @property
    def channel_names(self) -> list[str]:
        """Channel names from CZI metadata (best-effort; empty if unavailable)."""
        self._ensure_reader()
        assert self._reader is not None
        try:
            meta = self._reader.meta
            names = [
                c.get("Name") for c in meta.findall(".//Dimensions/Channels/Channel")
            ]
            return [n for n in names if n]
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
        """Read a single ``(H, W, C)`` plane for ``(scene=position, frame)``.

        Only the requested plane is materialized (via ``read_image``); the whole
        scene/file is never loaded.

        Args:
            frame: Time index of the frame to read.

        Returns:
            BaseImage: A :class:`~acia.segm.local.LocalImage` wrapping the
            ``(H, W, C)`` plane.
        """
        self._ensure_reader()
        assert self._reader is not None

        # Validate bounds: a negative frame would silently wrap, and a file with
        # no T axis would otherwise ignore `frame`.
        if not 0 <= frame < self.size_t:
            raise IndexError(f"frame {frame} out of range for size_t={self.size_t}")

        constraints = self._build_constraints(frame)
        data, shape = self._reader.read_image(**constraints)
        plane = self._arrange_plane(np.asarray(data), shape)
        return LocalImage(plane, frame=frame)

    # --- axis-name mapping -----------------------------------------------------

    def _build_constraints(self, frame: int) -> dict[str, int]:
        """Build the ``read_image`` constraints selecting one plane.

        Constrains ``S == position`` and ``T == frame`` and squeezes size-1
        ``Z``/``M`` axes, leaving ``C``/``Y``/``X`` unconstrained. Only axes that
        exist in the file are constrained.
        """
        constraints: dict[str, int] = {}
        for axis in self._dims:
            if axis == "S":
                constraints["S"] = self.position
            elif axis == "T":
                constraints["T"] = frame
            elif axis in ("Z", "M"):
                constraints[axis] = 0  # size-1, squeezed
        return constraints

    def _arrange_plane(self, data: np.ndarray, shape) -> np.ndarray:
        """Arrange a ``read_image`` result to ``(H, W, C)``.

        ``shape`` is ``read_image``'s descriptor: a list of ``(dim, size)`` in the
        array's axis order. Non-spatial axes (``S``/``T``/``Z``/``M``) are squeezed
        by taking index 0; the channel axis is moved last; a grayscale plane (no
        ``C`` axis) gains a trailing channel axis.
        """
        axes = [dim for dim, _ in shape]
        if len(axes) != data.ndim:
            raise ValueError(
                f"read_image returned {data.ndim}D data but shape descriptor "
                f"{shape!r} lists {len(axes)} axes for {self.path!r}."
            )
        index = tuple(slice(None) if a in _SPATIAL_AXES else 0 for a in axes)
        plane = data[index]

        remaining = [a for a in axes if a in _SPATIAL_AXES]
        if "C" in remaining:
            plane = np.moveaxis(plane, remaining.index("C"), -1)
        else:
            plane = plane[..., np.newaxis]
        return plane

    # --- cleanup ---------------------------------------------------------------

    @staticmethod
    def _safe_close(reader) -> None:
        """Close the reader if it exposes a ``close`` (aicspylibczi may not)."""
        try:
            close = getattr(reader, "close", None)
            if callable(close):
                close()
        except Exception:  # noqa: BLE001 - cleanup must never raise
            pass

    def __del__(self) -> None:
        """Best-effort close of the underlying CZI file handle."""
        if self._reader is not None:
            self._safe_close(self._reader)
