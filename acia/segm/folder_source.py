"""A folder of per-timepoint TIFFs exposed as one lazy time series.

Many acquisition setups (µManager, LabVIEW rigs, "save each frame" exports) write
**one folder per movie and one TIFF file per timepoint** rather than a single
stack. :class:`FolderSequenceSource` reads that layout with the frame index *being*
the file index: file ``i`` (natural-sorted) is frame ``i``, decoded on demand, so
peak memory stays at roughly one frame no matter how long the movie is.

:func:`resolve_layout` decides, from directory listings alone, whether a folder is
a single movie (it holds the TIFFs itself) or a set of **positions** (its immediate
subfolders hold them) -- the same position axis that ND2 ``P`` and CZI ``S``
expose, so :func:`acia.segm.open.open_sequence` can serve all three identically.
"""

from __future__ import annotations

import fnmatch
import os
import re
import warnings

import fsspec
import numpy as np
import tifffile

from acia.base import BaseImage, ImageSequenceSource
from acia.config import resolve_storage_options
from acia.notebook import JupyterVisualizationMixin
from acia.segm.local import LocalImage, prepare_image
from acia.segm.tiff_metadata import read_tiff_calibration

_TIFF_SUFFIXES = (".tif", ".tiff")

# a 3-D plane whose trailing axis is longer than this is not plausibly a channel
# axis -- see _to_hwc, where it disambiguates (C, H, W) from a multi-page file
_MAX_CHANNELS = 4


def natural_key(name: str) -> tuple:
    """Digit-aware sort key so ``t2.tif`` sorts before ``t10.tif``.

    Lexicographic ordering of per-timepoint filenames is a *silent* frame-order
    corruption (it yields plausible-looking but wrong tracking), so ordering is
    always done through this key. ``re.split`` on a captured digit group always
    alternates text/number, so the tuples compare element-wise without ever
    mixing ``int`` and ``str`` at the same position.

    ``img_1.tif`` and ``img_01.tif`` produce the same numeric key, so the raw
    basename is appended as a tie-break: without it their relative order would
    come from the filesystem's listing order and could differ between machines.
    """
    basename = os.path.basename(name.rstrip("/")).lower()
    numeric = tuple(
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", basename)
    )
    return (numeric, basename)


def _pattern_text(pattern: str | None) -> str:
    """Human-readable description of the active file filter (for messages)."""
    return repr(pattern) if pattern else "'*.tif' / '*.tiff'"


def _matches(path: str, pattern: str | None) -> bool:
    """Whether a listed entry counts as a frame file.

    Matching is case-insensitive (``fsspec``'s own ``glob`` is not, and would need
    a second pass for ``.TIF``). Dot-files are always skipped, which also drops
    macOS AppleDouble siblings (``._img_001.tif``) that would otherwise be read as
    extra frames. A TIFF suffix is required even when ``pattern`` is given, so a
    broad pattern (``"*"``, ``"img_*"``) cannot pull a ``metadata.txt`` into the
    frame list and fail deep inside a decode later.
    """
    name = os.path.basename(path.rstrip("/")).lower()
    if name.startswith(".") or not name.endswith(_TIFF_SUFFIXES):
        return False
    return pattern is None or fnmatch.fnmatch(name, pattern.lower())


def _listing(
    folder: str, storage_options: dict | None = None
) -> tuple[list[str], list[str]]:
    """One ``ls`` of ``folder`` -> ``(file_locations, subdir_locations)``.

    A single call yields both files and subdirectories, so layout detection, the
    frame list and the position list never cost more round trips than necessary
    (this matters on ``smb://``). Remote entries keep their protocol/host so that
    per-file credential resolution still works at read time, mirroring
    :func:`acia.segm.local.list_sequence_sources`.
    """
    opts = resolve_storage_options(folder, storage_options)
    fs, root = fsspec.core.url_to_fs(folder, **opts)
    is_local = "file" in fs.protocol or "local" in fs.protocol

    files: list[str] = []
    subdirs: list[str] = []
    for entry in fs.ls(root, detail=True):
        name = entry["name"]
        kind = entry.get("type")
        if kind not in ("directory", "file"):
            # a symlink lists as type "other" -- resolve it, or a symlinked
            # position folder (common on lab storage) is silently dropped
            kind = "directory" if fs.isdir(name) else "file"
        location = name if is_local else fs.unstrip_protocol(name)
        (subdirs if kind == "directory" else files).append(location)
    return files, subdirs


def resolve_layout(
    folder: str | os.PathLike,
    *,
    pattern: str | None = None,
    storage_options: dict | None = None,
) -> tuple[list[str], bool]:
    """Resolve a folder into its position folders.

    Args:
        folder: directory holding either the frames themselves or one subfolder
            per position. A plain path or any fsspec URL.
        pattern: glob for the frame filenames; ``None`` means ``.tif``/``.tiff``.
        storage_options: extra fsspec options, resolved as everywhere else.

    Returns:
        ``(position_folders, nested)``. ``nested`` is False when ``folder`` holds
        the frames itself (one position, the folder itself), True when each
        matching immediate subfolder is a position (natural-sorted).

    Raises:
        ValueError: If neither the folder nor any immediate subfolder holds a
            matching file. Detection is exactly one level deep -- a deeper tree
            raises rather than guessing which level is the position axis.
    """
    folder = str(folder)
    files, subdirs = _listing(folder, storage_options)

    own = [f for f in files if _matches(f, pattern)]
    positions = []
    for subdir in sorted(subdirs, key=natural_key):
        sub_files, _ = _listing(subdir, storage_options)
        if any(_matches(f, pattern) for f in sub_files):
            positions.append(subdir)

    if own:
        if positions:
            # e.g. an "overview.tif" dropped next to pos001/ ... pos060/: flat
            # wins (documented), but silently reading a 1-frame movie instead of
            # 60 positions is the kind of thing that must not pass unremarked
            warnings.warn(
                f"{folder!r} holds both {len(own)} matching file(s) and "
                f"{len(positions)} subfolder(s) that look like positions; reading "
                f"it as a single {len(own)}-frame movie and ignoring the "
                f"subfolders. Point at a subfolder, or use `pattern`, to change "
                f"that.",
                stacklevel=2,
            )
        return [folder], False

    if positions:
        return positions, True

    raise ValueError(
        f"no files matching {_pattern_text(pattern)} in {folder!r}, nor in any of "
        "its immediate subfolders -- a folder source expects one TIFF per "
        "timepoint (optionally one subfolder per position)"
    )


class FolderSequenceSource(ImageSequenceSource, JupyterVisualizationMixin):
    """A folder of per-timepoint TIFFs as a lazy ``(T, H, W, C)`` time series.

    Frame ``i`` is the ``i``-th file in natural-sorted order and is decoded only
    when asked for, so a 3000-frame folder streams the same way ND2/CZI does.
    ``size_t`` costs a directory listing and no pixel decode at all.

    The most recently read frame is cached (segmentation and property extraction
    ask for the same frame repeatedly), which keeps peak memory at ~one frame.
    Use :meth:`~acia.base.ImageSequenceSource.materialize` to trade RAM for IO and
    hold the whole movie in memory.

    Two consequences of that cache, shared with
    :class:`~acia.segm.local.LocalSequenceSource`: a returned frame's ``raw`` array
    is the cached array, so **do not modify it in place** (copy first, or call
    :meth:`close` to drop the cache); and an instance is not thread-safe -- give
    each worker its own source rather than sharing one across threads.

    Args:
        folder: directory holding the per-timepoint files. A plain local path or
            any fsspec-supported URL (e.g. ``smb://``).
        pattern: glob for the frame filenames, matched case-insensitively against
            the basename. ``None`` (default) matches ``.tif``/``.tiff``.
        storage_options: extra fsspec storage options (e.g. credentials), merged
            on top of the acia config entry for the host (see :mod:`acia.config`).
        normalize_image: normalize frames into uint8 RGB for display. Defaults to
            False, i.e. the file's own dtype and intensities are preserved (the
            raw-frame convention of ND2SequenceSource/CZISequenceSource).
        channel_axis: for 3-D files, which axis holds the channels: ``-1``
            (default) for ``(H, W, C)``, ``0`` for ``(C, H, W)``.
        pixel_size: physical pixel size (pint length per pixel); overrides the
            first file's own OME-XML/ImageJ metadata.
        frame_interval: scalar time between frames (pint Quantity or a string like
            ``"5 min"``). Per-timepoint files rarely carry timing, so this is
            usually the only source of it.
        timepoints: explicit per-frame timepoints (pint Quantity array); same
            override-vs-auto-detect behavior as the other calibration args.
    """

    def __init__(
        self,
        folder: str | os.PathLike,
        *,
        pattern: str | None = None,
        storage_options: dict | None = None,
        normalize_image: bool = False,
        channel_axis: int = -1,
        pixel_size=None,
        frame_interval=None,
        timepoints=None,
    ) -> None:
        if channel_axis not in (-1, 0, 2):
            raise ValueError(
                f"channel_axis must be -1/2 ((H, W, C)) or 0 ((C, H, W)), "
                f"got {channel_axis!r}"
            )
        self.folder = str(folder)
        self.pattern = pattern
        self.storage_options = storage_options
        self.normalize_image = normalize_image
        self.channel_axis = channel_axis
        # user-supplied calibration overrides (first-file metadata is read
        # otherwise, lazily -- construction itself must do no IO)
        self._user_pixel_size = pixel_size
        self._user_frame_interval = frame_interval
        self._user_timepoints = timepoints
        self._calibration_resolved = False
        self._calibration_source: str | None = None
        # lazy state
        self._files: list[str] | None = None
        self._reference: tuple[tuple[int, int, int], np.dtype] | None = None
        self._reference_path: str | None = None
        self._cached_index: int = -1
        self._cached_frame: np.ndarray | None = None

    # --- file listing ----------------------------------------------------------

    def _ensure_files(self) -> list[str]:
        """Resolve (once) the natural-sorted list of frame files. No pixel reads."""
        if self._files is None:
            files, _ = _listing(self.folder, self.storage_options)
            matches = sorted(
                (f for f in files if _matches(f, self.pattern)), key=natural_key
            )
            if not matches:
                raise ValueError(
                    f"no files matching {_pattern_text(self.pattern)} in "
                    f"{self.folder!r} -- a folder source expects one TIFF per timepoint"
                )
            self._files = matches
        return self._files

    @property
    def files(self) -> tuple[str, ...]:
        """The resolved frame files, in frame order (frame ``i`` is ``files[i]``).

        Worth printing the first and last entry when opening an unfamiliar folder:
        wrong ordering is the one failure mode here that is otherwise silent.
        """
        return tuple(self._ensure_files())

    # --- frames ----------------------------------------------------------------

    def _read(self, path: str) -> np.ndarray:
        opts = resolve_storage_options(path, self.storage_options)
        with fsspec.open(path, mode="rb", **opts) as handle:
            return tifffile.imread(handle)

    def _to_hwc(self, array: np.ndarray, path: str) -> np.ndarray:
        """Bring one decoded file to ``(H, W, C)``, rejecting non-plane files."""
        if array.ndim > 3 or array.ndim < 2:
            raise ValueError(
                f"{path!r} has {array.ndim} dimensions -- a folder source expects "
                "one plane per file ((H, W) or (H, W, C)). For a multi-frame stack "
                "inside a single file, open that file directly instead of its folder."
            )
        if array.ndim == 3:
            if self.channel_axis == 0:
                array = np.moveaxis(array, 0, -1)
            elif self.channel_axis == -1 and array.shape[-1] > _MAX_CHANNELS:
                # channel_axis=-1 is the *default*, i.e. "assume trailing channels".
                # A pages-first file ((C, H, W), a z-stack, a multi-page export)
                # would then be read as an image of shape[0] rows with shape[-1]
                # channels -- garbage that segments and tracks to completion. No
                # plane has hundreds of channels, so refuse to guess and say how
                # to resolve it. channel_axis=2 asserts (H, W, C) explicitly.
                raise ValueError(
                    f"{path!r} has shape {array.shape}, whose trailing axis is too "
                    f"long ({array.shape[-1]} > {_MAX_CHANNELS}) to be channels. If "
                    "it is (C, H, W) pass channel_axis=0; if it really is (H, W, C) "
                    "pass channel_axis=2; if it is a multi-frame stack, open that "
                    "file directly instead of its folder."
                )
        return np.asarray(prepare_image(array, self.normalize_image))

    def get_frame(self, frame: int) -> BaseImage:
        """Return frame ``frame``, decoding exactly the one file that backs it."""
        files = self._ensure_files()
        index = self._resolve_t_index(int(frame))
        if index == self._cached_index and self._cached_frame is not None:
            return LocalImage(self._cached_frame, frame=index)

        path = files[index]
        image = self._to_hwc(self._read(path), path)

        shape = (int(image.shape[0]), int(image.shape[1]), int(image.shape[2]))
        if self._reference is None:
            # the first frame actually decoded fixes the geometry; a normal
            # traversal (iteration, materialize, metadata) starts at frame 0, and
            # anchoring on it instead would cost a second decode for get_frame(i)
            self._reference = (shape, image.dtype)
            self._reference_path = path
        else:
            ref_shape, ref_dtype = self._reference
            if shape != ref_shape or image.dtype != ref_dtype:
                # name both files: whichever was read first defines "correct", so
                # the mismatch alone does not say which of the two is the odd one
                raise ValueError(
                    f"{path!r} is {shape} of {image.dtype}, but {self._reference_path!r} "
                    f"(read first) is {ref_shape} of {ref_dtype} -- every file in a "
                    "folder must hold the same plane geometry"
                )

        self._cached_index, self._cached_frame = index, image
        return LocalImage(image, frame=index)

    def _ensure_reference(self) -> tuple[tuple[int, int, int], np.dtype]:
        """Geometry of the frames, reading frame 0 once if nothing was read yet."""
        reference = self._reference
        if reference is None:
            self.get_frame(0)
            reference = self._reference
        if reference is None:  # pragma: no cover -- get_frame always records it
            raise RuntimeError(f"could not determine frame geometry of {self.folder!r}")
        return reference

    # --- shape -----------------------------------------------------------------

    @property
    def size_t(self) -> int:
        """Number of frames == number of matched files (listing only, no decode)."""
        return len(self._ensure_files())

    @property
    def size_h(self) -> int:
        return self._ensure_reference()[0][0]

    @property
    def size_w(self) -> int:
        return self._ensure_reference()[0][1]

    @property
    def size_c(self) -> int:
        return self._ensure_reference()[0][2]

    @property
    def num_channels(self) -> int:
        return self.size_c

    @property
    def dtype(self) -> np.dtype:
        """dtype of the frames as stored in the files."""
        return self._ensure_reference()[1]

    # --- calibration -----------------------------------------------------------

    def _ensure_calibration(self) -> None:
        """Resolve calibration once: user override > first file's metadata > None.

        Only the *first* file is inspected, and only its headers; if the caller
        supplied both ``pixel_size`` and a time calibration, no file is touched.

        Per-frame ``timepoints`` are **never** taken from the file. In a stack,
        the ``<Plane>`` ``DeltaT`` list spans the movie; in a per-timepoint file
        it describes that one file only, so adopting it would both contradict the
        frame count (a length-1 array for an N-frame folder) and quietly override
        the caller's ``frame_interval``. Reconstructing folder timing from each
        file's own timestamp is a separate, opt-in feature.
        """
        if self._calibration_resolved:
            return

        pixel_size = self._user_pixel_size
        frame_interval = self._user_frame_interval
        timepoints = self._user_timepoints

        if pixel_size is None or (frame_interval is None and timepoints is None):
            cal = read_tiff_calibration(self._ensure_files()[0], self.storage_options)
            if pixel_size is None:
                pixel_size = cal.pixel_size
            if frame_interval is None and timepoints is None:
                frame_interval = cal.frame_interval
            self._calibration_source = cal.source

        self._init_calibration(frame_interval, timepoints, pixel_size)
        self._calibration_resolved = True

    # `with_*` tag the calibration directly on the source. Resolve first, so the
    # lazy read below cannot fire afterwards and overwrite what was just set.
    def with_pixel_size(self, pixel_size):
        self._ensure_calibration()
        return ImageSequenceSource.with_pixel_size(self, pixel_size)

    def with_frame_interval(self, interval):
        self._ensure_calibration()
        return ImageSequenceSource.with_frame_interval(self, interval)

    def with_timepoints(self, timepoints):
        self._ensure_calibration()
        return ImageSequenceSource.with_timepoints(self, timepoints)

    @property
    def pixel_size(self):
        """Pint length per pixel: user override, else first file, else ``None``."""
        self._ensure_calibration()
        return ImageSequenceSource.pixel_size.fget(self)

    @property
    def timepoints(self):
        """Per-frame pint timepoints: user override, else first file, else ``None``."""
        self._ensure_calibration()
        return ImageSequenceSource.timepoints.fget(self)

    @property
    def calibration_source(self) -> str | None:
        """Where auto-detected calibration came from: ``"ome"``, ``"imagej"``, or
        ``None`` (nothing detected, or every field was user-supplied)."""
        self._ensure_calibration()
        return self._calibration_source

    # --- cleanup ---------------------------------------------------------------

    def close(self) -> None:
        """Drop the cached frame (the source stays usable)."""
        self._cached_index, self._cached_frame = -1, None

    def __repr__(self) -> str:
        n = len(self._files) if self._files is not None else "?"
        return f"FolderSequenceSource({self.folder!r}, {n} frames)"
