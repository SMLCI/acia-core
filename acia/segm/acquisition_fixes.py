"""In-memory correction of acquisition slips in multi-position timelapses.

A multi-position ND2/CZI timelapse is a fixed ``T x P`` grid filled in acquisition
order -- time outer, position inner. Occasionally the microscope records **two
consecutive images of the same position**: the extra image takes the next slot, so
every later slot, through the end of the file, holds the image that belongs one
slot earlier, and the final real image is lost.

An :class:`AcquisitionSlip` names the raw slot ``(t, position)`` that holds the
repeated image -- the first slot that looks wrong in a viewer. A :class:`SlotMap`
drops those slots and maps every true ``(t, p)`` onto the raw slot where its image
actually landed; :class:`CorrectedPositionSource` reads through that map. Nothing
is ever written to the raw file.

Fixes are declared once in a JSON sidecar next to the raw file
(:func:`save_fixes`, ``<file>.acia-fixes.json``), which
:func:`acia.segm.open.open_sequence` picks up automatically, so every notebook and
every stage reads the corrected order. :func:`detect_acquisition_slips` suggests
fixes from the per-image stage XY metadata: a double shot is two consecutive raw
slots at the same stage position.
"""

from __future__ import annotations

import bisect
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from acia.base import BaseImage, ImageSequenceSource
from acia.segm.local import LocalImage

SIDECAR_SUFFIX = ".acia-fixes.json"
_SCHEMA_VERSION = 1


@dataclass(frozen=True, order=True)
class AcquisitionSlip:
    """A raw slot ``(t, position)`` that holds a repeat of the previous slot's image.

    Coordinates are those of the **raw** file (what a viewer shows), not corrected
    ones. The previous slot is ``(t, position - 1)``, or ``(t - 1, P - 1)`` for
    ``position == 0``.
    """

    t: int
    position: int

    def to_dict(self) -> dict:
        return {"t": int(self.t), "position": int(self.position)}

    @classmethod
    def coerce(cls, value) -> AcquisitionSlip:
        """Accept an ``AcquisitionSlip``, a ``(t, position)`` pair or a dict."""
        if isinstance(value, AcquisitionSlip):
            return value
        if isinstance(value, dict):
            return cls(int(value["t"]), int(value["position"]))
        t, position = value
        return cls(int(t), int(position))


def coerce_slips(values: Iterable) -> list[AcquisitionSlip]:
    """Normalize user input into a sorted list of :class:`AcquisitionSlip`."""
    return sorted(AcquisitionSlip.coerce(v) for v in values)


class SlotMap:
    """Map true ``(t, p)`` onto the raw slot that holds its image.

    Args:
        num_t: raw number of timepoints ``T``.
        num_p: number of positions ``P``.
        slips: the raw slots holding a repeated image.

    Raises:
        ValueError: on a slip out of range, a duplicate, or a slip at ``(0, 0)``
            (the very first image has nothing to repeat).
    """

    def __init__(self, num_t: int, num_p: int, slips: Iterable) -> None:
        self.num_t = int(num_t)
        self.num_p = int(num_p)
        self.slips = coerce_slips(slips)
        if len(set(self.slips)) != len(self.slips):
            raise ValueError(f"duplicate acquisition slips in {self.slips}")
        flat = []
        for s in self.slips:
            if not (0 <= s.t < self.num_t and 0 <= s.position < self.num_p):
                raise ValueError(
                    f"acquisition slip {s} out of range for T={self.num_t}, "
                    f"P={self.num_p}"
                )
            k = s.t * self.num_p + s.position
            if k == 0:
                raise ValueError(
                    "acquisition slip at (t=0, position=0) is impossible: the "
                    "first image has no previous image to repeat"
                )
            flat.append(k)
        self._flat = sorted(flat)
        self._n_true = self.num_t * self.num_p - len(self._flat)

    def raw_slot(self, t: int, p: int) -> tuple[int, int] | None:
        """The raw ``(t, p)`` holding true ``(t, p)``, or ``None`` if it was lost."""
        j = t * self.num_p + p
        if not 0 <= j < self._n_true:
            return None
        # the j-th raw slot that is not a slip: step past every slip at or
        # before the running raw index (slips are sorted ascending)
        k = j
        for s in self._flat:
            if s <= k:
                k += 1
            else:
                break
        return divmod(k, self.num_p)

    def size_t(self, p: int) -> int:
        """Number of true timepoints of position ``p`` that have an image."""
        # true flat indices t*P + p < n_true -> t < (n_true - p) / P
        return max(0, min(self.num_t, -(-(self._n_true - p) // self.num_p)))

    def shift_at(self, t: int, p: int) -> int:
        """How many slots the raw file is shifted at raw ``(t, p)`` (diagnostics)."""
        return bisect.bisect_right(self._flat, t * self.num_p + p)


class CorrectedPositionSource(ImageSequenceSource):
    """One position of a :class:`~acia.segm.open.SequenceFile`, slip-corrected.

    Frame ``t`` is read from the raw slot the :class:`SlotMap` points at -- often a
    *different* raw position's reader -- and relabelled as frame ``t``. Lazy: only
    the requested frame is read.
    """

    def __init__(self, seqfile, position: int, slot_map: SlotMap) -> None:
        self._file = seqfile
        self.position = int(position)
        self.slot_map = slot_map
        self.path = seqfile.path
        self._init_calibration(frame_interval=seqfile._frame_interval)

    def _raw(self):
        return self._file.raw_position(self.position)

    # --- metadata delegated to this position's raw reader ---------------------

    @property
    def sizes(self) -> dict[str, int]:
        sizes = dict(self._raw().sizes)
        if "T" in sizes:
            sizes["T"] = self.size_t
        return sizes

    @property
    def dtype(self) -> str:
        return str(self._raw().dtype)

    @property
    def channel_names(self) -> list[str]:
        return list(self._raw().channel_names)

    @property
    def pixel_size(self):
        return self._raw().pixel_size

    # --- ImageSequenceSource contract -----------------------------------------

    @property
    def size_t(self) -> int:
        return self.slot_map.size_t(self.position)

    @property
    def size_h(self) -> int:
        return int(self._raw().size_h)

    @property
    def size_w(self) -> int:
        return int(self._raw().size_w)

    @property
    def size_c(self) -> int:
        return int(self._raw().size_c)

    @property
    def num_channels(self) -> int:
        return int(self._raw().num_channels)

    def raw_slot(self, frame: int) -> tuple[int, int]:
        """The raw ``(t, position)`` that true ``frame`` is read from."""
        slot = self.slot_map.raw_slot(frame, self.position)
        if slot is None or frame < 0:
            raise IndexError(f"frame {frame} out of range for size_t={self.size_t}")
        return slot

    def get_frame(self, frame: int) -> BaseImage:
        t_raw, p_raw = self.raw_slot(frame)
        image = self._file.raw_position(p_raw).get_frame(t_raw)
        return LocalImage(image.raw, frame=frame)

    def stage_xy(self, frame: int) -> tuple[float, float] | None:
        """Recorded stage XY of the image behind true ``frame`` (µm), if known."""
        t_raw, p_raw = self.raw_slot(frame)
        xy = self._file.raw_position(p_raw).stage_xy(t_raw)
        return None if xy is None else (float(xy[0]), float(xy[1]))


def preview_slip(seqfile, slip, *, downscale: int = 8, channel: int = 0):
    """The raw slots before / at / after ``slip``, side by side and labelled.

    A double shot is confirmed by eye: the middle tile should be identical to the
    one on its left. Reads the file's **raw** slot order (three frames), so it
    shows what is in the file rather than what a correction would make of it.

    Args:
        seqfile: the :class:`~acia.segm.open.SequenceFile` to read (any fixes it
            carries are bypassed -- the raw slots are shown).
        slip: the :class:`AcquisitionSlip` (or ``(t, position)``) to preview.
        downscale: stride applied to each tile, as in
            :meth:`~acia.segm.open.SequenceFile.thumbnail`.
        channel: which channel to display.

    Returns:
        PIL.Image.Image: the labelled strip, ready to ``display()``.
    """
    from PIL import Image, ImageDraw

    from acia.notebook import normalize_to_uint8

    slip = AcquisitionSlip.coerce(slip)
    num_p = seqfile.num_positions
    num_slots = seqfile.metadata.num_timepoints * num_p
    k = slip.t * num_p + slip.position

    tiles, labels = [], []
    for slot in (k - 1, k, k + 1):
        if not 0 <= slot < num_slots:
            continue
        t, p = divmod(slot, num_p)
        raw = np.asarray(seqfile.raw_position(p).get_frame(t).raw)
        plane = raw[..., channel] if raw.ndim == 3 else raw
        tiles.append(normalize_to_uint8(plane[::downscale, ::downscale]))
        labels.append(f"t={t} pos={p}" + (" (repeat)" if slot == k else ""))

    band, gap = 14, 8
    height = max(t.shape[0] for t in tiles)
    width = sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)
    strip = Image.new("L", (width, height + band), color=255)
    draw = ImageDraw.Draw(strip)
    x = 0
    for tile, label in zip(tiles, labels, strict=True):
        strip.paste(Image.fromarray(tile), (x, band))
        draw.text((x + 2, 2), label, fill=0)
        x += tile.shape[1] + gap
    return strip


# --------------------------------------------------------------------------- #
# sidecar
# --------------------------------------------------------------------------- #
def sidecar_path(path: str | os.PathLike) -> str:
    """The fixes sidecar next to a raw file: ``<file>.acia-fixes.json``."""
    return os.fspath(path) + SIDECAR_SUFFIX


def _open(path: str, mode: str):
    if "://" not in path:
        return open(path, mode, encoding="utf-8")
    import fsspec

    from acia.config import resolve_storage_options

    return fsspec.open(path, mode, encoding="utf-8", **resolve_storage_options(path))


def _exists(path: str) -> bool:
    if "://" not in path:
        return os.path.exists(path)
    import fsspec

    from acia.config import resolve_storage_options

    try:
        fs, root = fsspec.core.url_to_fs(path, **resolve_storage_options(path))
        return bool(fs.exists(root))
    except Exception:  # noqa: BLE001 - an unreachable remote means "no sidecar"
        return False


def has_fixes(path: str | os.PathLike) -> bool:
    """Whether a fixes sidecar exists next to ``path``."""
    return _exists(sidecar_path(path))


def load_fixes(
    path: str | os.PathLike, *, num_t: int | None = None, num_p: int | None = None
) -> list[AcquisitionSlip]:
    """Read the slips from the sidecar next to ``path`` (``[]`` if there is none).

    Args:
        path: the raw acquisition (not the sidecar itself).
        num_t, num_p: the raw file's sizes; when given, a sidecar written for a
            file of different shape raises instead of being applied.

    Raises:
        ValueError: on an unknown schema version or a size mismatch.
    """
    side = sidecar_path(path)
    if not _exists(side):
        return []
    with _open(side, "r") as fh:
        data = json.load(fh)
    version = data.get("version")
    if version != _SCHEMA_VERSION:
        raise ValueError(f"{side!r}: unsupported fixes schema version {version!r}")
    sizes = data.get("sizes") or {}
    for key, actual in (("T", num_t), ("P", num_p)):
        expected = sizes.get(key)
        if actual is not None and expected is not None and int(expected) != actual:
            raise ValueError(
                f"{side!r} was written for {key}={expected}, but the file has "
                f"{key}={actual}; it does not belong to this acquisition"
            )
    return coerce_slips(data.get("slips", []))


def save_fixes(
    path: str | os.PathLike,
    slips: Iterable,
    *,
    num_t: int | None = None,
    num_p: int | None = None,
    note: str = "",
) -> str:
    """Write the fixes sidecar next to ``path``. The raw file is not touched.

    ``num_t``/``num_p`` are read from the file when not given (a metadata read,
    no pixels), and are stored so the sidecar can't be applied to another file.

    Returns:
        The sidecar's path.
    """
    path = os.fspath(path)
    if num_t is None or num_p is None:
        from acia.segm.open import open_sequence

        with open_sequence(path, acquisition_fixes=[]) as seqfile:
            num_t, num_p = seqfile.metadata.num_timepoints, seqfile.num_positions
    slip_list = SlotMap(num_t, num_p, slips).slips  # validates
    side = sidecar_path(path)
    payload = {
        "version": _SCHEMA_VERSION,
        "source": os.path.basename(path),
        "sizes": {"T": int(num_t), "P": int(num_p)},
        "slips": [s.to_dict() for s in slip_list],
        "note": note,
    }
    with _open(side, "w") as fh:
        json.dump(payload, fh, indent=2)
    return side


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def detect_acquisition_slips(
    stage_positions, *, tol_um: float = 2.0
) -> list[AcquisitionSlip]:
    """Suggest slips from recorded stage XY: two consecutive raw slots at one place.

    Args:
        stage_positions: a :class:`~acia.segm.open.SequenceFile` (its
            :meth:`~acia.segm.open.SequenceFile.stage_positions` is read), or a raw
            ``(T, P, 2)`` array of stage XY in µm (NaN where unknown).
        tol_um: two images closer than this are "the same position".

    Returns:
        The suggested slips in raw coordinates, sorted. Nothing is applied.

    Raises:
        ValueError: if there is no stage metadata, only one position, so many
            slots look repeated that the values can't be stage µm, or the
            positions are still indistinguishable once the slips are taken out (a
            double shot would then be indistinguishable from the normal sequence).
    """
    if hasattr(stage_positions, "stage_positions"):
        xy = stage_positions.stage_positions()
    else:
        xy = np.asarray(stage_positions, dtype=float)
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"expected a (T, P, 2) stage array, got shape {xy.shape}")
    num_t, num_p, _ = xy.shape
    if num_p < 2:
        raise ValueError(
            "slip detection needs at least two positions: with one position every "
            "consecutive image is at the same place by design"
        )
    if np.all(np.isnan(xy)):
        raise ValueError(
            "the file carries no per-image stage XY metadata; declare slips by hand"
        )

    flat = xy.reshape(num_t * num_p, 2)
    step = np.linalg.norm(flat[1:] - flat[:-1], axis=-1)  # NaN where unknown
    ks = np.nonzero(step < tol_um)[0] + 1
    slips = [AcquisitionSlip(*map(int, divmod(int(k), num_p))) for k in ks]

    # A handful of repeats is a double shot; a large fraction of them means the
    # values aren't stage µm (some formats report mm) or tol_um is far too large.
    typical = float(np.nanmedian(step)) if np.any(np.isfinite(step)) else float("nan")
    if len(slips) > max(3, 0.05 * num_t * num_p):
        raise ValueError(
            f"{len(slips)} of {num_t * num_p} slots look like repeats at "
            f"tol_um={tol_um}, with a typical step of {typical:.3f} between "
            "consecutive images; that reads like a unit mismatch (stage values in "
            "mm rather than µm) or too large a tolerance, not a double shot"
        )

    # Once the repeats are taken out, each position must sit at its own place --
    # otherwise a repeat cannot be told apart from moving on to the next position.
    # Checked here, after the fact, because a slip in the very first timepoint
    # would make the raw layout of that timepoint look ambiguous by itself.
    slot_map = SlotMap(num_t, num_p, slips)
    centers = np.full((num_p, 2), np.nan)
    for p in range(num_p):
        seen = [slot_map.raw_slot(t, p) for t in range(slot_map.size_t(p))]
        if seen:
            centers[p] = np.nanmedian(np.array([xy[s] for s in seen if s]), axis=0)
    pts = centers[~np.isnan(centers).any(axis=1)]
    if len(pts) >= 2:
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        d[np.diag_indices_from(d)] = np.inf
        if d.min() < tol_um:
            raise ValueError(
                f"two positions still lie {d.min():.2f} µm apart (< tol_um="
                f"{tol_um}) once the {len(slips)} detected repeat(s) are removed; "
                "slips can't be told apart from the next position in this layout"
            )
    return slips


# --------------------------------------------------------------------------- #
# manifests
# --------------------------------------------------------------------------- #
def manifest_fixes(source_block: dict) -> list[AcquisitionSlip] | None:
    """The fixes a manifest's ``source`` block was recorded under.

    ``None`` for a manifest written before fixes existed, so reopening it falls
    back to the sidecar (the pre-existing behaviour).
    """
    if "acquisition_fixes" not in source_block:
        return None
    return coerce_slips(source_block["acquisition_fixes"] or [])


def warn_on_fixes_mismatch(source_block: dict, seqfile) -> None:
    """Warn when ``seqfile`` reads under different fixes than the manifest recorded.

    Frame indices in a manifest (selection anchor frames, registration
    transforms) are only meaningful under the fixes they were recorded with.
    """
    import warnings

    recorded = manifest_fixes(source_block)
    current = getattr(seqfile, "acquisition_fixes", None)
    if recorded is None or current is None:
        return
    if list(recorded) != list(current):
        warnings.warn(
            f"the manifest was recorded with acquisition fixes {recorded}, but the "
            f"source now uses {list(current)}; its frame indices may point at "
            "different images",
            stacklevel=3,
        )
