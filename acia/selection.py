"""Selection manifest — persist curation and reconstruct lazy crops.

A :class:`SelectionManifest` records a set of ROI selections (position + rotated
rectangle) plus the **source metadata baked in** (pixel size, timing, axes), so
downstream analysis reconstructs only the selected crops — lazily — without
re-opening the original (possibly hundreds-of-GB) file for calibration.

:func:`load_selection` turns a manifest back into a list of lazy cropped
:class:`~acia.base.ImageSequenceSource` views, optionally against a *different*
file ("load file, apply selection, work only on the crops"). No pixel data is
read while building the crops; each is a lazy rotated-rectangle view.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field

from acia.base import RotatedCropSpec

SCHEMA = "acia.selection/v1"


@dataclass
class RoiSelection:
    """One curated ROI: a rotated-rectangle crop of a single position."""

    position: int
    roi: RotatedCropSpec
    label: str = ""
    id: str = ""
    notes: str = ""
    preview: str | None = None

    @property
    def spec(self) -> RotatedCropSpec:
        """The rotated-crop spec (image pixels), usable by ``crop_rotated``."""
        return self.roi

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "position": int(self.position),
            "roi": self.roi.to_dict(),
            "label": self.label,
            "notes": self.notes,
            "preview": self.preview,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RoiSelection:
        return cls(
            position=int(data["position"]),
            roi=RotatedCropSpec.from_dict(data["roi"]),
            label=data.get("label", ""),
            id=data.get("id", ""),
            notes=data.get("notes", ""),
            preview=data.get("preview"),
        )


@dataclass
class SelectionManifest:
    """A curation result: selections + baked-in source metadata."""

    source: dict
    selections: list[RoiSelection] = field(default_factory=list)
    roi_mode: str = "single"
    schema: str = SCHEMA
    created: str | None = None

    @property
    def source_path(self) -> str:
        """Path of the original file the selections were made against."""
        return str(self.source.get("path", ""))

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "source": dict(self.source),
            "created": self.created,
            "roi_mode": self.roi_mode,
            "selections": [s.to_dict() for s in self.selections],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SelectionManifest:
        return cls(
            source=dict(data.get("source", {})),
            selections=[RoiSelection.from_dict(s) for s in data.get("selections", [])],
            roi_mode=data.get("roi_mode", "single"),
            schema=data.get("schema", SCHEMA),
            created=data.get("created"),
        )

    def save(self, path: str | os.PathLike) -> str:
        """Write the manifest as pretty JSON; returns the written path."""
        path = os.fspath(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str | os.PathLike) -> SelectionManifest:
        """Read a manifest from a ``selection.json`` file."""
        with open(os.fspath(path), encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def make_source_block(seqfile) -> dict:
    """Build the manifest ``source`` block from a :class:`SequenceFile`.

    Bakes in the metadata (pixel size, timing, axes, channels) and a
    ``(size, mtime)`` fingerprint so a moved/changed file can be detected later.
    """
    block = {
        "path": seqfile.path,
        "format": seqfile.format,
        "fingerprint": _fingerprint(seqfile.path),
        **seqfile.metadata.to_dict(),
    }
    return block


def save_selection(
    manifest: SelectionManifest,
    directory: str | os.PathLike,
    *,
    previews: dict[str, bytes] | None = None,
) -> str:
    """Write ``selection.json`` (+ optional ``previews/*.png``) into ``directory``.

    Args:
        manifest: The manifest to persist.
        directory: Output directory (created if missing) — typically beside the
            notebook.
        previews: Optional mapping of ``selection id -> PNG bytes`` written under
            ``directory/previews/``.

    Returns:
        The path to the written ``selection.json``.
    """
    directory = os.fspath(directory)
    os.makedirs(directory, exist_ok=True)
    if previews:
        preview_dir = os.path.join(directory, "previews")
        os.makedirs(preview_dir, exist_ok=True)
        for sel_id, data in previews.items():
            with open(os.path.join(preview_dir, f"{sel_id}.png"), "wb") as fh:
                fh.write(data)
    return manifest.save(os.path.join(directory, "selection.json"))


def load_selection(manifest: SelectionManifest, source=None) -> list:
    """Reconstruct lazy cropped sources from a manifest.

    Each selection becomes ``seqfile.position(i).crop_rotated(spec)`` — a lazy
    rotated-rectangle view; no pixel data is read here. Calibration comes from the
    (possibly overriding) source. The returned sources carry ``.selection`` and
    ``.label`` attributes so the ROI's identity is recoverable.

    Args:
        manifest: The manifest to reconstruct.
        source: ``None`` to open the manifest's original file, a path/str to apply
            the selections to a *different* file, or an already-open
            :class:`SequenceFile`.

    Returns:
        A list of lazy cropped ``ImageSequenceSource`` (one per selection).

    Raises:
        ValueError: If a selection's ``position`` is out of range for the source.
    """
    from acia.segm.open import SequenceFile, open_sequence

    if source is None:
        seqfile = open_sequence(manifest.source_path)
    elif isinstance(source, (str, os.PathLike)):
        seqfile = open_sequence(source)
    else:
        seqfile = source  # assume a SequenceFile (or compatible)

    if isinstance(seqfile, SequenceFile):
        _warn_on_fingerprint_mismatch(manifest, seqfile.path)

    crops = []
    for sel in manifest.selections:
        crop = seqfile.position(sel.position).crop_rotated(sel.spec)
        # Attach identity so the label/selection is recoverable from the crop.
        try:
            crop.selection = sel
            crop.label = sel.label
        except (AttributeError, TypeError):  # pragma: no cover - defensive
            pass
        crops.append(crop)
    return crops


def _fingerprint(path: str) -> dict:
    """A cheap ``(size, mtime)`` fingerprint (can't hash hundreds of GB)."""
    try:
        st = os.stat(path)
        return {"size": int(st.st_size), "mtime": float(st.st_mtime)}
    except OSError:
        return {}


def _warn_on_fingerprint_mismatch(manifest: SelectionManifest, path: str) -> None:
    fp = manifest.source.get("fingerprint") or {}
    expected = fp.get("size")
    if expected is None or not path or not os.path.exists(path):
        return
    actual = os.path.getsize(path)
    if actual != expected:
        warnings.warn(
            f"source file size {actual} != manifest fingerprint {expected} for "
            f"{path!r}; the file may have moved or changed.",
            stacklevel=2,
        )
