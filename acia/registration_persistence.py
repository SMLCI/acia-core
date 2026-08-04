"""Registration manifest — persist drift-correction transforms and reconstruct
lazy corrected sources.

A :class:`RegistrationManifest` records, per position, the per-frame
:class:`~acia.registration.FrameTransform` estimated by a chosen
:class:`~acia.registration.RegistrationMethod` against that position's own
frame 0 -- plus the **source metadata baked in** (pixel size, timing, axes),
mirroring :mod:`acia.selection`'s manifest pattern exactly.

:func:`load_registration` turns a manifest back into a ``dict`` of lazy
:class:`~acia.base.RegisteredSequenceSource` views (position index -> source),
optionally against a *different* file ("load file, apply registration, work
only on the corrected sequence"). No pixel data is read while reconstructing.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field

from acia.registration import FrameTransform

SCHEMA = "acia.registration/v1"


@dataclass
class RegistrationRecord:
    """One position's registration result: per-frame transforms + failures.

    Every transform is expressed relative to :attr:`reference_frame`, whatever
    frame was actually compared against to compute it. ``reference_mode`` and
    ``reference_frames`` record *how* they were obtained (see
    :class:`~acia.registration.ReanchoringReference`), which matters when
    resuming a partially-registered position: progress made under one policy is
    not valid to continue under another.

    Attributes:
        reference_mode: The reference policy this record was produced under --
            one of :data:`~acia.registration.ReanchoringReference.MODES`.
            Defaults to ``"fixed"`` so a manifest written before the policy
            existed loads as what it in fact was.
        reference_frames: ``frame -> anchor`` for the frames that were
            estimated against something other than :attr:`reference_frame`.
            Only the exceptions are stored; a purely fixed-reference run leaves
            this empty and serializes without the key at all.
    """

    position: int
    method: str
    transforms: dict[int, FrameTransform]
    reference_frame: int = 0
    failed_frames: dict[int, str] = field(default_factory=dict)
    notes: str = ""
    reference_mode: str = "fixed"
    reference_frames: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {
            "position": int(self.position),
            "method": self.method,
            "transforms": {
                str(frame): transform.to_dict()
                for frame, transform in self.transforms.items()
            },
            "reference_frame": int(self.reference_frame),
            "failed_frames": {
                str(frame): message for frame, message in self.failed_frames.items()
            },
            "notes": self.notes,
        }
        # Emit the newer keys only when they carry information, so a
        # fixed-reference run's JSON stays byte-identical to what earlier
        # versions of acia wrote (and still read).
        if self.reference_mode != "fixed":
            data["reference_mode"] = self.reference_mode
        if self.reference_frames:
            data["reference_frames"] = {
                str(frame): int(anchor)
                for frame, anchor in self.reference_frames.items()
            }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> RegistrationRecord:
        return cls(
            position=int(data["position"]),
            method=data.get("method", ""),
            transforms={
                int(frame): FrameTransform.from_dict(transform)
                for frame, transform in data.get("transforms", {}).items()
            },
            reference_frame=int(data.get("reference_frame", 0)),
            failed_frames={
                int(frame): message
                for frame, message in data.get("failed_frames", {}).items()
            },
            notes=data.get("notes", ""),
            reference_mode=data.get("reference_mode", "fixed"),
            reference_frames={
                int(frame): int(anchor)
                for frame, anchor in data.get("reference_frames", {}).items()
            },
        )


@dataclass
class RegistrationManifest:
    """A batch-apply result: per-position records + baked-in source metadata.

    Attributes:
        method_params: The settings the registration method was constructed
            with (``min_confidence``, ``exclude_shrink_px``, ...), so a run is
            reproducible from the file rather than only from the notebook that
            produced it. Only JSON-representable values are kept; anything else
            is stringified.
    """

    source: dict
    records: list[RegistrationRecord] = field(default_factory=list)
    method: str = ""
    schema: str = SCHEMA
    created: str | None = None
    method_params: dict = field(default_factory=dict)

    @property
    def source_path(self) -> str:
        """Path of the original file the registration was estimated against."""
        return str(self.source.get("path", ""))

    def to_dict(self) -> dict:
        data = {
            "schema": self.schema,
            "source": dict(self.source),
            "created": self.created,
            "method": self.method,
            "records": [r.to_dict() for r in self.records],
        }
        if self.method_params:
            data["method_params"] = _jsonable(self.method_params)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> RegistrationManifest:
        return cls(
            source=dict(data.get("source", {})),
            records=[RegistrationRecord.from_dict(r) for r in data.get("records", [])],
            method=data.get("method", ""),
            schema=data.get("schema", SCHEMA),
            created=data.get("created"),
            method_params=dict(data.get("method_params", {})),
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
    def load(cls, path: str | os.PathLike) -> RegistrationManifest:
        """Read a manifest from a ``registration_transforms.json`` file."""
        with open(os.fspath(path), encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def _jsonable(value):
    """Best-effort conversion of arbitrary method params to JSON-safe values.

    Method settings are plain scalars in the common case, but one of them
    (``exclude_rects``) is a sequence of dataclasses. Rather than teach this
    module every possible parameter type, anything ``json`` cannot represent is
    recorded as its ``repr`` -- these values exist to make a run *legible*, not
    to be loaded back as objects.
    """
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def save_registration(
    manifest: RegistrationManifest, directory: str | os.PathLike
) -> str:
    """Write ``registration_transforms.json`` into ``directory``.

    Args:
        manifest: The manifest to persist.
        directory: Output directory (created if missing) — typically beside the
            notebook.

    Returns:
        The path to the written ``registration_transforms.json``.
    """
    directory = os.fspath(directory)
    os.makedirs(directory, exist_ok=True)
    return manifest.save(os.path.join(directory, "registration_transforms.json"))


def load_registration(
    manifest: RegistrationManifest, source=None, *, on_missing: str = "warn"
) -> dict:
    """Reconstruct lazy registered sources from a manifest.

    Each completed record becomes ``seqfile.position(record.position).register(
    record.transforms)`` — a lazy per-frame-corrected view; no pixel data is
    read here. Calibration comes from the (possibly overriding) source.

    Args:
        manifest: The manifest to reconstruct.
        source: ``None`` to open the manifest's original file, a path/str to
            apply the records to a *different* file, or an already-open
            :class:`~acia.segm.open.SequenceFile`.
        on_missing: How each reconstructed view should handle a frame that has
            no stored transform (one that landed in ``failed_frames``) — see
            :class:`~acia.base.RegisteredSequenceSource`.

    Returns:
        dict[int, ~acia.base.RegisteredSequenceSource]: Position index -> lazy
        registered source, one entry per record in the manifest.

    Raises:
        ValueError: If a record's ``position`` is out of range for the source.
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

    sources = {}
    for record in manifest.records:
        sources[record.position] = seqfile.position(record.position).register(
            record.transforms, on_missing=on_missing
        )
    return sources


def _warn_on_fingerprint_mismatch(manifest: RegistrationManifest, path: str) -> None:
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
