"""Export curated ROIs to one TIFF per selection, optionally drift-corrected.

:func:`export_rois` is the writing half of the curation workflow:
:func:`acia.selection.load_selection` turns a manifest into lazy crops, this turns
it into files. It works position by position -- a position with no ROIs is skipped
outright, and registration (when enabled) runs for one position at a time,
immediately followed by that position's crops -- so the work stays scoped to what
is actually being exported.

Registering a growing colony needs care, and two defaults here address it:

* ``exclude_rois`` passes each position's own ROI rectangles to the registration
  method as ``exclude_rects``, so its confidence is computed on the static device
  geometry around them rather than on the cells that grow inside them. Without
  that, the confidence decays with elapsed biology rather than with misalignment,
  and the method's gate rejects perfectly good fits in a tail of late frames.
  ``exclude_shrink_px`` keeps a band that wide just inside each ROI border, whose
  sharp channel walls measurably help precision. The rectangles are used in their
  as-drawn coordinates on every frame: mapping them per frame would need the very
  transforms the mask exists to help estimate, and the shrink band already covers
  a few px of drift.
* ``low_confidence='keep'`` stores a weak fit (carrying its score) rather than
  dropping it, so the transforms come out gap-free and a weak run announces
  itself in the per-position report instead of as missing frames.

Each selection remembers the frame it was drawn on. That matters only once
registration is in play: a registered view puts every frame into the reference
frame's coordinate system, so a box drawn on frame 148 is carried across with
:func:`acia.registration.apply_correction_to_spec` -- otherwise the crop lands
where frame 148 had drifted to, off by the whole accumulated drift.

Positions are **1-based** wherever a person reads them -- file names
(``pos001_roi001.tiff`` is the first position), log lines and
``roi_export.json`` -- to match how Nikon NIS-Elements and Zeiss ZEN number
them. The manifest (``selection.json``) keeps the 0-based position index.

Which index starts where:

==============  ==========  ===================================================
index           first       where
==============  ==========  ===================================================
position        1           dashboard (``pos 001``), file names, log lines,
                            ``roi_export.json`` (``position``)
position index  0           ``selection.json``, ``acquisition_fixes``,
                            ``roi_export.json`` (``position_index``), Python API
ROI number      1           file names (``roi001``), ``roi_export.json``
frame / t       0           everywhere: dashboard frame bar, ``anchor_frame``,
                            ``max_frames``, ``acquisition_fixes`` ``(t, ...)``,
                            the ``frame`` column of every table
time            0 min       the first frame is at t = 0
==============  ==========  ===================================================

Next to the TIFFs, :func:`export_rois` writes ``roi_export.json``: where the
export came from (the ``.nd2``/``.czi`` file, which names the *experiment*),
how it was made, and which file holds which ROI. Batch analysis reads it to name
its output folders after the experiment (see :func:`experiment_name`).

**How a position is exported, and what it costs in RAM.** A position's frames
are walked once, and each frame is cropped for *every* ROI of that position
while it is in hand -- rather than once per ROI, which would read the position
again for each one. So a position costs one pass over its frames however many
ROIs sit in it, and peak memory is the ROI stacks being filled (bounded by
``max_roi_buffer_bytes``, which splits the ROIs over as many passes as it takes)
plus the handful of frames the reader is holding (``chunk_frames``). Neither
grows with the length of the timelapse, which is the point: the whole position
used to be held in memory at once.

Frames are read **uncorrected**, and each ROI's rectangle is carried back into
that frame's own coordinate system instead
(:meth:`acia.base.RegisteredSequenceSource.spec_for`), so the drift correction
and the crop happen in a single warp. The two are affine and compose exactly --
it is the same region -- but out of one interpolation rather than two, so the
export is sharper as well as cheaper: the full-frame warp, which dominates the
per-frame cost on a large frame, is gone. :data:`CROP_PIPELINE` records which
pipeline wrote a file, so a change here re-exports rather than mixing two
resamplings in one folder.

**Re-running is incremental.** Draw one more ROI in the curation dashboard, run
the export again, and only that ROI is written: a selection whose file is
already there, unchanged and complete, is skipped, and a position whose ROIs are
all skipped is never even opened -- so it costs no read and no registration.
That is also how an export that was interrupted picks up where it stopped. Three
things make it trustworthy rather than merely fast:

* ``roi_export.json`` is rewritten after *every* ROI, not once at the end, so a
  run that dies still leaves an accurate record of what it finished.
* A TIFF is written beside its target and renamed into place, so the final path
  never holds a half-written file (and ``verify_existing`` checks the frame count
  in the header of one that was written before this was true).
* The recorded entry carries the selection's own geometry and the settings that
  change pixels (``max_frames``, registration, ``ome``, ``compression``), so
  moving an ROI or changing those re-exports it instead of being skipped.

``roi_nb`` -- which goes into the file name -- is kept stable across runs for
this reason: a selection that already has an entry keeps its number, and a new
one takes the lowest number still free in its position. Numbering purely by
manifest order would reassign existing files the moment a new ROI was inserted
before an old one. Pass ``on_existing='overwrite'`` to redo an export from
scratch, or ``'error'`` to refuse to touch one.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import queue
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

# Which constructor argument is the confidence gate, per method. Only the methods
# whose reported ``confidence`` is in the *same units* as their gate are listed:
# FeatureRANSACEuclidean gates on an inlier count while reporting an inlier
# fraction, so there is no honest "below threshold" count to report for it.
_CONFIDENCE_GATE = {
    "GradientECC": "min_confidence",
    "MaskedTemplateCorrelation": "min_score",
}

DEFAULT_NAMING_TEMPLATE = "pos{position:03d}_roi{roi_nb:03d}.tiff"

#: the export description written next to the TIFFs
EXPORT_INFO_NAME = "roi_export.json"
EXPORT_INFO_SCHEMA = "acia.roi_export/v2"

#: what ``export_rois(on_existing=...)`` accepts
ON_EXISTING_POLICIES = ("skip", "overwrite", "error")

#: inserted before the extension while a TIFF is still being written
PART_SUFFIX = ".part"

#: how many frames the crop pass reads ahead of itself by default
DEFAULT_CHUNK_FRAMES = 8

#: default RAM budget for one position's in-flight ROI buffers
DEFAULT_MAX_ROI_BUFFER_BYTES = 2 * 1024**3

#: minimum seconds between the registration dashboard's progress messages
PROGRESS_INTERVAL_S = 0.05

#: How the exported pixels are produced, recorded per ROI. Bumped whenever that
#: changes, so an existing file written by an older pipeline is re-exported
#: rather than silently mixed with new ones. ``single-warp`` composes the drift
#: correction into the crop, resampling once; the ``two-warp`` it replaced
#: corrected the whole frame and then cropped the corrected copy.
CROP_PIPELINE = "single-warp/v1"

#: default folder (next to the raw file) that exports go into
DEFAULT_EXPORT_FOLDER = "roi_exports"


def default_export_dir(source_path: str | os.PathLike) -> Path:
    """``<folder of the raw file>/roi_exports``.

    Keeps the exported ROIs next to the experiment they come from. The raw data
    folder usually already names the experiment, so there is no per-file
    subfolder; the raw file's name is recorded in ``roi_export.json`` instead.
    A second acquisition in the same folder needs its own ``output_dir`` --
    :func:`export_rois` refuses to mix two acquisitions in one folder.
    """
    return Path(source_path).parent / DEFAULT_EXPORT_FOLDER


def experiment_name(export_dir: str | os.PathLike) -> str:
    """The experiment an export folder belongs to.

    The ``experiment`` recorded in its ``roi_export.json``, else the raw file's
    stem from ``selection.json``, else the folder's own name.
    """
    folder = Path(export_dir)
    for name, pick in (
        (EXPORT_INFO_NAME, lambda d: d.get("experiment")),
        (
            "selection.json",
            lambda d: Path((d.get("source") or {}).get("path") or "").stem,
        ),
    ):
        try:
            value = pick(json.loads((folder / name).read_text()))
        except (OSError, ValueError, AttributeError):
            continue
        if value:
            return str(value)
    return folder.name


def export_rois(
    manifest,
    output_dir: str | os.PathLike,
    naming_template: str = DEFAULT_NAMING_TEMPLATE,
    registration_method_name: str | None = None,
    *,
    source=None,
    max_frames: int | None = None,
    method_kwargs: dict | None = None,
    reference_mode: str = "reanchor",
    low_confidence: str = "keep",
    exclude_rois: bool = True,
    exclude_shrink_px: float = 10.0,
    on_missing: str = "nearest",
    ome: bool = True,
    compression: str | int | None = "zlib",
    progress: bool = True,
    experiment: str | None = None,
    verbose: bool = False,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    max_roi_buffer_bytes: int | None = DEFAULT_MAX_ROI_BUFFER_BYTES,
    on_existing: str = "skip",
    verify_existing: bool = True,
) -> list[str]:
    """Crop every selection to its own TIFF, optionally registering each position.

    Args:
        manifest: the :class:`~acia.selection.SelectionManifest` to export.
        output_dir: where the TIFFs (and, when registering,
            ``registration_transforms.json``) are written; created if missing.
        naming_template: filename pattern, formatted with ``position``
            (**1-based**, as the microscope software numbers positions),
            ``position_index`` (0-based, as stored in the manifest), ``roi_nb``
            (1-based within the position, stable across runs -- see
            ``on_existing``), ``label`` and ``id``.
        registration_method_name: an :mod:`acia.registration` method name to
            drift-correct each position first, or ``None`` to crop as-is.
        source: reuse an already-open :class:`~acia.segm.open.SequenceFile` (e.g.
            the dashboard's) instead of reopening ``manifest.source_path`` -- same
            convention as :func:`acia.selection.load_selection`.
        max_frames: lazily truncate each position to its first N frames *before*
            registration and cropping, so a quick test run never mixes registered
            and unregistered frames in one TIFF. ``None`` exports everything.
        method_kwargs: passed to the registration method's constructor
            (``early_stop_delta_px``, ``min_confidence``, ...).
        reference_mode: ``'fixed'``, ``'reanchor'`` or ``'chained'``.
        low_confidence: ``'keep'`` or ``'reject'``; see the module docstring.
        exclude_rois: keep this position's ROI interiors out of the registration
            objective (see the module docstring).
        exclude_shrink_px: width of the band kept just inside each ROI border.
        on_missing: what a frame with no transform does -- ``'nearest'``,
            ``'warn'`` or ``'error'``.
        ome: write OME-TIFF rather than an ImageJ hyperstack.
        compression: passed to :func:`~acia.segm.tiff_export.save_tiff_stack`.
        progress: show progress bars and one report line per position (plus
            a final summary). Per-frame registration warnings are folded into
            that line instead of being printed one by one.
        verbose: also print the per-ROI details (how far each ROI was moved
            onto the corrected view, the exclusion-rectangle note).
        experiment: name recorded in ``roi_export.json``, used by batch
            analysis to name this export's output folder; defaults to the raw
            file's stem.
        chunk_frames: how many frames the crop pass reads ahead of itself, on a
            single background reader thread, so the next read overlaps the
            current frame's warps. Peak memory is about this many full frames.
            ``1`` reads strictly one frame at a time (no thread).
        max_roi_buffer_bytes: RAM budget for one position's in-flight ROI
            buffers. ROIs are exported in as many frame passes as it takes to
            stay inside it; a single ROI that exceeds it on its own streams
            lazily instead. ``None`` lifts the cap.
        on_existing: what to do with a selection that already has a file here --
            ``'skip'`` (the default; export only what is new or out of date),
            ``'overwrite'`` or ``'error'``. See the module docstring.
        verify_existing: before trusting an existing file, check its TIFF header
            (frame count only -- no pixels are read). ``False`` trusts the file
            name alone, which is faster over a network share.

    Returns:
        The written paths, in export order. Skipped selections are not listed.

    Raises:
        ValueError: if an ROI was drawn on a frame outside the range being
            exported (checked up front, before any registration work), if a
            drawn-on frame has no transform after registration, or if
            ``on_existing='error'`` and a selection already has a file.
    """
    from acia.segm.acquisition_fixes import manifest_fixes
    from acia.segm.open import open_sequence

    if on_existing not in ON_EXISTING_POLICIES:
        raise ValueError(
            f"Unknown on_existing policy {on_existing!r}; expected one of "
            f"{', '.join(ON_EXISTING_POLICIES)}."
        )
    if int(chunk_frames) < 1:
        raise ValueError(f"chunk_frames must be >= 1, got {chunk_frames!r}")

    seqfile = (
        source
        if source is not None
        else open_sequence(
            manifest.source_path, acquisition_fixes=manifest_fixes(manifest.source)
        )
    )
    output_dir = os.fspath(output_dir)
    say = tqdm.write if progress else lambda *a, **k: None
    detail = say if verbose else lambda *a, **k: None

    by_position: dict[int, list] = {}
    for sel in manifest.selections:
        by_position.setdefault(sel.position, []).append(sel)
    positions = [p for p in range(seqfile.metadata.num_positions) if by_position.get(p)]

    _refuse_foreign_export(output_dir, manifest.source_path)
    os.makedirs(output_dir, exist_ok=True)
    _clean_part_files(output_dir)

    settings = _export_settings(max_frames, registration_method_name, ome, compression)
    previous = _previous_entries(output_dir)
    roi_numbers = _assign_roi_numbers(by_position, positions, previous)
    targets = _plan_targets(
        by_position,
        positions,
        previous,
        roi_numbers,
        naming_template=naming_template,
        output_dir=output_dir,
        settings=settings,
        on_existing=on_existing,
        verify_existing=verify_existing,
        size_t_of=lambda p: _truncated_size_t(seqfile, p, max_frames),
        release=lambda p: _release_position(seqfile, p),
        say=say,
    )
    todo = {p: [t for t in targets[p] if t.export] for p in positions}
    skipped = sum(len(targets[p]) - len(todo[p]) for p in positions)
    positions = [p for p in positions if todo[p]]
    if skipped:
        say(
            f"{skipped} ROI(s) already exported, skipped "
            "(pass on_existing='overwrite' to redo them)"
        )
    if not positions:
        # Nothing to crop, but the folder-level description is still refreshed:
        # a re-run may carry a new `experiment` name, and the recorded ROIs are
        # merged through untouched.
        _write_export_info(
            output_dir,
            manifest,
            [],
            experiment=experiment,
            naming_template=naming_template,
            registration_method_name=registration_method_name,
            max_frames=max_frames,
        )
        say(f"nothing to export -- {output_dir!r} is already up to date")
        return []

    reg_dash = None
    reg_bar = None
    if registration_method_name is not None:
        # Before any registration work: an ROI drawn outside the range about to be
        # registered can never be placed on the corrected sequence.
        _check_anchor_frames(
            {p: [t.sel for t in todo[p]] for p in positions},
            positions,
            lambda p: min(seqfile.position(p).size_t, max_frames or 1 << 62),
        )
        reg_dash, reg_bar = _make_registration(
            seqfile,
            registration_method_name,
            method_kwargs,
            reference_mode,
            low_confidence,
            progress,
        )

    written: list[str] = []
    exported: list[dict] = []
    try:
        for position in tqdm(
            positions, desc="positions", unit="pos", disable=not progress
        ):
            group = todo[position]
            pos_source = seqfile.position(position)
            if max_frames is not None:
                pos_source = pos_source[:max_frames]
            raw_pos_source = pos_source

            record = None
            if reg_dash is not None:
                record = _register_position(
                    reg_dash,
                    position,
                    pos_source,
                    by_position[position],
                    output_dir,
                    exclude_rois,
                    exclude_shrink_px,
                    registration_method_name,
                    method_kwargs,
                    say,
                    detail,
                )
                pos_source = pos_source.register(
                    record.transforms, on_missing=on_missing
                )

            # Full-frame (H, W) for the anchor mapping, read off pos_source: a
            # correction never changes frame dimensions, while the manifest's
            # source block is the *file's* shape and would disagree whenever
            # `source=` points at a different file.
            frame_shape = (pos_source.size_h, pos_source.size_w)
            for target in group:
                target.spec = target.sel.spec
                if record is not None:
                    target.spec, note = _spec_for_registered(
                        target.sel, record, frame_shape
                    )
                    if note:
                        detail(f"  pos {position + 1} roi {target.roi_nb}: {note}")

            _export_position(
                pos_source,
                raw_pos_source,
                group,
                size_t=int(pos_source.size_t),
                chunk_frames=int(chunk_frames),
                max_roi_buffer_bytes=max_roi_buffer_bytes,
                ome=ome,
                compression=compression,
                channel_names=manifest.source.get("channels"),
                progress=progress,
                position=position,
                on_written=lambda target, frames, position=position: _record_written(
                    target,
                    frames,
                    position=position,
                    settings=settings,
                    written=written,
                    exported=exported,
                    output_dir=output_dir,
                    manifest=manifest,
                    experiment=experiment,
                    naming_template=naming_template,
                    registration_method_name=registration_method_name,
                    max_frames=max_frames,
                ),
            )

            missing = sorted(getattr(pos_source, "_missing", ()) or ())
            if missing:
                shown = ", ".join(str(t) for t in missing[:5])
                more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
                action = (
                    "corrected with the nearest frame's transform"
                    if on_missing == "nearest"
                    else "exported uncorrected"
                )
                say(
                    f"pos {position + 1}: {len(missing)} frame(s) had no transform "
                    f"of their own and were {action}: {shown}{more}"
                )
            # Nothing below reads this position again; dropping it closes the
            # reader instead of keeping one open per position for the whole run.
            del pos_source, raw_pos_source
            _release_position(seqfile, position)
    finally:
        if reg_bar is not None:
            reg_bar.close()
        if reg_dash is not None:
            # An un-closed widget keeps its per-position records -- and through
            # `self._file` the whole SequenceFile -- alive for the kernel's life.
            with contextlib.suppress(Exception):
                reg_dash.close()

    note = f" (registered via {registration_method_name})" if reg_dash else ""
    say(f"{len(written)} TIFF(s) written to {output_dir!r}{note}")
    return written


# --------------------------------------------------------------------------- #
# what to export: stable numbering, and skipping what is already there
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class _Target:
    """One selection's export slot: where it goes and whether it still needs doing."""

    sel: Any
    roi_nb: int
    filename: str
    path: str
    export: bool = True
    spec: Any = None


def _export_settings(max_frames, registration_method_name, ome, compression) -> dict:
    """The settings that change an exported ROI's pixels, as recorded per entry.

    An existing file is only trusted while these still match -- changing
    ``max_frames`` or the compression has to re-export, where merely re-running
    the cell must not.
    """
    return {
        "max_frames": max_frames,
        "registration": registration_method_name,
        "ome": bool(ome),
        "compression": compression,
        "crop_pipeline": CROP_PIPELINE,
    }


def _previous_entries(output_dir) -> dict[str, dict]:
    """``{selection id: entry}`` from an existing ``roi_export.json``."""
    return {
        str(entry["id"]): entry
        for entry in _read_export_rois(output_dir)
        if entry.get("id")
    }


def _read_export_rois(output_dir) -> list[dict]:
    """The ``rois`` list of ``roi_export.json``, or ``[]`` if there is none."""
    try:
        rois = json.loads((Path(output_dir) / EXPORT_INFO_NAME).read_text()).get("rois")
    except (OSError, ValueError, AttributeError):
        return []
    return list(rois or [])


def _assign_roi_numbers(by_position, positions, previous) -> dict[str, int]:
    """``{selection id: roi_nb}``, keeping the numbers an earlier run handed out.

    ``roi_nb`` goes into the file name, so re-numbering by manifest order (what
    a plain ``enumerate`` does) would quietly reassign existing files the moment
    a new ROI is inserted before an old one. A selection that already has an
    entry keeps its number; a new one takes the lowest number still free in its
    position. With no previous export this is exactly ``1..N`` in manifest order.
    """
    numbers: dict[str, int] = {}
    for position in positions:
        used: set[int] = set()
        pending = []
        for sel in by_position[position]:
            entry = previous.get(str(sel.id))
            recorded = entry.get("roi_nb") if entry else None
            if (
                recorded
                and entry.get("position_index") == position
                and int(recorded) not in used
            ):
                numbers[str(sel.id)] = int(recorded)
                used.add(int(recorded))
            else:
                pending.append(sel)
        nxt = 1
        for sel in pending:
            while nxt in used:
                nxt += 1
            numbers[str(sel.id)] = nxt
            used.add(nxt)
    return numbers


def _plan_targets(
    by_position,
    positions,
    previous,
    roi_numbers,
    *,
    naming_template,
    output_dir,
    settings,
    on_existing,
    verify_existing,
    size_t_of,
    release,
    say,
) -> dict[int, list[_Target]]:
    """Decide, per selection, whether it still has to be exported.

    Two passes, because the cheap one is worth doing first: a selection with a
    matching entry in ``roi_export.json`` is resolved without touching the raw
    file at all, so a position whose ROIs are all done is never opened, never
    registered and never read. Only a position that still holds something
    unresolved -- a file written by a run that died before it could record it,
    or by a pre-``v2`` export -- costs a metadata read to learn its frame count.
    """
    plan: dict[int, list[_Target]] = {}
    unresolved: dict[int, list[_Target]] = {}
    for position in positions:
        plan[position] = []
        for sel in by_position[position]:
            roi_nb = roi_numbers[str(sel.id)]
            filename = naming_template.format(
                position=position + 1,
                position_index=position,
                roi_nb=roi_nb,
                label=sel.label,
                id=sel.id,
            )
            target = _Target(
                sel=sel,
                roi_nb=roi_nb,
                filename=filename,
                path=os.path.join(output_dir, filename),
            )
            plan[position].append(target)

            if on_existing == "overwrite" or not os.path.exists(target.path):
                continue
            if on_existing == "error":
                raise ValueError(
                    f"{target.path} already exists and on_existing='error'. "
                    "Pass on_existing='skip' to keep it, or 'overwrite' to redo it."
                )
            state = _recorded_state(
                target, previous.get(str(sel.id)), settings, verify_existing
            )
            if state == "skip":
                target.export = False
            elif state == "unresolved":
                unresolved.setdefault(position, []).append(target)

    for position, pending in unresolved.items():
        size_t = size_t_of(position)
        for target in pending:
            if not verify_existing or _tiff_frames(target.path) == size_t:
                target.export = False
                say(
                    f"pos {position + 1} roi {target.roi_nb}: file is there but not "
                    f"described in {EXPORT_INFO_NAME} (older or interrupted export); "
                    "keeping it -- pass on_existing='overwrite' to redo it"
                )
        if not any(target.export for target in plan[position]):
            # Learning the frame count above opened this position; nothing is
            # going to read it after all, so let go of the handle again.
            release(position)
    return plan


def _recorded_state(target, entry, settings, verify_existing) -> str:
    """``'skip'``, ``'export'`` or ``'unresolved'`` from the recorded entry alone.

    ``'unresolved'`` means the file is there but nothing describes it well
    enough to judge -- the caller settles those once it knows the frame count.
    """
    if entry is None or entry.get("file") != target.filename or "frames" not in entry:
        return "unresolved"
    if entry.get("spec") != target.sel.spec.to_dict():
        return "export"
    if int(entry.get("anchor_frame", 0)) != int(getattr(target.sel, "anchor_frame", 0)):
        return "export"
    if entry.get("settings") != settings:
        return "export"
    if verify_existing and _tiff_frames(target.path) != entry["frames"]:
        return "export"
    return "skip"


def _tiff_frames(path) -> int | None:
    """Frame count from a TIFF's header -- no pixel data is read.

    ``None`` when the file cannot be read at all, which is what a run killed
    mid-write leaves behind.
    """
    import tifffile

    try:
        with tifffile.TiffFile(os.fspath(path)) as tif:
            series = tif.series[0]
            axes, shape = series.axes, series.shape
            if "T" in axes:
                return int(shape[axes.index("T")])
            return int(shape[0]) if len(shape) >= 3 else 1
    except Exception:  # noqa: BLE001 - an unreadable file simply has no frame count
        return None


def _truncated_size_t(seqfile, position, max_frames) -> int:
    """A position's exported frame count (metadata only, no pixels)."""
    size_t = int(seqfile.position(position).size_t)
    return min(size_t, max_frames) if max_frames is not None else size_t


def _release_position(seqfile, position) -> None:
    """Best-effort: drop one position's cached reader from the ``SequenceFile``."""
    release = getattr(seqfile, "release", None)
    if callable(release):
        with contextlib.suppress(Exception):
            release(position)


# --------------------------------------------------------------------------- #
# writing one position: one read and one correction per frame, for every ROI
# --------------------------------------------------------------------------- #
def _export_position(
    source,
    raw_source,
    targets,
    *,
    size_t,
    chunk_frames,
    max_roi_buffer_bytes,
    ome,
    compression,
    channel_names,
    progress,
    position,
    on_written,
) -> None:
    """Write this position's ROIs, reading each frame once and warping it once.

    The ROIs of a position are cropped out of the *same* frame while it is in
    hand, so a position costs one pass over its frames however many ROIs it has
    -- where a pass per ROI would read the position again for each one. The
    price is holding each ROI's stack until it is written, which is what
    ``max_roi_buffer_bytes`` bounds: the ROIs are split into as many passes as it
    takes to stay inside the budget.

    Frames are read **uncorrected** and each ROI's rectangle is carried back
    into that frame's own coordinates (:meth:`~acia.base.RegisteredSequenceSource.spec_for`)
    rather than correcting the frame and then cropping the corrected copy. The
    two warps compose exactly, so it is the same crop -- out of one
    interpolation instead of two, which is sharper, and without the full-frame
    warp, which on a large frame is most of the per-frame cost.
    """
    from acia.base import warp_rotated_crop
    from acia.segm.local import THWCSequenceSource

    # None when nothing is being corrected: then a spec needs no mapping.
    registered = source if source is not raw_source else None

    # Shape/dtype are invariant under the correction, so probe the cheaper
    # (unregistered) source -- one frame, once per position.
    probe = np.asarray(raw_source.get_frame(0).raw)
    n_channels = 1 if probe.ndim == 2 else int(probe.shape[2])
    dtype = probe.dtype
    del probe

    timepoints = source.timepoints
    pixel_size = source.pixel_size

    groups = _roi_groups(
        [t.spec for t in targets],
        size_t,
        n_channels,
        dtype.itemsize,
        max_roi_buffer_bytes,
    )
    bar = tqdm(
        total=size_t * len(groups),
        desc=f"cropping pos {position + 1}",
        unit="frame",
        leave=False,
        disable=not progress,
    )
    try:
        with warnings.catch_warnings():
            # one warning per frame without its own transform -- they are
            # counted once per position by the caller instead
            warnings.filterwarnings(
                "ignore",
                message="RegisteredSequenceSource: no stored correction",
            )
            for indices in groups:
                group = [targets[i] for i in indices]
                if len(group) == 1 and _roi_bytes(
                    group[0].spec, size_t, n_channels, dtype.itemsize
                ) > (max_roi_buffer_bytes or float("inf")):
                    # Too big to buffer even on its own: stream it lazily.
                    # `crop_rotated` on a corrected view composes the two warps
                    # too, so this writes the same pixels as the buffered path.
                    _write_roi(
                        source.crop_rotated(group[0].spec),
                        group[0],
                        ome=ome,
                        compression=compression,
                        channel_names=channel_names,
                    )
                    on_written(group[0], size_t)
                    bar.update(size_t)
                    continue

                buffers = [
                    np.empty(
                        (size_t, int(t.spec.size[1]), int(t.spec.size[0]), n_channels),
                        dtype=dtype,
                    )
                    for t in group
                ]
                # The spec mapping goes through `registered.spec_for`, which is
                # also what fills its `_missing` set -- composing the transform
                # by hand here would silently empty the per-position "no
                # transform" report (and be a third place for that matrix).
                for t, raw in _read_frames(raw_source, size_t, chunk_frames):
                    for buffer, target in zip(buffers, group, strict=True):
                        spec = (
                            target.spec
                            if registered is None
                            else registered.spec_for(t, target.spec)
                        )
                        buffer[t] = _as_hwc(warp_rotated_crop(raw, spec))
                    bar.update(1)

                for i, target in enumerate(group):
                    _write_roi(
                        THWCSequenceSource(
                            buffers[i], timepoints=timepoints, pixel_size=pixel_size
                        ),
                        target,
                        ome=ome,
                        compression=compression,
                        channel_names=channel_names,
                    )
                    buffers[i] = None  # type: ignore[call-overload]
                    on_written(target, size_t)
    finally:
        bar.close()


def _roi_bytes(spec, size_t: int, n_channels: int, itemsize: int) -> int:
    """How much RAM one ROI's buffered stack takes."""
    return (
        int(size_t)
        * int(spec.size[1])
        * int(spec.size[0])
        * int(n_channels)
        * int(itemsize)
    )


def _roi_groups(specs, size_t, n_channels, itemsize, budget) -> list[list[int]]:
    """Partition ROI indices into consecutive groups that each fit ``budget``.

    Consecutive and in manifest order, so the export order (and with it the
    order of ``roi_export.json``) is the same as it has always been. An ROI that
    does not fit on its own ends up alone in its group; the caller streams it.
    """
    if budget is None:
        return [list(range(len(specs)))]
    groups: list[list[int]] = []
    current: list[int] = []
    current_bytes = 0
    for index, spec in enumerate(specs):
        size = _roi_bytes(spec, size_t, n_channels, itemsize)
        if current and current_bytes + size > budget:
            groups.append(current)
            current, current_bytes = [], 0
        current.append(index)
        current_bytes += size
    if current:
        groups.append(current)
    return groups


def _as_hwc(frame: np.ndarray) -> np.ndarray:
    """``(h, w)`` -> ``(h, w, 1)``; anything already channel-last passes through."""
    return frame[..., None] if frame.ndim == 2 else frame


def _read_frames(source, size_t, depth):
    """Yield ``(t, raw)`` for every frame, reading up to ``depth`` of them ahead.

    The crop pass reads each frame once and then spends real time warping it for
    each ROI; with a network-mounted acquisition that read is latency, not work.
    One reader thread (never more: the CZI/ND2 readers are a single unlocked file
    handle, so concurrent reads are not safe) keeps ``depth`` frames in flight so
    the next read happens while the current frame is being cropped.
    """
    if depth <= 1:
        for t in range(size_t):
            yield t, np.asarray(source.get_frame(t).raw)
        return

    done = object()
    frames: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def _read() -> None:
        try:
            for t in range(size_t):
                if stop.is_set():
                    break
                item = (t, np.asarray(source.get_frame(t).raw))
                while not stop.is_set():
                    try:
                        frames.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer
            with contextlib.suppress(queue.Full):
                frames.put(exc, timeout=0.1)
        finally:
            with contextlib.suppress(queue.Full):
                frames.put(done, timeout=0.1)

    reader = threading.Thread(target=_read, name="acia-roi-export-reader", daemon=True)
    reader.start()
    try:
        while True:
            item = frames.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # A consumer that stops early (an exception, a break) must not leave the
        # reader blocked on a full queue.
        stop.set()
        while reader.is_alive():
            try:
                frames.get_nowait()
            except queue.Empty:
                reader.join(timeout=0.05)


def _write_roi(source, target, *, ome, compression, channel_names) -> None:
    """Write one ROI, via a temporary file so a kill never leaves half a TIFF.

    ``save_tiff_stack`` streams straight into the target path, so a run that
    dies mid-write used to leave a truncated file that looks finished to
    everything downstream. Writing beside it and renaming makes the final path
    appear only once it is complete (``os.replace`` is atomic within a
    filesystem, and the temporary is in the same directory).
    """
    from acia.segm.tiff_export import save_tiff_stack

    partial = _part_path(target.path)
    save_tiff_stack(
        source, partial, ome=ome, compression=compression, channel_names=channel_names
    )
    os.replace(partial, target.path)


def _part_path(path: str) -> str:
    """``.../pos001_roi001.tiff`` -> ``.../pos001_roi001.part.tiff``."""
    root, ext = os.path.splitext(path)
    return f"{root}{PART_SUFFIX}{ext}"


def _clean_part_files(output_dir) -> None:
    """Remove leftovers of a run that was killed mid-write."""
    for leftover in Path(output_dir).rglob(f"*{PART_SUFFIX}.*"):
        with contextlib.suppress(OSError):
            leftover.unlink()


def _record_written(
    target,
    frames,
    *,
    position,
    settings,
    written,
    exported,
    output_dir,
    manifest,
    experiment,
    naming_template,
    registration_method_name,
    max_frames,
) -> None:
    """Note one finished ROI, and re-write ``roi_export.json`` right away.

    Per ROI rather than once at the end: the file is what a later run reads to
    decide what is already done, so a run that is interrupted has to leave
    behind an accurate record of everything it did finish.
    """
    written.append(target.path)
    exported.append(
        {
            "file": os.path.basename(target.path),
            "position": position + 1,
            "position_index": position,
            "roi_nb": target.roi_nb,
            "label": target.sel.label,
            "id": target.sel.id,
            "anchor_frame": getattr(target.sel, "anchor_frame", 0),
            "spec": target.sel.spec.to_dict(),
            "frames": int(frames),
            "settings": settings,
        }
    )
    _write_export_info(
        output_dir,
        manifest,
        exported,
        experiment=experiment,
        naming_template=naming_template,
        registration_method_name=registration_method_name,
        max_frames=max_frames,
    )


def _refuse_foreign_export(output_dir: str, source_path: str) -> None:
    """Fail before writing if ``output_dir`` holds another acquisition's export.

    ROI files are named by position and ROI number only, so exporting a second
    raw file into the same folder would silently overwrite the first one's
    ``pos001_roi001.tiff`` -- and mix both in one analysis.
    """
    try:
        info = json.loads((Path(output_dir) / EXPORT_INFO_NAME).read_text())
    except (OSError, ValueError):
        return
    previous = str((info.get("source") or {}).get("path") or "")
    if previous and source_path and Path(previous).name != Path(source_path).name:
        raise ValueError(
            f"{output_dir} already holds the ROI export of {Path(previous).name}; "
            f"exporting {Path(source_path).name} there would overwrite its files. "
            "Choose another output_dir for this acquisition (e.g. a subfolder "
            "named after it)."
        )


def _write_export_info(
    output_dir,
    manifest,
    exported,
    *,
    experiment,
    naming_template,
    registration_method_name,
    max_frames,
) -> Path:
    """Write ``roi_export.json``, merging with the ROIs of earlier exports.

    Exporting in several runs into one folder (e.g. a position at a time, or one
    newly drawn ROI) keeps every file's entry; a file -- or a selection --
    written again replaces its old entry.
    """
    path = Path(output_dir) / EXPORT_INFO_NAME
    source = dict(manifest.source)
    source_path = str(source.get("path", ""))
    previous = _read_export_rois(output_dir)
    files = {e["file"] for e in exported}
    ids = {str(e["id"]) for e in exported if e.get("id")}
    rois = [
        e
        for e in previous
        if e.get("file") not in files and str(e.get("id")) not in ids
    ] + exported

    info = {
        "schema": EXPORT_INFO_SCHEMA,
        "experiment": experiment or Path(source_path).stem or Path(output_dir).name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "path": source_path,
            "name": Path(source_path).name,
            "format": source.get("format"),
            "pixel_size_um": source.get("pixel_size_um"),
            "frame_interval_s": source.get("frame_interval_s"),
            "channels": source.get("channels"),
        },
        "naming_template": naming_template,
        "registration": registration_method_name,
        "max_frames": max_frames,
        "rois": sorted(rois, key=lambda e: (e.get("position", 0), e.get("roi_nb", 0))),
    }
    path.write_text(json.dumps(info, indent=2, default=str))
    return path


def _make_registration(
    seqfile, method_name, method_kwargs, reference_mode, low_confidence, progress
):
    """A ``RegistrationDashboard`` plus a tqdm bar fed by its progress messages.

    The dashboard has no plain Python progress callback, so its ``"progress"``
    widget messages are intercepted -- which also means the bar ticks per frame
    without the widget ever being displayed.
    """
    from acia.notebook import RegistrationDashboard

    dash = RegistrationDashboard(
        seqfile,
        method_name=method_name,
        method_kwargs=method_kwargs or {},
        reference_mode=reference_mode,
        low_confidence=low_confidence,
    )
    if not progress:
        return dash, None

    bar = tqdm(desc="registration", unit="frame", leave=False)
    original_send = dash.send
    shown_position = [None]
    last_sent = [0.0]

    def _send_with_progress(msg, *args, **kwargs):
        if msg.get("type") == "progress" and "position" in msg:
            new_position = shown_position[0] != msg["position"]
            if new_position:
                bar.reset(total=msg["num_frames"])
                bar.set_description(
                    f"registering pos {msg['position'] + 1} "
                    f"({msg['position'] + 1}/{msg['num_positions']})"
                )
                shown_position[0] = msg["position"]
            bar.n = msg["frame"] + 1
            bar.refresh()
            # The dashboard is never displayed here, so its per-frame message
            # only feeds the bar above -- but it is still a widget comm message
            # per frame, queued in the kernel whether or not anything reads it.
            # One every PROGRESS_INTERVAL_S is plenty for a progress bar.
            now = time.monotonic()
            last = msg["frame"] + 1 >= msg["num_frames"]
            if not (new_position or last or now - last_sent[0] >= PROGRESS_INTERVAL_S):
                return None
            last_sent[0] = now
        return original_send(msg, *args, **kwargs)

    dash.send = _send_with_progress
    return dash, bar


def _register_position(
    reg_dash,
    position,
    pos_source,
    sels,
    output_dir,
    exclude_rois,
    exclude_shrink_px,
    method_name,
    method_kwargs,
    say,
    detail=None,
):
    """Register one position and return its record, reporting how it went.

    Per-frame low-confidence warnings are silenced here: the one report line
    per position (:func:`_report_registration`) already counts them and names
    the worst frame.
    """
    from acia.registration import LowConfidenceWarning

    detail = detail or (lambda *a, **k: None)
    pos_kwargs = None
    if exclude_rois:
        anchors = sorted({getattr(s, "anchor_frame", 0) for s in sels})
        if any(anchors):
            detail(
                f"pos {position + 1}: exclusion rectangles use as-drawn coordinates "
                f"(anchor frames {anchors}); they are approximate by the drift at "
                "those frames."
            )
        pos_kwargs = {
            "exclude_rects": [s.spec for s in sels],
            "exclude_shrink_px": exclude_shrink_px,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LowConfidenceWarning)
        reg_dash.batch_apply(
            directory=output_dir,
            positions=[position],
            sources={position: pos_source},
            method_kwargs=pos_kwargs,
        )
    record = next(r for r in reg_dash.manifest.records if r.position == position)
    _report_registration(
        position,
        record,
        threshold=_confidence_threshold(method_name, method_kwargs),
        say=say,
    )
    return record


def _confidence_threshold(method_name: str, method_kwargs: dict | None):
    """The confidence gate this run actually used, or ``None`` if there isn't one.

    Read from the method's own signature when it wasn't overridden, so a report
    never disagrees with acia's default by duplicating the number here.
    """
    import inspect

    import acia.registration

    param = _CONFIDENCE_GATE.get(method_name)
    if param is None:
        return None
    if method_kwargs and param in method_kwargs:
        return method_kwargs[param]
    cls = getattr(acia.registration, method_name)
    return inspect.signature(cls).parameters[param].default


def _check_anchor_frames(by_position, positions, size_t_of) -> None:
    """Fail fast when an ROI was drawn on a frame the export will not register.

    ``max_frames`` slices each position *before* registration, so the transforms
    only cover the truncated range. The curation dashboard opens mid-sequence, so
    ROIs typically sit around ``T // 2`` -- which a quick ``max_frames=30`` run
    would never reach. Checking every anchor up front turns a twenty-minute
    registration run that ends in an error into a one-second one.
    """
    bad = []
    for position in positions:
        size_t = size_t_of(position)
        for sel in by_position[position]:
            anchor = getattr(sel, "anchor_frame", 0)
            if anchor >= size_t:
                bad.append(
                    f"  pos {position + 1} {sel.label!r}: drawn on frame {anchor}, but "
                    f"only frames 0..{size_t - 1} will be registered"
                )
    if bad:
        raise ValueError(
            "These ROIs were annotated on frames outside the range being "
            "exported, so their crops cannot be placed after drift correction:\n"
            + "\n".join(bad)
            + "\n\nRaise max_frames past the highest anchor, or re-draw those ROIs "
            "on an earlier frame."
        )


def _spec_for_registered(sel, record, shape):
    """``sel.spec`` moved onto the registered (reference-frame) view.

    Returns ``(spec, note)``; ``note`` is ``None`` when there was nothing to say.
    """
    from acia.registration import apply_correction_to_spec

    anchor = getattr(sel, "anchor_frame", 0)
    if anchor == record.reference_frame:
        # Nothing to do -- and deliberately not routed through
        # transforms[reference_frame]: batch_apply *estimates* the reference
        # frame's own transform rather than storing an exact identity, so mapping
        # through it would nudge every frame-0 ROI by a fraction of a pixel.
        return sel.spec, None

    transform = record.transforms.get(anchor)
    if transform is None:
        raise ValueError(
            f"pos {record.position} {sel.label!r}: no transform for anchor frame "
            f"{anchor}, so the ROI cannot be placed on the corrected sequence. "
            "Registration failed on that frame "
            f"({record.failed_frames.get(anchor, 'reason unknown')})."
        )
    return apply_correction_to_spec(sel.spec, transform, shape=shape), (
        f"anchor frame {anchor}: ROI moved by "
        f"({-transform.dx:+.1f}, {-transform.dy:+.1f}) px, "
        f"{-transform.theta:+.2f} deg onto the corrected view"
    )


def _report_registration(position, record, threshold=None, say=tqdm.write) -> None:
    """One line per position summarizing registration quality.

    A steadily falling confidence means the imaged content is changing, not that
    the fit is going wrong -- worth seeing on the run rather than discovering it
    later in ``registration_transforms.json``. Under ``low_confidence='keep'`` a
    fit below ``threshold`` is kept rather than dropped, so this line is the only
    place a weak run announces itself.
    """
    confs = [
        t.confidence for t in record.transforms.values() if t.confidence is not None
    ]
    reanchors = sorted(set(record.reference_frames.values()))
    parts = [f"{len(record.transforms)} frame(s) registered"]
    if confs:
        parts.append(f"confidence {min(confs):.3f}-{max(confs):.3f}")
    if threshold is not None:
        weak = {
            t: tr.confidence
            for t, tr in record.transforms.items()
            if tr.confidence is not None and tr.confidence < threshold
        }
        if weak:
            worst = min(weak, key=lambda t: weak[t])
            parts.append(
                f"{len(weak)} below {threshold} (kept; worst "
                f"{weak[worst]:.3f} at frame {worst})"
            )
    if record.failed_frames:
        failed = sorted(record.failed_frames)
        shown = ", ".join(str(t) for t in failed[:5])
        more = f" (+{len(failed) - 5} more)" if len(failed) > 5 else ""
        parts.append(f"{len(failed)} FAILED: {shown}{more}")
    if reanchors:
        parts.append(f"re-anchored at {reanchors}")
    say(f"pos {position + 1}: " + "; ".join(parts))


# --------------------------------------------------------------------------- #
# reviewing an export: one ROI -> a small preview video
# --------------------------------------------------------------------------- #
PREVIEW_FOLDER = ".previews"


def export_entries(export_dir: str | os.PathLike) -> list[dict]:
    """The exported ROIs of a folder, from ``roi_export.json`` (glob as fallback).

    Each entry has at least ``file``; entries from ``roi_export.json`` also
    carry ``position`` (1-based), ``roi_nb`` and ``label``.
    """
    folder = Path(export_dir)
    try:
        rois = json.loads((folder / EXPORT_INFO_NAME).read_text()).get("rois") or []
        rois = [r for r in rois if (folder / r.get("file", "")).is_file()]
    except (OSError, ValueError):
        rois = []
    if rois:
        return rois
    return [{"file": p.name} for p in sorted(folder.glob("pos*_roi*.tif*"))]


def render_roi_preview(
    tiff: str | os.PathLike,
    video: str | os.PathLike,
    *,
    fps: int = 10,
    max_size: int = 512,
    channel: int = 0,
) -> dict:
    """Render an exported ROI stack to a small mp4 for review.

    The contrast is fixed over the whole movie (1st-99.5th percentile of a
    sample of frames), so brightness changes stay visible instead of being
    normalized away frame by frame. Frames larger than ``max_size`` are scaled
    down (keeping the aspect ratio). Rendering a typical ROI takes well under a
    second.

    Returns:
        ``{"frames", "height", "width", "channels", "frame_interval", "seconds"}``
        describing the stack and how long the render took.
    """
    import cv2
    import imageio.v2 as iio

    from acia.segm.open import open_sequence

    t0 = time.perf_counter()
    seqfile = open_sequence(os.fspath(tiff))
    source = seqfile.position(0)
    n = int(source.size_t)

    def plane(i):
        raw = np.asarray(source.get_frame(i).raw)
        return raw[..., channel] if raw.ndim == 3 else raw

    sample = np.stack([plane(int(i)) for i in np.linspace(0, n - 1, min(n, 20))])
    lo, hi = np.percentile(sample, [1, 99.5])
    hi = hi if hi > lo else lo + 1
    first = plane(0)
    height, width = first.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    size = (
        max(2, int(round(width * scale)) // 2 * 2),
        max(2, int(round(height * scale)) // 2 * 2),
    )

    Path(video).parent.mkdir(parents=True, exist_ok=True)
    with iio.get_writer(
        os.fspath(video),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
        output_params=["-movflags", "+faststart"],
        ffmpeg_params=["-preset", "veryfast"],
    ) as writer:
        for i in range(n):
            img = np.clip((plane(i).astype(np.float32) - lo) / (hi - lo) * 255, 0, 255)
            img = cv2.resize(img.astype(np.uint8), size, interpolation=cv2.INTER_AREA)
            writer.append_data(np.repeat(img[:, :, None], 3, axis=-1))  # type: ignore[attr-defined]

    timepoints = source.timepoints
    interval = None
    if timepoints is not None and len(timepoints) > 1:
        step = (timepoints[1] - timepoints[0]).to_compact()
        interval = f"{step.magnitude:g} {step.units:~P}"
    info = {
        "frames": n,
        "height": int(height),
        "width": int(width),
        "channels": int(getattr(source, "num_channels", 1) or 1),
        "frame_interval": interval,
        "seconds": time.perf_counter() - t0,
    }
    # A TIFF source caches its whole decoded stack, so reviewing a folder of
    # ROIs would hold one stack per preview until the GC got to it; closing the
    # handle drops the cache of the position source it opened.
    seqfile.close()
    return info


def preview_path(
    export_dir: str | os.PathLike, file: str, *, fps: int, max_size: int
) -> Path:
    """Where the cached preview of ``file`` lives (``<export_dir>/.previews/``)."""
    stem = Path(file).name.split(".")[0]
    return Path(export_dir) / PREVIEW_FOLDER / f"{stem}_{fps}fps_{max_size}px.mp4"
