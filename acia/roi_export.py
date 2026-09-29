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
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

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
EXPORT_INFO_SCHEMA = "acia.roi_export/v1"

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
) -> list[str]:
    """Crop every selection to its own TIFF, optionally registering each position.

    Args:
        manifest: the :class:`~acia.selection.SelectionManifest` to export.
        output_dir: where the TIFFs (and, when registering,
            ``registration_transforms.json``) are written; created if missing.
        naming_template: filename pattern, formatted with ``position``
            (**1-based**, as the microscope software numbers positions),
            ``position_index`` (0-based, as stored in the manifest), ``roi_nb``
            (1-based within the position, in manifest order), ``label`` and
            ``id``.
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

    Returns:
        The written paths, in export order.

    Raises:
        ValueError: if an ROI was drawn on a frame outside the range being
            exported (checked up front, before any registration work), or if a
            drawn-on frame has no transform after registration.
    """
    from acia.segm.acquisition_fixes import manifest_fixes
    from acia.segm.open import open_sequence
    from acia.segm.tiff_export import save_tiff_stack

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

    reg_dash = None
    reg_bar = None
    if registration_method_name is not None:
        # Before any registration work: an ROI drawn outside the range about to be
        # registered can never be placed on the corrected sequence.
        _check_anchor_frames(
            by_position,
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

    _refuse_foreign_export(output_dir, manifest.source_path)
    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []
    exported: list[dict] = []
    try:
        for position in tqdm(
            positions, desc="positions", unit="pos", disable=not progress
        ):
            sels = by_position[position]
            pos_source = seqfile.position(position)
            if max_frames is not None:
                pos_source = pos_source[:max_frames]

            record = None
            if reg_dash is not None:
                # Read every frame from disk exactly once: the registration loop
                # and the crop/write pass below both read from this in-memory copy
                # instead of hitting disk twice (both the registered and the
                # cropped view are lazy, and call get_frame() on their parent).
                pos_source = pos_source.materialize()
                record = _register_position(
                    reg_dash,
                    position,
                    pos_source,
                    sels,
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
            roi_bar = tqdm(
                sels,
                desc=f"ROIs (pos {position + 1})",
                unit="roi",
                leave=False,
                disable=not progress,
            )
            for roi_nb, sel in enumerate(roi_bar, start=1):
                spec = sel.spec
                if record is not None:
                    spec, note = _spec_for_registered(sel, record, frame_shape)
                    if note:
                        detail(f"  pos {position + 1} roi {roi_nb}: {note}")
                filename = naming_template.format(
                    position=position + 1,
                    position_index=position,
                    roi_nb=roi_nb,
                    label=sel.label,
                    id=sel.id,
                )
                with warnings.catch_warnings():
                    # one warning per frame without its own transform -- they
                    # are counted once per position below instead
                    warnings.filterwarnings(
                        "ignore",
                        message="RegisteredSequenceSource: no stored correction",
                    )
                    path = save_tiff_stack(
                        pos_source.crop_rotated(spec),
                        os.path.join(output_dir, filename),
                        ome=ome,
                        compression=compression,
                        channel_names=manifest.source.get("channels"),
                    )
                written.append(path)
                exported.append(
                    {
                        "file": os.path.basename(path),
                        "position": position + 1,
                        "position_index": position,
                        "roi_nb": roi_nb,
                        "label": sel.label,
                        "id": sel.id,
                        "anchor_frame": getattr(sel, "anchor_frame", 0),
                    }
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
    finally:
        if reg_bar is not None:
            reg_bar.close()

    _write_export_info(
        output_dir,
        manifest,
        exported,
        experiment=experiment,
        naming_template=naming_template,
        registration_method_name=registration_method_name,
        max_frames=max_frames,
    )

    note = f" (registered via {registration_method_name})" if reg_dash else ""
    say(f"{len(written)} TIFF(s) written to {output_dir!r}{note}")
    return written


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

    Exporting in several runs into one folder (e.g. a position at a time) keeps
    every file's entry; a file written again replaces its old entry.
    """
    path = Path(output_dir) / EXPORT_INFO_NAME
    source = dict(manifest.source)
    source_path = str(source.get("path", ""))
    try:
        previous = json.loads(path.read_text()).get("rois", [])
    except (OSError, ValueError):
        previous = []
    files = {e["file"] for e in exported}
    rois = [e for e in previous if e.get("file") not in files] + exported

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

    def _send_with_progress(msg, *args, **kwargs):
        if msg.get("type") == "progress" and "position" in msg:
            if shown_position[0] != msg["position"]:
                bar.reset(total=msg["num_frames"])
                bar.set_description(
                    f"registering pos {msg['position'] + 1} "
                    f"({msg['position'] + 1}/{msg['num_positions']})"
                )
                shown_position[0] = msg["position"]
            bar.n = msg["frame"] + 1
            bar.refresh()
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
