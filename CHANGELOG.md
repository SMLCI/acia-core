# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Add your changes under **[Unreleased]**; the release workflow promotes that
section to a dated version entry automatically.

Entries for 0.3.2 and earlier were reconstructed from the git history after the
fact and are summarised rather than exhaustive; from 0.4.0 on they are written
as the work lands.

## [Unreleased]

### Changed

**Per-cell growth rates come with their uncertainty, and cells are flagged by what the
measurement can resolve** (`acia.analysis.growth_rate.compute_cell_growth_rates`):
- `compute_doubling_times` adds `doubling_time_lo`/`doubling_time_hi` (the narrowest and
  widest cycle the frames allow: birth and division are each only known to within one
  frame interval), `doubling_time_sd` (`Δ/√6` for equal frames) and `frame_gap` (frames
  between the mother's last and the daughters' first observation).
- The size fit adds `mu_fit_ci_lo`/`mu_fit_ci_hi`, optional HAC (Newey-West) standard
  errors (`se_type='hac'`), the smallest/largest frame-to-frame size ratio, and
  `mu_from_dt_se`/`_lo`/`_hi` from the frame timing.
- **Flags changed.** A cell is now flagged when its two rates differ by more than
  `agreement_z` (default 2) combined standard errors (`agreement_z` column), or when its
  size drops below `size_jump_ratio` (0.65) times the previous frame (*missed division?*) or
  jumps above its inverse (*merged cells?*). The R² rule is off by default
  (`r2_min=None`): one cycle spans only ln 2 in log-size, so R² ≥ 0.9 flagged healthy cells
  for ordinary segmentation noise. The fixed relative tolerance is opt-in (`rel_tol=None`).
  **Existing calls that relied on the defaults flag different cells.**
- `flag_report(rates)` counts flagged cells per reason, next to the thresholds used
  (`rates.attrs["flag_thresholds"]`).

**`scale()` takes one `on_existing` instead of `exist_ok`/`exist_skip`** — the same
`'skip' | 'overwrite' | 'error'` vocabulary as `export_rois`, decided per stage notebook
and execution folder:
- `'skip'` is the former `exist_skip=True`: a stage that already ran is kept unless an
  input it recorded changed, and once one stage re-runs every later stage of the chain does.
- `'overwrite'` is new: before any stage runs, every stage in `analysis_script` is
  cleared (see `clear_stage` below), and its executed notebook (+ `.html`) is deleted. Stages
  not in `analysis_script` are kept, so `scale(..., ['03_DoublingTime.ipynb'],
  on_existing='overwrite')` redoes only that stage. A notebook is matched to its stage(s)
  through the `code.notebook` the manifest records.
- `'error'` (the default) refuses a folder in which one of the stages already ran; the
  source fails and the others still run. Previously the default refused any existing
  folder.
- `exist_ok`/`exist_skip` still work and warn (`DeprecationWarning`): `exist_skip=True`
  → `'skip'`, `exist_ok=True` → `'overwrite'`, `exist_ok=False` → `'error'`. Passing both
  styles raises `TypeError`.

**`StageContext.clear(stage)` (now also `acia.analysis.stage.clear_stage(output_dir,
stage)`) no longer deletes other stages' results.**
- It used to unlink every *file* the stage recorded. That included `summary.json`, which
  every stage writing a summary records, so clearing 03_DoublingTime also wiped
  03_GrowthRate's numbers. Now a file another stage also recorded is kept, as directories
  already were.
- Inside a shared directory (`figures/`) the stage's own artifacts (written via
  `ctx.output_path`) are removed, so its figures do not outlive it.
- Its `summary.json` sections are removed (see below). The file is deleted once empty.

**The SequenceDashboard gallery no longer reflows on hover.** Hovering a position row
used to animate that row's height to a large preview, pushing every row below it down
and shifting the list under the pointer. Rows now keep their fixed height; the larger
preview is a single popover placed beside the gallery (never over it), centred on the
hovered row and clamped into the image pane (`placePreview` in the dashboard ESM). It
ignores the pointer, so moving down the list just moves the preview along.
`roi_export_viewer` gains `box_size` (default `max_size`): the player is a fixed square
box with `object-fit: contain`, so switching between ROIs of different aspect ratios no
longer resizes the panel.

**`export_rois` no longer holds a whole position in memory, and a re-run only exports
what is new.** Two changes to `acia.roi_export.export_rois`, plus the handle hygiene
around it:
- A position's frames are now walked **once**, and each frame is cropped for every ROI of
  that position while it is in hand. Previously the whole position was read into memory up
  front (`materialize()`, one `(T, H, W, C)` array — gigabytes on a long timelapse) and then
  re-read per ROI, re-correcting the same full frame once for every ROI. Peak memory is now
  the ROI stacks being filled, bounded by the new `max_roi_buffer_bytes` (default 2 GiB,
  which splits the ROIs over as many passes as it takes), plus the frames the reader holds
  (`chunk_frames`, default 8, read ahead on one background thread so a network read overlaps
  the warping). Neither grows with the length of the timelapse.
- **The drift correction is now composed into the crop, so each exported pixel is
  interpolated once instead of twice.** The export used to correct the whole frame onto the
  reference grid and then resample that into the ROI rectangle; it now carries the rectangle
  back into the uncorrected frame's own coordinates and takes the crop in a single warp.
  The two warps are affine and compose exactly — verified to 1e-13 on point mapping and
  0.004 px on a blob centroid — so it is the same region, but less blurred (a Gaussian blob's
  peak loses ~0.9% to the second resample) and much cheaper: on a 2048×2048 frame the
  full-frame warp was 2.53 ms of the 2.77 ms per-frame warp cost at one ROI. **This changes
  exported pixel values** (mean |difference| ~0.02% of range on smooth content, ~1% on
  hard-edged content), so an export is no longer byte-identical to one made by an earlier
  version. Each `roi_export.json` entry now records `crop_pipeline`, so a future change here
  re-exports instead of mixing two resamplings in one folder; files written before this (which
  carry no recorded settings at all) are still kept by `on_existing='skip'` — pass
  `on_existing='overwrite'` once if you want a folder consistently on the new pipeline.
- **Re-running is incremental** (`on_existing`, default `'skip'`): a selection whose file is
  already there, unchanged and complete, is skipped, and a position whose ROIs are all
  skipped is never opened, read or registered — so drawing one more ROI in the dashboard and
  re-running the export costs one position. This is also how an interrupted export resumes.
  `roi_export.json` is now written after every ROI rather than once at the end, each entry
  records the selection's geometry and the settings that change pixels, and each TIFF is
  written beside its target and renamed into place so a killed run cannot leave a truncated
  file that looks finished (`verify_existing` header-checks one that was written before this
  was true). `roi_nb` is kept stable across runs — a selection keeps the number it was given,
  so inserting a new ROI no longer renumbers (and silently reassigns) the existing files.
  `on_existing='overwrite'` restores the previous always-redo behaviour; `'error'` refuses.
  `roi_export.json` is now `acia.roi_export/v2`; v1 files are still read.
- `CZISequenceSource.close()` / `ND2SequenceSource.close()` and
  `SequenceFile.release(index)` are new: a per-position reader is now released when the
  export is done with it, instead of one open file handle per position accumulating for the
  life of the kernel (`SequenceFile.close()` could not release CZI/ND2 readers at all, since
  neither defined `close`). `export_rois` also closes the `RegistrationDashboard` it builds
  (which pinned every position's transforms and the whole `SequenceFile`), closes its
  per-position progress bar, and sends the dashboard's per-frame progress message at most
  every 50 ms rather than once per frame. `render_roi_preview` closes the handle it opens, so
  reviewing a folder of ROIs no longer holds one decoded stack per preview.

### Added

**Cross-run plots: choose, order and colour the conditions; see them over time, one at a
time** (`acia.analysis.results`):
- `distribution` and `dynamics` take `order=` (which groups, in which order -- a subset
  leaves the rest out) and `palette=` (a `{group: colour}` mapping or a list). Static plots
  are now coloured per group. `group_palette(groups)` fixes one colour per group for every
  plot; `natural_sorted(values)` is the default group order.
- `plot_over_time(df, value, kind=("violin", "dots", "bars"), by="condition", ...)`:
  per-cell doubling time or μ over the experiment, one colour per group, each group's mean
  (the reported value) dashed. Interactively, each group is a legend group and every
  layer its own entry, so one condition or one layer can be hidden.
- `step_through(plot_fn, df, order, ...)` re-draws any of these plots with one more group
  per step, axes and colours fixed: one PNG per step for slides, or a plotly slider.
- `division_census_table(rs)` sums the runs' division census per group and warns, naming
  the group, when no cell divided (the group is then missing from every per-cell table and
  from a Monod fit on them) or when only a minority did.
- `plot_monod(fit, ci=0.95)` shades the confidence band of the fitted curve
  (`MonodFit.band`, delta method on the fit's covariance, now stored in `MonodFit.cov`).

**Cells that never divide are accounted for, not dropped**
(`acia.analysis.doubling_time`):
- `classify_tracklets(tracklet_graph, source)` gives every tracklet a status (`divided`,
  `censored`, `excluded`) and a reason (alive at movie end, lost, present at start, one
  daughter, …), so nothing disappears silently from the doubling-time analysis.
- `growth_rate.fit_tracklet_sizes(overlay, tracklets, source)` fits any tracklet's size,
  dividing or not: a non-growing cell measures μ ≈ 0 with its own interval.
- `division_census(status, sizes, arrest_min_observation=…, arrest_max_growth=0.2)` reports
  `n_divided`, `n_censored`, `n_long_not_divided`, `n_arrested` and `fraction_divided`. It is
  a report, not an analysis input; undefined values are `None`.
- `cell_cycle.plot_fit_vs_doubling` (static) and a reworked
  `plot_fit_vs_doubling_interactive`: μ_fit against ln 2 / DT with both intervals as error
  bars, coloured (and, interactively, toggleable) by flag reason.

**`summary.json` sections have an owner.** `update_summary` attributes the section to the
stage context open on that folder (built with `stage=`, not yet finished), recorded as
`summary_sections` in its manifest entry; no notebook change is needed. Entries from
before this have no owner and their sections are kept by `clear_stage`.
`acia.analysis.cell_cycle.remove_summary_sections(output_dir, sections)` is the atomic
undo of `update_summary`.

**`acia.viz.annotate_sequence(source, *, timepoints, pixel_size, ...)`** — burns a time
stamp and a scale bar into every frame of a movie, lazily (frames are drawn on demand).
By default it adds a black band **below** the image (`placement='border_bottom'`: time on
the left, scale bar with its length label on the right), so nothing covers the cells;
`'border_top'` puts the band above and `'overlay'` draws onto the image in small boxes,
keeping the frame size. Pass the calibration of the *original* source — the rendered
overlays (`render_segmentation_mask` & co.) do not carry it; without a pixel size or
timepoints the bar or time is skipped with a warning. The default band height keeps the
frame height even, as yuv420p video requires. Two fixes in the older burn-in renderers:
`render_time` drew the text at the raw `xy_position`, so relative (float) positions put
it at the top-left corner; `render_scalebar` and `draw_scale_bar` ignored `color` for
the bar itself (always white).

**`acia.base.warp_rotated_crop(raw, spec)`** — the frame-level half of
`RotatedCropSequenceSource`, factored out so a caller holding a decoded frame can crop it
without going back through the lazy view. The class now delegates to it, so there is one
implementation of that warp matrix (as there is one `acia.registration.apply_correction`).

**`acia.registration.invert(transform)`** — the transform that undoes a `FrameTransform`
(`compose(t, invert(t))` is the identity). Pivot-independent, like `compose`, so it needs no
frame shape.

**`RegisteredSequenceSource.transform_for` / `.spec_for` / `.crop_rotated`** — the
missing-transform policy (`on_missing`, the `missing_frames` bookkeeping) now lives in
`transform_for`, so a caller that wants to *compose* the correction rather than apply it to
pixels can ask for the transform and still be counted in the per-position report. `spec_for`
carries a reference-frame crop spec back into one frame's own coordinates, and
`crop_rotated` on a corrected view returns a `RegisteredRotatedCropSource`, which crops the
raw frame in a single warp instead of resampling twice.


**Exporting curated ROIs** (`acia.roi_export.export_rois`) — the writing half of the
curation workflow, moved out of `SequencePreprocessing.ipynb` and now tested:
- One TIFF per selection, named from a template (`position`, `roi_nb`, `label`, `id`), with
  the positions that carry no ROI skipped.
- Optional per-position drift correction first (`registration_method_name`), with the ROI
  interiors kept out of the registration objective and each ROI's anchor frame carried onto
  the corrected view. Anchors outside the exported range are rejected before any
  registration work starts.
- `max_frames` truncates each position before registering and cropping, so a quick test run
  never mixes registered and unregistered frames; `progress=False` silences the bars.

**Correcting acquisition slips** (`acia.segm.acquisition_fixes`) — when the microscope
images one position twice in a row, every later slot of a multi-position ND2/CZI holds the
image of the slot before it. acia now undoes that in memory, without writing to the raw file:
- `open_sequence(path, acquisition_fixes=[(t, position), ...])` names the raw slot(s) that
  hold the repeated image. `SequenceFile.position(i)` then reads every frame from where it
  actually landed; `SequenceFile.raw_position(i)` keeps the file's own order. Positions the
  shift runs through end one frame early.
- `save_fixes(path, slips)` writes `<file>.acia-fixes.json` next to the raw file, and
  `open_sequence` picks it up automatically. `acquisition_fixes=[]` forces the raw order.
- `detect_acquisition_slips(seqfile)` suggests slips from the per-image stage XY: two
  consecutive images at the same place. ND2/CZI readers gain `stage_xy(frame)` and
  `SequenceFile` gains `stage_positions()`.
- The fixes are recorded in the manifest `source` block. `load_selection` /
  `load_registration` reopen the file with them and warn when a passed source uses others.

**Recording a stage as it happens** (`acia.analysis.StageContext`) — the record is now
written while the stage runs, and is visible in the notebook that produced it:
- `StageContext.for_image(..., stage='Segment')` names the stage up front, which is what
  lets everything below be written immediately rather than in one call at the end.
- `log_params()`, `log_metrics()`, `log_figure()`, `log_artifact()` persist to
  `stage_manifest.json` the moment they are called. Parameters and figures are the parts
  of a run that re-running cannot recover, and a notebook that died half-way used to
  record nothing — while `scale(exist_skip=True)` then skips that stage on every later
  run, so the run that died was exactly the one whose record was wanted.
- An unfinished stage is marked `"running"` and a stage whose cell raised is marked
  `"failed"`, with the exception recorded. `finish()` marks it `"ok"`. Previously a
  crashed stage was indistinguishable from one that never ran.
- `input_path()` / `output_path()` state which artifacts a stage reads and writes.
  Direction is still *observed* rather than declared — that is what makes the dependency
  graph unable to fall out of date — but a declared output never written is reported as
  `io.missing`, and a declared read is recorded even for readers the audit hook cannot
  see (OpenCV's C++ layer, a remote OMERO source). `require()` is now an alias of
  `input_path()`.
- `log_calibration(source)` records the pixel size and frame interval a stage actually
  resolved, and warns when a later stage of the same population resolves a different one.
  The movie stays the single authority; this exists so that "what interval did this run
  use?" has an answer and a silent disagreement between two stages has something to
  notice it. `calibration()` reads it back.
- `stage_table()` gains `error_type` and `error_message`, so a fan-out can say *why* a
  population stopped rather than only that it did — across a batch that is what separates
  one ROI running out of GPU memory from a chain that is broken for everything.
- `ctx` now renders in a notebook: stages and how they ended, this stage's parameters,
  metrics and figures, and every file in the folder with the stage that produced it.
- `scale()` records the parameters papermill injected, so a notebook no longer has to
  repeat its parameter cell into `log_params()`, and renders each executed stage notebook
  to HTML beside it (needs the new `report` extra; best-effort, never fatal).
- The manifest gains `status`, `params`, `metrics`, `figures`, `calibration`, `error` and
  `io.declared_inputs`, under `acia.stage/v2`. Additive: entries written by earlier
  versions have no `schema` key and are read unchanged, and `record()` still writes its
  `**extra` flat at the top of the entry.

**Interactive curation widgets** (`acia.notebook`, new `widget` extra — the
whole module stays importable without `anywidget` installed):
- `SequenceDashboard`: a three-pane curation UI (position gallery, ROI editor,
  selections list) over a multi-position acquisition, with a live client-side
  crop preview, numbered `roi_01`/`roi_02` default labels, Delete/Backspace to
  remove and Ctrl/Cmd+C to duplicate the active ROI, the open file's path shown
  in the header, and auto-save on by default (`save_dir=`, honored by `resume()`).
- `RegistrationDashboard`: pick, verify and batch-apply a drift-correction
  method across every position, with verify progress and batch-apply ETA, a
  manifest checkpoint every 20 frames so an interrupted run loses at most that
  many, and a play/pause + scrubber side-by-side comparison player.
- `ROICropper`: a manual rotated-rectangle ROI on frame 0, fitted from ≥3
  clicked points (`cv2.minAreaRect`) and/or dragged, resized and rotated;
  emits a `RotatedCropSpec`.
- `FilterExplorer`: live thresholds for the cell filters, one (min, max) slider
  per filter, recolouring the overlay as the handles move. Each contour's value
  is precomputed once in Python and shipped to the browser, so dragging needs no
  kernel round-trip. `.params` emits pint quantities; `save()` writes
  `filter_params.json` for a scaled run.

**Frame registration** (`acia.registration`):
- A `RegistrationMethod` abstraction with five peer implementations for rigid
  inter-frame drift — `PhaseCorrelationHighpass`, `MaskedTemplateCorrelation`,
  `HoughLineRigidFit`, `FeatureRANSACEuclidean` and `GradientECC` — each
  treating the microfluidic device's static structure as the signal and the
  growing colony as noise.
- `apply_correction`, `RegisteredSequenceSource` (lazy per-frame correction),
  `ImageSequenceSource.register()`, and `acia.registration_persistence`
  (`RegistrationRecord`/`RegistrationManifest` + save/load).
- `GradientECC` gained a coarse-to-fine pyramid (single-level ECC was slow and
  empirically prone to non-convergence at production resolution), plus opt-in
  `early_stop_delta_px` (6–9× faster within the same ±0.5 px/±0.5° tolerance)
  and `translation_only`.
- Registration on time-lapses whose content changes (a colony growing into the
  field of view), where a fixed-reference correlation coefficient decays with
  elapsed biology rather than with misalignment and a confidence gate starts
  rejecting perfectly good fits in a contiguous tail of late frames:
  - `GradientECC(exclude_rects=..., exclude_shrink_px=...)` leaves the regions
    whose content changes (typically the ROIs a caller already marked) out of
    the ECC objective, via `cv2.findTransformECC`'s `inputMask`.
    `exclude_shrink_px` keeps a band just inside each rectangle's border, whose
    static device geometry measurably helps precision.
  - `acia.registration.ReanchoringReference` wraps any `RegistrationMethod`
    with a reference policy: `"fixed"` (the previous behavior), `"reanchor"`
    (fall back to the last successfully registered frame and compose, only
    where a frame would otherwise be recorded as a failure), or `"chained"`.
  - `acia.registration.compose(first, second)` chains two `FrameTransform`s;
    pivot-independent, so it needs no frame shape.
  - `FrameTransform.confidence`: the estimating method's own goodness-of-fit
    score, persisted per frame so a run's confidence trend is auditable.
    Reported by `GradientECC`, `MaskedTemplateCorrelation` and
    `FeatureRANSACEuclidean`; `None` elsewhere.
  - `on_low_confidence` on each of those three methods (and
    `RegistrationDashboard(low_confidence=...)`, which forwards it): `"keep"`
    (new default) returns a fit scoring below the method's threshold anyway,
    with its `confidence` set and a `LowConfidenceWarning` naming the score and
    the threshold it missed; `"reject"` raises `RegistrationError` as before.
    Keeping means `registration_transforms.json` holds a transform for *every*
    frame, so `RegisteredSequenceSource` stops falling back to an uncorrected
    (or nearest-neighbour) frame downstream — a weak fit is now reported by its
    score rather than by its absence. Failures with no estimate to keep (blank
    input, non-convergence, a non-finite result, a match pinned to the
    search-window edge, too few correspondences to fit a model at all) still
    raise regardless of the policy.
  - `RegistrationDashboard(method_kwargs=..., reference_mode=...)` and
    `batch_apply(..., method_kwargs=...)` for per-position settings —
    registration method settings are now a supported argument rather than
    something a caller has to reach in and patch.
  - `batch_apply(sources={position: source})` registers a supplied (e.g. lazily
    sliced) source per position instead of the full position.
  - `acia.registration.apply_correction_to_spec` carries a `RotatedCropSpec`
    drawn on one frame onto the drift-corrected (reference-frame) view, so
    `source.register(transforms).crop_rotated(spec)` takes the region the spec
    was drawn around rather than wherever that frame had drifted to. It shares
    its matrix with `apply_correction`, so crop geometry and corrected pixels
    cannot disagree.
  - `ImageSequenceSource.register(..., on_missing=...)`, also on
    `load_registration`: `"warn"` (default), `"nearest"` (correct a failed
    frame with its neighbor's transform instead of exporting it uncorrected,
    which is off by the full accumulated drift), or `"error"`.
    `RegisteredSequenceSource.missing_frames` reports every gap at once.

**ROI curation** (`acia.selection`, `acia.notebook`):
- `SequenceDashboard(preview_frame=...)` chooses the frame the gallery
  thumbnails and the ROI editor open on, defaulting to the **middle** frame. In
  a growing culture frame 0 is typically empty, so opening there gave no
  indication whether a chamber held any cells worth curating.
- `RoiSelection.anchor_frame`: the frame an ROI was drawn on, recorded in
  `selection.json`. Additive and omitted when zero, so manifests written by
  earlier versions load unchanged and are re-saved byte-identically (schema
  stays `acia.selection/v1`).

**Image sources and I/O**:
- `open_sequence(path)`: one entry point that dispatches by suffix to the ND2,
  CZI, TIFF or folder reader.
- `ND2SequenceSource`: one position of a Nikon ND2 file as a lazy `(T, H, W, C)`
  series, read one frame at a time so peak memory is a single frame (handles
  ~80 GB files). Optional `nd2` extra.
- Zeiss CZI read support (`CZISequenceSource`) and TIFF export; `save_tiff_stack`
  gained `ome`/`compression`/`channel_names`, so a cropped or registered export
  can carry OME metadata and land compressed rather than as a raw ImageJ
  hyperstack.
- `FolderSequenceSource`: a folder of per-timepoint TIFFs as one lazy sequence,
  where file *i* (natural-sorted) is frame *i*. `open_sequence` grew a directory
  branch — a folder holding the TIFFs is one position, a folder whose immediate
  subfolders hold them is one position per subfolder — so the dashboards and the
  selection/registration manifests get folder input for free.
- fsspec-based remote image sources: read TIFFs from SMB/SAMBA shares (and any
  fsspec backend) via `SambaSequenceSource` / `LocalSequenceSource`, plus
  `list_sequence_sources` for folder discovery and a per-user credentials store
  (`acia.config`).
- `SelectionManifest` / `save_selection`: persistence for curation selections.
- `read_tiff_calibration`: pixel size and frame interval auto-detected from
  OME-XML or ImageJ metadata, resolved lazily on first access so the
  zero-I/O-on-construct contract holds for remote paths. An explicit constructor
  argument still wins per field.
- `RotatedCropSpec`, `crop_rotated()` and the lazy `RotatedCropSequenceSource`
  (warps each frame on demand, calibration passed through), plus a generic
  `materialize()` that freezes any lazy source into memory.
- `save_crop_capture` / `load_crop_spec`: the full source frame as a normalized
  8-bit PNG plus a JSON sidecar holding the crop as an oriented-box label with
  provenance, auto-enumerated into a dataset directory.
- `ImageSequenceSource.to_channel(c)`: a generic lazy single-channel view
  (equivalent to `self[..., c]`), now available on every source implementation
  (e.g. `LocalSequenceSource`), not just `THWCSequenceSource`.
- `ImageSequenceSource.to_rgb(*, channel=0, colors=None)`: a lazy RGB view
  across every source — grayscale (normalize + triplicate) or per-channel colour
  composite, with a starter palette in the new `acia.colors`.
- numpy-style `(T, H, W, C)` indexing on image sequence sources
  (`src[::2, 100:200, 50:150, 0]`, composable lazy views) and temporal slicing of
  overlays (`overlay[:20]`).
- pint calibration defined at load: `pixel_size` and `frame_interval`/`timepoints`
  flow through slices, into overlays (per-detection `contour.time`) and are pulled
  automatically by the extractors (explicit `input_unit` still overrides).

**Stage chains and batch execution** (`acia.analysis`):
- `StageContext.for_image(...)` replaces the ~50 lines every stage notebook
  re-declared: it resolves the output folder, parses the population identity from
  the ROI name (`key_pattern` takes any regex, or `None`), and exposes
  `path()`/`require()`/`has()` for artifacts, `record()`/`manifest()` for the
  append-only `stage_manifest.json`, and `keyed()` for adding key columns to an
  exported table (carrying `df.attrs["units"]` across the assign). `read_manifest()`
  and `stages_run()` are the reader side.
- Stages now record **what the filesystem observed** rather than what the author
  remembered to declare — a notebook written before this gains provenance with no
  edit. A PEP 578 audit hook sees reads; a before/after diff of the output folder
  sees writes that Python cannot observe (`cv2.imwrite`, ffmpeg). Measured
  overhead of the hook: none. What follows from having the record:
  `stage_graph()` derives the dependency graph instead of declaring it,
  `check_stale()` reports a stage whose input changed since it ran (warning only),
  `ctx.clear(stage)` removes exactly what a stage recorded, and `stage_table()`
  turns a batch into one DataFrame.
- Source-aware notebook scaling: `acia.analysis.scale` accepts OMERO ids, file
  paths/URLs, or parameter dicts, with per-type default execution naming.
- `scale(max_workers=...)`: optional parallel notebook execution over a
  `ProcessPoolExecutor` started with `spawn` (papermill's `chdir` is
  process-global, so threads race; `fork` would duplicate a CUDA-initialised
  parent). Default `1` is the previous sequential behaviour.
- Labelled progress bars for `scale`: sequential runs name the stage and its
  input (`02_Track.ipynb | pos001_roi002.tiff`); parallel runs render one bar per
  pool worker in the parent process, since children cannot draw into a shared
  stderr legibly. New `stage_progress` chooses whether finished bars stay as a
  timing log, collapse, or are suppressed.

**Analysis**:
- `estimate_growth_rate`: the log-linear growth-rate fit as a reusable,
  unit-aware call (statsmodels OLS of log(y) on time), returning a
  `GrowthRateResult` — growth rate, standard error, confidence interval,
  doubling time, R², p-value as pint quantities — plus a figure.
- `extract_growth`: one call combining `ExtractorExecutor` (frame + physical
  time + physical area) with `estimate_growth_rate`.
- A pluggable `CellFilter` abstraction with calibrated `value()` and a pint
  `(vmin, vmax)` range, built-ins `AreaFilter`, `LengthFilter`, `WidthFilter`,
  `CircularityFilter` and `BoundaryClosenessFilter`, and `apply_cell_filters`.
  Thresholds are given in µm/µm², so they are camera-invariant.
- `compute_doubling_times` and `plot_doubling_time_hose`
  (`acia.analysis.doubling_time`): per-cell doubling time from clean divisions
  (exactly one identified mother, exactly two daughters), always reading real time
  via `source.timepoints[frame_idx]` so irregular timestamps are handled, with the
  mean's evolution shown as percentile-bootstrap confidence bands.
- `plot_property_histograms` (`acia.analysis.properties`): properties h-stacked as
  columns, before/after v-stacked as rows, sharing bin edges and limits per column
  so outliers are directly comparable. Optional `show_removed` overlays the
  filtered-out cells in red.
- Unit-aware property-extractor tables via pint-pandas:
  `ExtractorExecutor.execute(units="none"|"header"|"pint")` plus
  `attach_units` / `strip_units` / `units_in_header` converters.
- `write_units_csv` / `read_units_csv`: store a CSV *with* its units and reload it
  into unit-aware pint columns, so derived columns get their units automatically.
- `BoundaryClosenessEx`, which also makes the boundary margin plottable alongside
  the other properties.
- `ExtractorExecutor` is empty-overlay safe: a 0-cell ROI no longer raises.

**Visualization** (`acia.viz`):
- `tracklet_graph_to_segments()` and `plot_tracklet_lineage()`: a tracklet graph
  reshaped to one line per cell cycle, and a one-call wrapper over it.
  `annotate_tracklet_times()` stamps `start_time`/`end_time`, and
  `TrackastraTracker` now calibrates the tracked overlay from the source, so
  lineage plots read real time off the nodes and label the axis themselves.
- `compose_sequences()` / `label_sequence()`: tile image sequences horizontally or
  vertically with optional per-panel titles. `ComposedSequenceSource` is itself an
  `ImageSequenceSource` (so grids come from nesting) and lazy.
- `render_segmentation_mask(colors=...)`: colour masks from a per-cell value table
  (a pandas Series, dict or single-column DataFrame) matched by id — an (r, g, b)
  triple used directly, a number mapped through `cmap`, anything else categorical.

**Segmentation**:
- Three new backends, each a `SegmentationProcessor` subclass behind its own
  optional-dependency extra: `StarDistSegmenter` (`acia.segm.processor.stardist`,
  `stardist` extra) wraps StarDist's star-convex instance segmentation — no
  bacteria-specific pretrained model exists upstream, so it's a contrast baseline
  for round/oval targets rather than a primary bacterial segmenter;
  `MicroSAMSegmenter` (`acia.segm.processor.microsam`, `microsam` extra) wraps
  micro-sam's (μSAM) AIS automatic-instance-segmentation path via the
  light-microscopy-finetuned `vit_b_lm` checkpoint by default; `DeltaSegmenter`
  (`acia.segm.processor.delta`, `delta` extra) wraps DeLTA 3.x's segmentation
  U-Net only (no tracking model touched), selectable by imaging `regime`
  (`"2D"` agar-pad vs `"mothermachine"`), targeting the torch backend so it
  shares a torch stack with Cellpose/Omnipose instead of pulling in
  TensorFlow. See
  `_bmad-output/implementation-artifacts/spec-additional-segmentation-backends.md`.
- `FlowposeRTSegmenter` (`acia.segm.processor.flowpose_rt`): omnipose-compatible
  segmentation backed by the lightweight `flowpose-rt` package (no
  cellpose/omnipose/numba at runtime), selectable via the new `flowpose-rt` extra.
  Processes frames in configurable batches (`batch_size`, default 20) with a
  `tqdm` progress bar, matching `OmniposeSegmenter`'s existing behavior.
  `weights_path=...` loads a local checkpoint (e.g. a fine-tuned model) instead of
  the downloaded zoo weights; `model` then names the zoo entry whose preprocessing
  contract the checkpoint follows.
- Per-backend optional-dependency extras (`cellpose`, `cellpose-sam`, `omnipose`,
  `flowpose-rt`). The backends were previously imported lazily and never declared,
  so a user met missing dependencies one failure at a time at segmentation time.
  They have conflicting torch/numpy/cellpose pins and are **mutually exclusive** —
  install one per environment.
- Lazy model construction across `SegmentationProcessor`, with `release()`, a
  re-entrant `load()` context manager, and an `autorelease` flag (default on) so a
  one-time-lapse run frees the GPU after each call.
- `CellposeSAMSegmenter` segments under a `tqdm.auto` bar (Cellpose-SAM routes its
  own progress to a logger at 30 s intervals, so it looked frozen), reports which
  model/device/`diam_mean` actually loaded, and accepts `pretrained_model`.

**Persistence formats**:
- `acia.segm.formats.save_segmentation` / `load_segmentation`: a compressed binary
  polygon archive. Ids, labels and sub-pixel coordinates survive exactly, so a
  reloaded overlay still joins to a property table exported from the same
  segmentation. `load_segmentation` sniffs the format from magic bytes and also
  reads plain and gzipped simple-segmentation JSON.
- `acia.tracking.formats.save_tracking` / `load_tracking`: returns the same
  `(overlay, tracklet_graph, tracking_graph)` triple a tracking processor returns,
  so a step that loads is a drop-in for a step that tracked. Format choice is
  benchmarked, not assumed: on 150k detections the archive is 41 MiB / 1.5 s
  against gzipped JSON's 106 MiB / 43 s.

**Documentation**: the published site was rebuilt around a getting-started path —
five runnable tutorials (committed source-only and executed at build time, so every
build proves they still run), an API reference generated by autosummary during the
normal `sphinx-build`, task-shaped guide pages, and a glossary. The build is
warning-free under `-W`.

### Changed
- `RegistrationDashboard` now defaults to `reference_mode="reanchor"`, so a
  frame that cannot be estimated against the reference is retried against the
  last successfully registered one instead of being recorded as a failure. This
  is a pure fallback — a frame that already succeeded takes exactly the path it
  always did — and `reference_mode="fixed"` restores the prior behavior.
  Resuming a partial run recorded under a different policy re-registers the
  position rather than merging incompatible transforms; a *clean* fixed-mode
  record (no failed frames) is still reused under `"reanchor"`, since
  re-anchoring would have produced it identically.
- `registration_transforms.json` gained optional `reference_mode`,
  `reference_frames`, per-transform `confidence`, and `method_params` fields.
  All are additive in both directions: older manifests load unchanged, and a
  fixed-reference run still writes exactly the JSON it always did (schema stays
  `acia.registration/v1`).
- A registration fit scoring below its method's confidence threshold is now
  **kept** by default (`on_low_confidence="keep"`), warned about, and stored
  like any other transform, instead of raising and leaving the frame in
  `failed_frames` without one. Two consequences worth knowing:
  - `ReanchoringReference`'s fallback is driven by `RegistrationError`, so with
    the default policy a merely unconfident fit no longer re-anchors — the
    method reports success and the estimate is carried forward as-is.
    `reference_mode="reanchor"` stays valid but now fires only on outright
    failures; construct the method with `on_low_confidence="reject"` to get the
    older interplay, where a low score re-anchors.
  - Re-running `batch_apply` over an existing `registration_transforms.json`
    does **not** repair it: a complete position is skipped and a partial one
    resumes by frame *count*, so frames a previous run recorded as failures are
    never re-estimated. Delete the file (or the affected records) and register
    those positions again to pick up the new policy.
- `SequenceDashboard` no longer resets the frame scrubber when you switch
  position — matching what clicking a row in the selections list already did,
  and letting the same timepoint be compared across positions. The scrubber's
  "(view only)" label is gone: the frame on screen is now the frame a newly
  drawn or edited ROI is anchored to. An ROI whose anchor is not the frame
  being shown renders muted and dashed with its anchor frame on the label;
  moving it re-anchors it to the displayed frame. Merely *clicking* an ROI
  selects it without re-anchoring.
- `RegisteredSequenceSource` now warns once per missing frame index instead of
  once per read, so a lazy multi-pass consumer (crop → write) no longer repeats
  the same warning on every pass.
- Unmeasurable geometry is reported as `nan` rather than raising or being
  silently measured as 0: `LengthEx`/`WidthEx` on a collapsed rotated rectangle
  and `PerimeterEx` on an absent polygon. 0 would be a claim, and a 0-length cell
  passes any open-below bound, so junk would survive filtering; `nan` states the
  property is undefined and `CellFilter.mask` drops non-finite rows. A genuinely
  zero measurement stays 0.
- `PropertyExtractor._calibrate` warns when a source has no `pixel_size`, since
  the column is then labelled µm while the values are px. This protects every
  consumer of the table, not just the filters.
- `write_ctc_tracking` writes zlib-compressed masks — mostly background and highly
  repetitive, so ~50× smaller (1 GiB → 18 MiB for a 500-frame 1024×1024 movie).
  Output stays a valid TIFF.
- `SelectionManifest.load` accepts the directory `save_selection` was given, not
  only the `selection.json` path.
- Migrated CI/CD to GitHub Actions with automated, OIDC-based PyPI releases and
  GitHub Pages documentation.
- Dependency floors and additions: `papermill>=2.6.0` (the dict form of
  `progress_bar` and the engine API post-date 2.4), `statsmodels` for the growth
  fit, and `scipy` now declared explicitly rather than relied on via scikit-image.

### Removed
- `apply_cell_filters(images=...)` and `FilterExplorer`'s row-wise fallback. Both
  now require the properties table; pass `properties=...`. A filter whose column is
  absent raises and names the extractor to add, rather than silently falling back
  to the slow path.

### Fixed
- `update_summary` wrote a bare `NaN` (invalid JSON) for undefined values, e.g. the
  population doubling time of a run that did not grow. Non-finite numbers are now written
  as `null`.
- `hover_table` (interactive cell plots) showed row numbers for tables read back from CSV;
  it now uses the `tracklet` column when there is one. `valid` read back as text is parsed.
- `plot_doubling_time_hose`'s docstring said a cell counts up to its last frame; it counts
  until its division (`start_time <= t < division_time`), as the code always did.
- Re-running a recorded stage left the manifest describing two runs as one. The new run
  inherited the previous one's recorded outputs, so until it collected its own the entry
  showed fresh parameters beside stale outputs — and a re-run that died half-way left
  exactly that mixture looking like a completed run. A new run now starts from a clean
  entry, and re-running a stage warns, naming when it last ran and which downstream
  stages its results have just made stale.
- `stage_manifest.json` was rewritten non-atomically. With one write per stage the window
  was academic; recording as the stage runs widens it a hundredfold, and a truncated
  manifest is not a lost record but a poisoned folder — every later `for_image()` reads it
  while resolving the source, so the next run would die on a `JSONDecodeError`. Writes now
  go through a temp file and `os.replace`.
- `StageContext.clear()` `rmtree`d a directory artifact even when another stage also wrote
  into it, so clearing one stage could delete another's figures. A shared directory is now
  left alone.
- `stage_table()` let a recorded setting shadow a derived column — a stage recording
  `duration_s=...` overwrote the real duration. Derived columns are now applied last.
- Tracking input was **silently misaligned** for any overlay whose first
  detection was not on frame 0. `overlay_to_masks` built its stack from
  `Overlay.timeIterator()`, which starts at the first *populated* frame, so an
  overlay beginning on frame 3 handed the tracker frame 3's cells as frame 0 —
  associating every cell against the wrong image. Frames are now indexed
  absolutely. Overlays from `load_segmentation(path, source)` carry an explicit
  frame extent and were never affected. `write_ctc_tracking` had the same bug in
  its `man_track{i:04d}.tif` numbering, and is fixed the same way.
- `overlay_to_masks` on an empty overlay raised
  `ValueError: zero-size array to reduction operation minimum`; it now returns an
  empty `(0, height, width)` stack.
- Label values above 65535 silently wrapped in the `uint16` mask stack. The stack
  now widens to `uint32` when any label needs it.
- Cropping a drift-corrected source with an ROI drawn on a frame other than the
  registration reference placed the crop off by the drift accumulated up to
  that frame — silently, and worse the later the frame. The ROI's
  `anchor_frame` is now recorded and applied via `apply_correction_to_spec`
  before cropping. ROIs drawn on frame 0 — every selection made before this
  release, since frame 0 was the only frame the editor showed — are unaffected.
- `scale()` aborted the whole batch when a notebook's kernel failed to start.
  Both the sequential and the parallel path caught only
  `papermill.PapermillExecutionError` — which means "a cell raised" — while a
  kernel that never becomes ready raises a bare `RuntimeError`
  (`Kernel died before replying to kernel_info`). That escaped, so the sources
  after the failing one were never attempted, losing a whole overnight batch to
  one transient hiccup. Every exception is now isolated to its own source, and
  the exception is logged: `failed_ids` only carries the id, so the closing
  summary could report how many sources failed but never why. A kernel that
  fails to start is additionally retried once — the handshake fails before any
  cell runs, so re-executing repeats no work and duplicates no side effect,
  which turns a lost stage into a slow one. A kernel that dies *mid*-notebook
  raises `DeadKernelError` and is not retried, since cells have already run.
- `merge_cells_to_colonies()` gave every blob in a frame the same id (`-1`), and
  `ExtractorExecutor.execute()` joins extractor results on that id index — a join
  on a non-unique index is a cartesian product, so *k* blobs became *k*⁶ rows and
  each area was repeated *k*⁵ times. Any per-frame sum (total colony area in
  particular) was 32× (k=2), 243× (k=3) or 1024× (k=4) too high, destroying the
  log-linear growth fit. Blobs now get running unique ids, the frame number comes
  from the detections rather than the enumerate position, the input's time model
  is carried into the colony overlay, and `execute()` raises on duplicate contour
  ids instead of returning a table wrong by orders of magnitude.
- `Instance.coordinates` raised `AttributeError: 'MultiPolygon' object has no
  attribute 'exterior'` when a detection's mask had more than one connected
  component — a cell the segmentation split in two, or a speck sharing its
  label. This aborted `save_segmentation` partway through. Such a mask has no
  single outline, so its largest part is now used, the same choice
  `Instance.draw` already made; the new `acia.utils.largest_polygon` helper is
  shared by both (and by the CTC readers, which had the same latent crash).
  `Instance.is_fragmented` reports when it applies, and `save_segmentation`
  warns once with a count, since those detections reload smaller than they were.
  `Instance.area` is computed from the mask and is unaffected.
- `HoughLineRigidFit` raised `IndexError` on every call under OpenCV 5, which
  changed `cv2.HoughLinesP`'s output from `(N, 1, 4)` to `(N, 4)`. Detected
  segments are now reshaped instead of indexed at a fixed layout, so both
  layouts work.
- `LocalSequenceSource.get_frame()`/indexing now honor `normalize_image` (previously
  only `__iter__` did), and the 2D-frame-to-3-channel duplication in `prepare_image`
  is gated by it too. TIFFs opened via `open_sequence` (which already requests
  `normalize_image=False`) now return their true dtype/channel data instead of
  silently-normalized, artificially-3-channel uint8 — this also fixes previously
  wrong `dtype`/channel-count metadata, and affects any quantitative analysis
  (registration, fluorescence/area extraction) run on `open_sequence`-opened TIFFs.
- `materialize()` on a TIFF source peaked at *T* × the full stack (measured 41× on
  a 40-frame stack) and did *T* full-file decodes: `_read_images()` re-decoded the
  whole file per `get_frame()`, and `materialize()` built a list of all frames,
  which — with `normalize_image=False`, where a frame is a *view* into its parent
  buffer — pinned *T* parent stacks alive at once. The decode is now cached
  (dropped by `close()`) and the output array is filled frame by frame. Peak for a
  40×512×512 uint16 stack: 860 MB → 21 MB.
- `save_tracking` guarantees full frame coverage. `write_ctc_tracking` named its
  masks by enumerating `timeIterator()`, which starts at the first *populated*
  frame, so an overlay with an empty frame 0 wrote a stack shifted against the
  movie — every reloaded detection on the wrong frame, with no error.
- `load_tracking` attaches the calibration before building the tracking graph.
  `read_ctc_tracking` built it while the reloaded overlay was still uncalibrated,
  and `ctc_track_graph` stamps node `time` from `cont.time`, so re-attaching the
  time model afterwards left a timeless graph and a lineage plotted over
  `time_feature='time'` silently had nothing to plot.
- `render_video` output is now Firefox-compatible: `macro_block_size` dropped to 2
  (the minimum yuv420p needs) so already-even frames are no longer stretched, and
  `-movflags +faststart` moves the moov atom to the front of the file, which
  Firefox requires in order to play it at all.
- `JupyterVisualizationMixin._repr_html_`'s interactive preview no longer crashes
  on a genuinely single-channel raw `(H, W, 1)` frame (as now returned by
  `open_sequence`-opened TIFFs, and already returned by `ND2SequenceSource`/
  `CZISequenceSource`) — PIL's `Image.fromarray` has no mode for a trailing
  size-1 channel axis; that axis is now squeezed before display.
- Releasing a `FlowposeRTSegmenter` now also clears `torch.compile`'s
  process-global CUDA-graph cache via `torch.compiler.reset()` — the base
  class's generic `torch.cuda.empty_cache()` doesn't reclaim it, since
  flowpose-rt's default `torch.compile(mode="reduce-overhead")` (on CUDA)
  caches graphs outside the model instance. Each processed batch is also
  followed by an explicit `gc.collect()` to encourage prompt reclamation of
  its activation memory before the next batch starts.
- `plot_property_histograms` degrades instead of raising when a property has no
  finite values — the common, valid case of an empty `df_after` (everything
  filtered out) or an empty ROI. That axis is drawn empty with a "no cells" note.
  Only an empty `properties` list still raises. Unit labels render with pint's
  `~P` (pretty Unicode, e.g. "µm²") instead of `~L`, which matplotlib showed as
  literal `\mathrm{...}` markup.
- `SequenceDashboard`'s Source field was always empty (`SequenceMetadata.to_dict()`
  never emitted a `path` key), and auto-save was a frontend-only flag: off by
  default, invisible to Python and lost on re-render. Both are now synced traits.
- Rendering fixes found during the viz performance work: track labels above 65535
  wrapped through a uint16 cast; uint16 sources blended 0–255 overlay colours
  against a 0–65535 frame, leaving the overlay invisible; foreground was derived
  from the overlay colours, so a cell that randomly drew `(0, 0, 0)` was treated
  as background.

### Performance
- Property extraction and cell filtering, measured at 1024×1024 with 300
  cells/frame, cumulatively **13.88 → 0.24 ms/cell** (Instance-backed) and
  0.66 → 0.045 (Contour-backed) — the 150k-detection reference ROI drops from
  ~35 min to under a minute, and a 107-ROI batch from ~62 h to ~1 h. Three
  independent causes:
  - The filters re-measured every contour that extraction had just measured.
    Each `CellFilter` now reads the column named after it out of the table the
    `ExtractorExecutor` already produced, comparing whole numpy columns against
    bounds converted once per run (filtering 7.12 → 0.001 ms/cell).
  - `Instance` held a reference to the whole frame's label image and every
    geometry access scanned all of it for one cell, so cost tracked *frame* area
    rather than cell area. Geometry now comes from a cached crop of the label's
    bounding box, shifted back into frame coordinates; `overlay_from_masks` gets
    every box from one `scipy.ndimage.find_objects` call. At 2048² `polygon`
    went 24.40 → 0.35 ms/cell (70×). `Contour.polygon` is cached too.
  - Unit conversion went through pint per value. `convert_array` applies it as
    one multiply by a precomputed factor, having solved for the affine
    `(scale, offset)` and *verified* the identity on probe values, falling back
    to the per-value path if it does not hold. The minimum rotated rectangle is
    now derived once per overlay via `shapely.oriented_envelope`.

  Verified against `tests/equivalence/golden.npz`, a snapshot of the pre-change
  implementation: identical kept-id sets across 5 scenes × 12 filter
  configurations, property values exact, polygons equal, units unchanged.
- Mask and tracking rendering is 10–25× faster. `render_tracking_mask` rebuilt
  the frame label mask with one full-image comparison per cell (O(n_cells·H·W))
  when `overlay_from_masks` had already handed every instance the same
  full-frame mask — a single LUT remap suffices. `render_tracking` recomputed
  constant cell centers inside its per-edge loop.

      render_tracking_mask 1024², 400 cells:  3.3 → 60 fps
      render_tracking_mask 2048², 400 cells:  0.7 → 17 fps
      render_tracking      1024², 300 cells:  34 → 328 fps

  `render_tracking` output is byte-identical; `render_tracking_mask` differs by
  at most 1/255 per channel, from blending in uint8 instead of float32.

- Overlay → label-mask rasterisation, the conversion every tracking backend runs
  before it can start (`overlay_to_masks`, used by `TrackastraTracker`,
  `LapTrack*`, `PyUATTracker`, `UltrackTracker`), plus `Overlay.toMasks`, the two
  CTC exporters and fluorescence extraction. Each rasterised **every cell over
  the whole frame** and combined the results with `np.maximum`, so cost tracked
  `n_cells × frame area` with 3–5 frame-sized temporaries per cell. The shared
  fast path from the viz work (`_frame_label_mask`) now lives in
  `acia.segm.rasterize` and backs all of them.

      overlay_to_masks 1024², 5×150 cells, Contour-backed:  1.70 → 0.031 s  (55×)
      overlay_to_masks 1024², 5×150 cells, Instance-backed: 0.27 → 0.010 s  (27×)

  The `Contour`-backed figure is the one that matters after `load_segmentation`,
  which returns polygon-backed detections: those went through a full-frame
  `rasterio` pass *per cell*. A frame of polygons is now burned in a single
  `rasterio` call, and mask-backed instances are written through their cached
  bounding box. Output is byte-identical — polygons keep rasterio's pixel-centre
  rule via the new `exact_polygons` flag rather than taking the renderers'
  `cv2.fillPoly` shortcut, which fills inclusively and would have dilated every
  cell by a pixel (~+11% area on bacterium-sized cells).

  Fluorescence extraction additionally stopped re-decoding the channel inside the
  per-cell loop (`image.get_channel()` ran once per cell *per channel*) and now
  gathers each cell's pixels inside its bounding box instead of building a
  frame-sized `np.ma.masked_array`. Values are unchanged.


## [0.3.2] - 2025-10-27

### Added
- Trace computation for tracking results.

### Changed
- Faster segmentation-mask rendering; updated video rendering and lineage
  visualization.

### Removed
- The superseded lineage-visualization helpers and the outdated examples.

## [0.3.1] - 2025-08-01

### Added
- Plotly-based cell-lineage rendering, with figure width/height parameters.
  Adds `plotly` as a dependency.

### Changed
- Clarified licensing information.

## [0.3.0] - 2025-07-31

### Added
- YOLO segmentation backend, lineage-tree visualization, tracking utilities, a
  per-detection `score` on `Instance`, and conversion of OMERO raw sources to
  `THWCSequenceSource`.
- `scale()` can run notebooks under an explicitly chosen Jupyter kernel.

### Changed
- Packaging moved to `pyproject.toml`; PyPI publishing from CI.
- Updated the Trackastra and PyUAT tracker integrations.

### Fixed
- Rendering of tracking and segmentation on frames with no detections;
  position, fluorescence and time extractor fixes; pint unit-registry fixes.

## [0.1.18] – [0.2.37] - 2021-12-23 … 2024-12-10

Thirty-seven releases from the GitLab-only era, before this changelog existed;
summarised here as one entry rather than reconstructed individually. Only
0.2.35 and later are available on PyPI — the earlier tags exist in git only.

Over this period acia grew from the initial OMERO-backed prototype into the
library the 0.3 line built on: OMERO image sources, connection handling and ROI
storers; the property-extractor framework with a single shared pint unit
registry (area, length, width, position, fluorescence, time); segmentation and
tracking processors with online/remote and local execution paths; the CTC
import/export formats, including an efficient rewrite; rendering of
segmentation, tracking and video, with scalebar/time overlays, LUT support and
lineage subsampling; and the `scale()` batch-execution helper. Tooling
converged on black, flake8, pylint and pre-commit, with CI on GitLab and
automated version bumps.

## [0.1.0] - 2021-07-30

### Added
- First release on PyPI.
