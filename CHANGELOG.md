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

### Added

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
  - `RegistrationDashboard(method_kwargs=..., reference_mode=...)` and
    `batch_apply(..., method_kwargs=...)` for per-position settings —
    registration method settings are now a supported argument rather than
    something a caller has to reach in and patch.
  - `batch_apply(sources={position: source})` registers a supplied (e.g. lazily
    sliced) source per position instead of the full position.
  - `ImageSequenceSource.register(..., on_missing=...)`, also on
    `load_registration`: `"warn"` (default), `"nearest"` (correct a failed
    frame with its neighbor's transform instead of exporting it uncorrected,
    which is off by the full accumulated drift), or `"error"`.
    `RegisteredSequenceSource.missing_frames` reports every gap at once.

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
