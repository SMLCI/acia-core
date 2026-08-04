# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Add your changes under **[Unreleased]**; the release workflow promotes that
section to a dated version entry automatically.

## [Unreleased]

### Added
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
    with a reference policy: `"fixed"` (today's behavior), `"reanchor"`
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
  - `ImageSequenceSource.register(..., on_missing=...)`, also on
    `load_registration`: `"warn"` (default), `"nearest"` (correct a failed
    frame with its neighbor's transform instead of exporting it uncorrected,
    which is off by the full accumulated drift), or `"error"`.
    `RegisteredSequenceSource.missing_frames` reports every gap at once.
- `FlowposeRTSegmenter` (`acia.segm.processor.flowpose_rt`): omnipose-compatible
  segmentation backed by the lightweight `flowpose-rt` package (no
  cellpose/omnipose/numba at runtime), selectable via the new `flowpose-rt` extra.
  Processes frames in configurable batches (`batch_size`, default 20) with a
  `tqdm` progress bar, matching `OmniposeSegmenter`'s existing behavior.
  `weights_path=...` loads a local checkpoint (e.g. a fine-tuned model) instead of
  the downloaded zoo weights; `model` then names the zoo entry whose preprocessing
  contract the checkpoint follows.
- `ImageSequenceSource.to_channel(c)`: a generic lazy single-channel view
  (equivalent to `self[..., c]`), now available on every source implementation
  (e.g. `LocalSequenceSource`), not just `THWCSequenceSource`.
- fsspec-based remote image sources: read TIFFs from SMB/SAMBA shares (and any
  fsspec backend) via `SambaSequenceSource` / `LocalSequenceSource`, plus
  `list_sequence_sources` for folder discovery and a per-user credentials store
  (`acia.config`).
- Source-aware notebook scaling: `acia.analysis.scale` accepts OMERO ids, file
  paths/URLs, or parameter dicts, with per-type default execution naming.
- Unit-aware property-extractor tables via pint-pandas:
  `ExtractorExecutor.execute(units="none"|"header"|"pint")` plus
  `attach_units` / `strip_units` / `units_in_header` converters.
- numpy-style `(T, H, W, C)` indexing on image sequence sources
  (`src[::2, 100:200, 50:150, 0]`, composable lazy views) and temporal slicing of
  overlays (`overlay[:20]`).
- pint calibration defined at load: `pixel_size` and `frame_interval`/`timepoints`
  flow through slices, into overlays (per-detection `contour.time`) and are pulled
  automatically by the extractors (explicit `input_unit` still overrides).

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
- Migrated CI/CD to GitHub Actions with automated, OIDC-based PyPI releases and
  GitHub Pages documentation.

### Fixed
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

## [0.1.0] - 2021-07-30

### Added
- First release on PyPI.
