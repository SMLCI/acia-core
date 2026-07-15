# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Add your changes under **[Unreleased]**; the release workflow promotes that
section to a dated version entry automatically.

## [Unreleased]

### Added
- `FlowposeRTSegmenter` (`acia.segm.processor.flowpose_rt`): omnipose-compatible
  segmentation backed by the lightweight `flowpose-rt` package (no
  cellpose/omnipose/numba at runtime), selectable via the new `flowpose-rt` extra.
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
- Migrated CI/CD to GitHub Actions with automated, OIDC-based PyPI releases and
  GitHub Pages documentation.

## [0.1.0] - 2021-07-30

### Added
- First release on PyPI.
