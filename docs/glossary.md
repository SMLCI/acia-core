# Glossary

The vocabulary `acia` uses, and the distinctions that trip people up most often.

```{glossary}
THWC
  The in-memory axis convention for image sequences: `(T, H, W, C)` — time,
  height, width, channel, with **channel last** and no Z axis. `acia` is a 2D+t
  library; the ND2 and CZI readers reject `Z > 1` rather than guess. Readers
  normalize to this layout by axis *name* where the format provides one, so you
  never have to reason about the file's native ordering.
  {class}`~acia.segm.local.THWCSequenceSource` is the canonical in-memory
  implementation and validates the shape on construction.

  Note this is the *in-memory* convention only.
  {func}`~acia.segm.tiff_export.save_tiff_stack` writes ImageJ-canonical
  channel-first `TCYX` to disk.

ImageSequenceSource
  The central abstraction ({class}`acia.base.ImageSequenceSource`): a sized,
  iterable time series of frames. A subclass only has to implement
  `get_frame()` and the size properties; indexing, slicing, channel selection,
  calibration, cropping, registration and RGB rendering all come from the base
  class. Every reader is one of these, which is why the same code works for ND2,
  CZI, TIFF stacks, folders of TIFFs, OMERO and SMB.

lazy view
  A source derived from another source that computes frames on demand instead of
  copying pixels — {class}`~acia.base.SlicedSequenceSource`,
  {class}`~acia.base.RotatedCropSequenceSource`,
  {class}`~acia.base.RegisteredSequenceSource`,
  {class}`~acia.base.RGBSequenceSource`. Views compose freely (`src[::2][1:]`).
  Call `materialize()` when you deliberately want the whole stack in memory.

Instance
  A **mask-backed** detection ({class}`acia.base.Instance`): it owns a boolean
  mask, and derives its area, centre and polygon from it. This is what
  segmentation backends produce.

Contour
  A **polygon-backed** detection ({class}`acia.base.Contour`): it owns an array
  of coordinates. Cheaper to store and serialize, and what
  {func}`~acia.segm.formats.save_segmentation` writes.

  `Instance` and `Contour` are two representations of the same idea — one
  detected object in one frame — and both satisfy the interface an
  {term}`Overlay` expects. Converting a mask with more than one connected
  component to a polygon is lossy; `Instance.is_fragmented` tells you when that
  applies.

Overlay
  A collection of detections across all frames of a sequence
  ({class}`acia.base.Overlay`), plus an optional time model. Index it by
  detection id (`overlay[contour_id]`) or slice it temporally
  (`overlay[:20]`, which remaps frames to `0..n-1`). Iterating frame by frame is
  `overlay.time_iterator()`.

Processor
  A callable that transforms data. Segmentation processors take a source and
  return an {term}`Overlay`; tracking processors take a source and an overlay and
  return an overlay plus two graphs. Models are built lazily and, with
  `autorelease=True`, GPU memory is freed after each call.

tracklet graph
  A graph whose **nodes are tracklets** — uninterrupted runs of the same cell
  between division events. This is the graph you plot lineages from and compute
  doubling times on.

tracking graph
  A graph whose **nodes are individual detections**, one per cell per frame,
  linked frame to frame. Finer-grained than the tracklet graph;
  {func}`~acia.tracking.utils.tracklet_to_tracking` converts between them.

  Both are returned by every tracking processor, in the order
  `(overlay, tracklet_graph, tracking_graph)`.

extractor
  A {class}`~acia.analysis.PropertyExtractor` that measures one property per
  detection — area, perimeter, length, circularity, fluorescence, time, position.
  {class}`~acia.analysis.ExtractorExecutor` runs a list of them and returns a
  tidy, id-indexed DataFrame. Extractors pull their units from the source's
  calibration automatically; see {doc}`guide/units`.

position
  One field of view within a multi-position acquisition — the ND2 `P` axis and
  the CZI `S` (scene) axis, unified by
  {func}`~acia.segm.open.open_sequence` so notebooks never branch on format. A
  folder whose subfolders each hold per-timepoint TIFFs also exposes one position
  per subfolder.

pixel size / frame interval
  The physical calibration, carried as [pint](https://pint.readthedocs.io)
  quantities on the source. Set them once at load and they survive slicing, flow
  into overlays as per-detection timestamps, and are picked up by the extractors
  — so results come out in µm² and hours rather than px² and frames.
```
