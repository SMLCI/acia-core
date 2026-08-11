# Slicing and calibration

## Numpy-style indexing

Image sequences support **numpy-style indexing** over the `(T, H, W, C)` axes,
implemented once on {class}`~acia.base.ImageSequenceSource` so every source —
local, in-memory, THWC, ND2, CZI, folder, OMERO, SMB — gets it:

```python
src[5]                        # the frame at index 5 (a BaseImage)
src[::2]                      # a view: every second frame
src[3:23]                     # a view: frames 3..22
src[:, 100:200, 50:150]       # spatial crop (all frames)
src[..., 0]                   # select channel 0 across all frames
src[::2, 100:200, 50:150, 0]  # subsample + crop + channel, composed
```

An integer on the `T` axis returns that frame; a slice or list returns a lazy
view sequence. Views compose (`src[::2][1:]`) and never copy pixel data eagerly —
which is what makes it cheap to subsample a hundred-gigabyte acquisition down to
something you can iterate on.

## Physical calibration

Define the **imaging interval** and **pixel size** in pint units once, at load.
They become metadata on the source and flow through slices and into extractors:

```python
from acia import ureg
from acia.segm.local import LocalSequenceSource

src = LocalSequenceSource(
    "exp.tif",
    pixel_size=0.065 * ureg.micrometer,   # space
    frame_interval=10 * ureg.minute,      # time
)

src.timepoints          # [0, 10, 20, ...] minute  (per frame)
src.pixel_size          # 0.065 micrometer
```

Slicing transforms the calibration automatically:

* temporal subsampling scales the interval — `src[::2].timepoints` is
  `[0, 20, 40, ...]` minute;
* a spatial **crop** keeps `pixel_size`; a uniform spatial **step** scales it —
  `src[:, ::2, ::2].pixel_size` is `0.13 micrometer`.

You can also tag an existing source or overlay fluently:
`src.with_frame_interval("10 minute")`, `src.with_timepoints(...)`,
`src.with_pixel_size(0.065 * ureg.micrometer)`.

For TIFFs, {func}`~acia.segm.tiff_metadata.read_tiff_calibration` reads OME-XML
or ImageJ calibration straight from the file headers, and the TIFF sources call
it lazily — so often you do not need to pass anything. Explicit constructor
arguments always win over file metadata.

## Overlays and detection timestamps

Overlays support **temporal slicing** with slices and lists (`overlay[:20]` cuts
after 20 frames; `overlay[::2]` subsamples), remapping frames to `0..n-1`.
Indexing by a single id is unchanged (`overlay[contour_id]`). When an overlay
carries a time model, every detection gets a pint timestamp:

```python
overlay = overlay.with_frame_interval(10 * ureg.minute)
overlay[:20]          # first 20 frames, frames remapped
overlay.timestamps    # pint array, one per contour
contour.time          # pint Quantity for a single detection
```

## Extractors pull the calibration

Spatial extractors derive their unit from `images.pixel_size` and the time
extractor reads `images.timepoints` (or the overlay's), so the common case needs
no per-extractor units:

```python
from acia.analysis import ExtractorExecutor, FrameEx, AreaEx, TimeEx

df = ExtractorExecutor().execute(overlay, src, extractors=[
    FrameEx(), AreaEx(), TimeEx(),     # no input_unit needed
])
```

Precedence is **explicit `input_unit` > source calibration > default**, so the
classic style keeps working and overrides the source:

```python
AreaEx(input_unit=(0.065 * ureg.micrometer) ** 2)   # explicit wins
TimeEx(input_unit="10 * minute")                    # legacy frame * interval
```

See {doc}`units` for how the resulting DataFrame can expose those units.
