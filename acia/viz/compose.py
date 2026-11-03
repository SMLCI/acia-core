"""Compose image sequences into a single sequence -- side by side, stacked, gridded.

The one primitive here, :class:`ComposedSequenceSource`, tiles several child
:class:`~acia.base.ImageSequenceSource`\\ s along one axis and is itself an
``ImageSequenceSource``. Because the operation is *closed* over the source type,
it nests: two horizontal composites stacked vertically make a 2x2 grid with no
extra code. Composition is **lazy** -- each output frame is built on demand from
the corresponding child frames, so no full second copy of a movie is held in
memory (relevant given how heavy video rendering already is).

Typical use -- a before/after comparison video::

    before = render_segmentation_mask(source.to_rgb(), overlay_before)
    after = render_segmentation_mask(source.to_rgb(), overlay_after)
    comparison = compose_sequences(
        [before, after], axis="horizontal",
        titles=["before", "after"], gap=4,
    )
    render_video(comparison, "compare.mp4")

:func:`label_sequence` (the titling primitive) is itself implemented *via*
:func:`compose_sequences` -- a title band is just a constant sub-sequence stacked
vertically on top of the panel -- which is why the whole thing is "composed once".
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from acia.base import BaseImage, ImageSequenceSource
from acia.segm.local import LocalImage

logger = logging.getLogger(__name__)

Axis = Literal["horizontal", "vertical"]
Align = Literal["start", "center", "end"]
NFrames = Literal["min", "max"]

# Linux/Colab font paths tried before falling back to Pillow's bundled default,
# matching the rest of acia.viz's text rendering.
_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A truetype font at ``size`` if available, else Pillow's default."""
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)  # Pillow >= 10
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def _frame_rgb(source: ImageSequenceSource, idx: int) -> np.ndarray:
    """Fetch frame ``idx`` of ``source`` as a fresh HxWx3 uint8 RGB array."""
    # Deferred import: acia.viz.__init__ imports this module, so a top-level
    # import of its _to_uint8_rgb would be circular.
    from acia.viz import _to_uint8_rgb

    return _to_uint8_rgb(source.get_frame(idx).raw)


def _align_offset(total: int, size: int, align: Align) -> int:
    """Offset that places a ``size``-long panel within ``total`` per ``align``."""
    if align == "start":
        return 0
    if align == "end":
        return total - size
    return (total - size) // 2  # center


class _ConstantImageSource(ImageSequenceSource):
    """A sequence that yields the same frame ``length`` times (e.g. a title band)."""

    def __init__(self, frame: np.ndarray, length: int):
        self._frame = np.ascontiguousarray(frame, dtype=np.uint8)
        self._length = int(length)
        self._init_calibration()

    def get_frame(self, frame: int) -> BaseImage:
        return LocalImage(self._frame)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_frame(i)

    @property
    def num_channels(self) -> int:
        return int(self._frame.shape[2])

    @property
    def size_c(self) -> int:
        return int(self._frame.shape[2])

    @property
    def size_t(self) -> int:
        return self._length

    @property
    def size_h(self) -> int:
        return int(self._frame.shape[0])

    @property
    def size_w(self) -> int:
        return int(self._frame.shape[1])


class ComposedSequenceSource(ImageSequenceSource):
    """Lazily tile several child sequences along one axis into one sequence.

    Each output frame is composed on demand: the ``k``-th child's frame is
    placed into a shared canvas along ``axis`` (``"horizontal"`` -> panels
    left-to-right, ``"vertical"`` -> top-to-bottom), padded on the perpendicular
    axis to the largest panel (positioned by ``align``) and separated by ``gap``
    pixels. The result is an :class:`~acia.base.ImageSequenceSource`, so it can
    itself be a child of another composition (grids via nesting).

    Args:
        sources: the child sequences (at least one).
        axis: ``"horizontal"`` (side by side) or ``"vertical"`` (stacked).
        gap: pixels of separation between adjacent panels (0 = flush).
        gap_color: RGB fill for the gap strips (only drawn when it differs from
            ``pad_color``).
        align: perpendicular placement of shorter panels -- ``"start"``,
            ``"center"`` (default), or ``"end"``.
        pad_color: RGB fill for the canvas / perpendicular padding.
        n_frames: reconcile differing child lengths by the ``"min"`` (truncate,
            default) or ``"max"`` (hold each panel's last frame) length. A
            warning is logged when child lengths differ.

    Time/pixel calibration is inherited from the first child.
    """

    def __init__(
        self,
        sources: Sequence[ImageSequenceSource],
        *,
        axis: Axis = "horizontal",
        gap: int = 0,
        gap_color: tuple[int, int, int] = (0, 0, 0),
        align: Align = "center",
        pad_color: tuple[int, int, int] = (0, 0, 0),
        n_frames: NFrames = "min",
    ):
        sources = list(sources)
        if not sources:
            raise ValueError("compose requires at least one source")
        if axis not in ("horizontal", "vertical"):
            raise ValueError(f"axis must be 'horizontal' or 'vertical', got {axis!r}")
        if align not in ("start", "center", "end"):
            raise ValueError(f"align must be 'start', 'center' or 'end', got {align!r}")
        if n_frames not in ("min", "max"):
            raise ValueError(f"n_frames must be 'min' or 'max', got {n_frames!r}")
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")

        self._sources = sources
        self._axis = axis
        self._gap = int(gap)
        self._gap_color = gap_color
        self._align = align
        self._pad_color = pad_color

        lengths = [len(s) for s in sources]
        if min(lengths) != max(lengths):
            logger.warning(
                "Composing sequences of unequal length %s; using %s (%d frames).",
                lengths,
                n_frames,
                min(lengths) if n_frames == "min" else max(lengths),
            )
        self._length = min(lengths) if n_frames == "min" else max(lengths)
        self._lengths = lengths

        # Panel sizes from each child's first frame (assumed constant per source).
        self._sizes = [_frame_rgb(s, 0).shape[:2] for s in sources]
        heights = [h for h, _ in self._sizes]
        widths = [w for _, w in self._sizes]
        n_gaps = (len(sources) - 1) * self._gap
        if axis == "horizontal":
            self._out_h = max(heights)
            self._out_w = sum(widths) + n_gaps
        else:
            self._out_h = sum(heights) + n_gaps
            self._out_w = max(widths)

        # Inherit calibration from the first child that carries it (a title band
        # is uncalibrated, so "first child" alone would drop timing). Timepoints
        # are sliced to the composite's own length so they stay frame-aligned
        # after length reconciliation.
        ref_timepoints = None
        ref_pixel_size = None
        for s in sources:
            tp = s.timepoints
            if ref_timepoints is None and tp is not None and len(tp) >= self._length:
                ref_timepoints = tp[: self._length]
            if ref_pixel_size is None and s.pixel_size is not None:
                ref_pixel_size = s.pixel_size
        self._init_calibration(timepoints=ref_timepoints, pixel_size=ref_pixel_size)

    def _child_frame(
        self, source: ImageSequenceSource, i: int, length: int
    ) -> np.ndarray:
        """Frame ``i`` of ``source``, holding its last frame past its own end."""
        return _frame_rgb(source, min(i, length - 1))

    def get_frame(self, frame: int) -> BaseImage:
        canvas = np.empty((self._out_h, self._out_w, 3), dtype=np.uint8)
        canvas[:] = self._pad_color
        draw_gap = self._gap > 0 and self._gap_color != self._pad_color

        cursor = 0
        for k, source in enumerate(self._sources):
            panel = self._child_frame(source, frame, self._lengths[k])
            ph, pw = panel.shape[:2]
            if self._axis == "horizontal":
                y = _align_offset(self._out_h, ph, self._align)
                canvas[y : y + ph, cursor : cursor + pw] = panel
                cursor += pw
                if k < len(self._sources) - 1:
                    if draw_gap:
                        canvas[:, cursor : cursor + self._gap] = self._gap_color
                    cursor += self._gap
            else:
                x = _align_offset(self._out_w, pw, self._align)
                canvas[cursor : cursor + ph, x : x + pw] = panel
                cursor += ph
                if k < len(self._sources) - 1:
                    if draw_gap:
                        canvas[cursor : cursor + self._gap, :] = self._gap_color
                    cursor += self._gap

        return LocalImage(canvas)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_frame(i)

    @property
    def num_channels(self) -> int:
        return 3

    @property
    def size_c(self) -> int:
        return 3

    @property
    def size_t(self) -> int:
        return self._length

    @property
    def size_h(self) -> int:
        return int(self._out_h)

    @property
    def size_w(self) -> int:
        return int(self._out_w)


def label_sequence(
    source: ImageSequenceSource,
    title: str,
    *,
    height: int = 28,
    bg_color: tuple[int, int, int] = (30, 30, 30),
    text_color: tuple[int, int, int] = (255, 255, 255),
    font_size: int | None = None,
) -> ComposedSequenceSource:
    """Prepend a titled caption band above every frame of ``source``.

    The band is a constant sub-sequence stacked on top of the panel via
    :func:`compose_sequences`, so the result is itself composable (e.g. several
    labelled panels can be placed side by side).

    Args:
        source: the sequence to caption.
        title: caption text, drawn centered on the band.
        height: band height in pixels.
        bg_color: RGB band background.
        text_color: RGB text color.
        font_size: text size; defaults to ~60% of ``height``.

    Returns:
        A :class:`ComposedSequenceSource` of ``[band, source]`` stacked
        vertically, ``height`` pixels taller than ``source``.
    """
    width = source.size_w
    band = np.empty((height, width, 3), dtype=np.uint8)
    band[:] = bg_color

    font = _load_font(font_size if font_size is not None else max(8, int(height * 0.6)))
    pil = Image.fromarray(band)
    draw = ImageDraw.Draw(pil)
    left, top, right, bottom = draw.textbbox((0, 0), title, font=font)
    tw, th = right - left, bottom - top
    draw.text(
        ((width - tw) // 2 - left, (height - th) // 2 - top),
        title,
        fill=text_color,
        font=font,
    )
    band_source = _ConstantImageSource(np.array(pil, dtype=np.uint8), len(source))
    return compose_sequences(
        [band_source, source], axis="vertical", align="center", pad_color=bg_color
    )


def compose_sequences(
    sources: Sequence[ImageSequenceSource],
    *,
    axis: Axis = "horizontal",
    titles: Sequence[str] | None = None,
    gap: int = 0,
    gap_color: tuple[int, int, int] = (0, 0, 0),
    align: Align = "center",
    pad_color: tuple[int, int, int] = (0, 0, 0),
    n_frames: NFrames = "min",
) -> ComposedSequenceSource:
    """Tile several image sequences into one -- side by side or stacked.

    A thin front-end over :class:`ComposedSequenceSource` that additionally
    captions each panel when ``titles`` is given (via :func:`label_sequence`).
    See :class:`ComposedSequenceSource` for the layout parameters.

    Args:
        sources: the child sequences (at least one).
        axis: ``"horizontal"`` (side by side) or ``"vertical"`` (stacked).
        titles: optional per-panel captions; must match ``sources`` in length.
            Each source is wrapped with :func:`label_sequence` before tiling.
        gap: pixels between adjacent panels.
        gap_color: RGB fill for the gap strips.
        align: perpendicular placement of shorter panels.
        pad_color: RGB fill for the canvas / perpendicular padding.
        n_frames: ``"min"`` (truncate) or ``"max"`` (hold last frame) length
            reconciliation for unequal-length children.

    Returns:
        A composed :class:`ComposedSequenceSource`, ready for
        :func:`acia.viz.render_video` or further composition.
    """
    sources = list(sources)
    if titles is not None:
        if len(titles) != len(sources):
            raise ValueError(
                f"titles ({len(titles)}) must match sources ({len(sources)})"
            )
        sources = [label_sequence(s, t) for s, t in zip(sources, titles, strict=True)]

    return ComposedSequenceSource(
        sources,
        axis=axis,
        gap=gap,
        gap_color=gap_color,
        align=align,
        pad_color=pad_color,
        n_frames=n_frames,
    )


# --- time stamp & scale bar annotation ---------------------------------------

Placement = Literal["border_bottom", "border_top", "overlay"]


def _annotation_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """DejaVu Sans (ships with matplotlib, has ``µ``) at ``size``, else fallback."""
    try:
        from matplotlib import font_manager

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:  # pragma: no cover - matplotlib font cache unavailable
        return _load_font(size)


def _pixel_size_x(pixel_size):
    """The x (width) component of a scalar or ``[y, x]`` pint pixel size."""
    magnitude = np.atleast_1d(np.asarray(pixel_size.magnitude, dtype=float))
    return float(magnitude[-1]) * pixel_size.units


class AnnotatedSequenceSource(ImageSequenceSource):
    """A lazy view of ``source`` with a time stamp and/or scale bar burnt in.

    Built by :func:`annotate_sequence`, which documents the parameters. Frames
    are drawn on demand, so annotating a movie holds no second copy of it.
    The calibration of the view is the one used for the annotation.
    """

    def __init__(
        self,
        source: ImageSequenceSource,
        *,
        time_texts: list[str] | None,
        bar_pixels: int | None,
        bar_label: str | None,
        placement: Placement,
        band_height: int,
        font_size: int,
        color: tuple[int, int, int],
        band_color: tuple[int, int, int],
        margin: int,
        timepoints=None,
        pixel_size=None,
    ):
        self._source = source
        self._time_texts = time_texts
        self._bar_pixels = bar_pixels
        self._bar_label = bar_label
        self._placement = placement
        self._band_height = int(band_height)
        self._font = _annotation_font(int(font_size))
        self._color = tuple(int(c) for c in color)
        self._band_color = tuple(int(c) for c in band_color)
        self._margin = int(margin)
        h, w = _frame_rgb(source, 0).shape[:2]
        self._h, self._w = int(h), int(w)
        self._init_calibration(timepoints=timepoints, pixel_size=pixel_size)

    # -- layout -------------------------------------------------------------

    def _text_size(
        self, draw: ImageDraw.ImageDraw, text: str
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=self._font)
        return int(left), int(top), int(right - left), int(bottom - top)

    def _draw_time(self, draw, text: str, x0: int, y0: int, y1: int, boxed: bool):
        """Draw ``text`` left-aligned at ``x0``, vertically centred in ``[y0, y1)``."""
        left, top, tw, th = self._text_size(draw, text)
        y = y0 + (y1 - y0 - th) // 2
        if boxed:
            pad = max(2, self._margin // 2)
            draw.rectangle(
                (x0 - pad, y - pad, x0 + tw + pad, y + th + pad), fill=self._band_color
            )
        draw.text((x0 - left, y - top), text, fill=self._color, font=self._font)

    def _draw_bar(self, draw, x1: int, y0: int, y1: int, boxed: bool):
        """Draw the bar (label above it) right-aligned at ``x1`` in ``[y0, y1)``."""
        assert self._bar_pixels is not None
        thickness = max(2, int(round((y1 - y0) * 0.14)))
        gap = max(2, thickness // 2)
        if self._bar_label:
            left, top, tw, th = self._text_size(draw, self._bar_label)
        else:
            left = top = tw = th = 0
        block_h = th + (gap if th else 0) + thickness
        by0 = y0 + (y1 - y0 - block_h) // 2
        bx0 = x1 - self._bar_pixels
        width = max(self._bar_pixels, tw)
        if boxed:
            pad = max(2, self._margin // 2)
            cx = bx0 + self._bar_pixels / 2
            draw.rectangle(
                (
                    cx - width / 2 - pad,
                    by0 - pad,
                    cx + width / 2 + pad,
                    by0 + block_h + pad,
                ),
                fill=self._band_color,
            )
        if self._bar_label:
            tx = bx0 + (self._bar_pixels - tw) / 2
            draw.text(
                (tx - left, by0 - top),
                self._bar_label,
                fill=self._color,
                font=self._font,
            )
        ry = by0 + block_h - thickness
        draw.rectangle((bx0, ry, x1 - 1, ry + thickness - 1), fill=self._color)

    def _annotate(
        self,
        canvas: np.ndarray,
        frame: int,
        band: tuple[int, int],
        *,
        boxed: bool,
        time: bool = True,
        bar: bool = True,
    ) -> np.ndarray:
        """Draw the time (left) and/or the bar (right) in the rows ``band``."""
        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        y0, y1 = band
        if time and self._time_texts is not None and frame < len(self._time_texts):
            self._draw_time(draw, self._time_texts[frame], self._margin, y0, y1, boxed)
        if bar and self._bar_pixels:
            self._draw_bar(draw, self._w - self._margin, y0, y1, boxed)
        return np.asarray(pil, dtype=np.uint8)

    # -- ImageSequenceSource --------------------------------------------------

    def get_frame(self, frame: int) -> BaseImage:
        image = _frame_rgb(self._source, frame)
        bh = self._band_height
        if self._placement == "overlay":
            # time in a strip at the top, scale bar in a strip at the bottom
            top = (self._margin, self._margin + bh)
            bottom = (self._h - self._margin - bh, self._h - self._margin)
            out = self._annotate(image, frame, top, boxed=True, bar=False)
            out = self._annotate(out, frame, bottom, boxed=True, time=False)
            return LocalImage(np.ascontiguousarray(out))

        canvas = np.empty((self._h + bh, self._w, 3), dtype=np.uint8)
        canvas[:] = self._band_color
        if self._placement == "border_top":
            canvas[bh:] = image
            band = (0, bh)
        else:
            canvas[: self._h] = image
            band = (self._h, self._h + bh)
        return LocalImage(self._annotate(canvas, frame, band, boxed=False))

    def __len__(self) -> int:
        return len(self._source)

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_frame(i)

    @property
    def num_channels(self) -> int:
        return 3

    @property
    def size_c(self) -> int:
        return 3

    @property
    def size_t(self) -> int:
        return len(self._source)

    @property
    def size_h(self) -> int:
        return self._h if self._placement == "overlay" else self._h + self._band_height

    @property
    def size_w(self) -> int:
        return self._w


def annotate_sequence(
    source: ImageSequenceSource,
    *,
    timepoints=None,
    pixel_size=None,
    show_time: bool = True,
    time_format: str = "{H:02}h {M:02}m",
    scalebar_length: str | object | None = "10 µm",
    placement: Placement = "border_bottom",
    band_height: int | None = None,
    font_size: int | None = None,
    color: tuple[int, int, int] = (255, 255, 255),
    band_color: tuple[int, int, int] = (0, 0, 0),
    margin: int | None = None,
) -> AnnotatedSequenceSource:
    """Burn a time stamp and a scale bar into every frame of a movie.

    Typical use, right before writing a video::

        video = render_segmentation_mask(source.to_rgb(), overlay)
        video = annotate_sequence(video, timepoints=source.timepoints,
                                  pixel_size=source.pixel_size)
        render_video(video, "seg.mp4")

    Pass ``timepoints`` / ``pixel_size`` from the *original* image source: the
    rendered sequences (``render_segmentation_mask`` & co.) do not carry the
    calibration along. When left ``None`` they are read from ``source``.

    Args:
        source: the frames to annotate (any channel layout; drawn as RGB uint8).
        timepoints: per-frame pint quantities (e.g. ``source.timepoints``).
            Without them no time stamp is drawn (with a warning if ``show_time``).
        pixel_size: pint length per pixel, scalar or ``[y, x]`` (the x part is
            used). Without it no scale bar is drawn (with a warning).
        show_time: draw the time stamp.
        time_format: :func:`acia.viz.utils.strfdelta` format, e.g.
            ``"{H:02}h {M:02}m"`` (hours keep counting past a day) or
            ``"{M} min"``.
        scalebar_length: bar length as a pint-parsable string or quantity
            (``"10 µm"``); ``None`` draws no scale bar.
        placement: ``"border_bottom"`` (default) adds a band below the image —
            time on the left, scale bar on the right — so nothing covers the
            cells. ``"border_top"`` puts the band above. ``"overlay"`` draws onto
            the image itself (time top-left, bar bottom-right, each on a small
            ``band_color`` box), keeping the frame size.
        band_height: band height in pixels; defaults to ~9 % of the frame
            height, at least 22 px.
        font_size: text size; defaults to ~40 % of ``band_height``.
        color: RGB colour of text and bar.
        band_color: RGB colour of the band (or of the boxes for ``"overlay"``).
        margin: distance from the image edges; defaults to ~3 % of the width.

    Returns:
        A lazy :class:`AnnotatedSequenceSource` (frames drawn on demand), taller
        by ``band_height`` for the border placements.
    """
    from acia import ureg
    from acia.viz.utils import strfdelta

    if placement not in ("border_bottom", "border_top", "overlay"):
        raise ValueError(
            "placement must be 'border_bottom', 'border_top' or 'overlay', "
            f"got {placement!r}"
        )
    if timepoints is None:
        timepoints = source.timepoints
    if pixel_size is None:
        pixel_size = source.pixel_size

    h, w = _frame_rgb(source, 0).shape[:2]
    if band_height is None:
        band_height = max(22, int(round(h * 0.09)))
        # keep the annotated frame height even (yuv420p video needs it)
        band_height += (h + band_height) % 2
    if font_size is None:
        font_size = max(8, int(round(band_height * 0.4)))
    if margin is None:
        margin = max(3, int(round(w * 0.03)))

    time_texts = None
    if show_time:
        if timepoints is None:
            logger.warning(
                "annotate_sequence: no timepoints (pass timepoints=source.timepoints "
                "from the original source); drawing no time stamp."
            )
        else:
            from datetime import timedelta

            seconds = np.asarray(
                ureg.Quantity(timepoints).to("s").magnitude, dtype=float
            )
            if len(seconds) < len(source):
                logger.warning(
                    "annotate_sequence: %d timepoints for %d frames; the last frames "
                    "get no time stamp.",
                    len(seconds),
                    len(source),
                )
            time_texts = [
                strfdelta(timedelta(seconds=float(s)), fmt=time_format) for s in seconds
            ]

    bar_pixels = None
    bar_label = None
    if scalebar_length is not None:
        if pixel_size is None:
            logger.warning(
                "annotate_sequence: no pixel size (pass pixel_size=source.pixel_size "
                "from the original source); drawing no scale bar."
            )
        else:
            length = ureg.Quantity(scalebar_length)
            px = _pixel_size_x(ureg.Quantity(pixel_size))
            bar_pixels = int(round(float((length / px).to("dimensionless").magnitude)))
            bar_label = f"{length:~P}"
            if bar_pixels < 2:
                logger.warning(
                    "annotate_sequence: a %s scale bar is %d px at %s/pixel; drawing "
                    "no scale bar (choose a longer scalebar_length).",
                    bar_label,
                    bar_pixels,
                    px,
                )
                bar_pixels = None
            elif bar_pixels > w - 2 * margin:
                logger.warning(
                    "annotate_sequence: a %s scale bar (%d px) is wider than the "
                    "%d px frame; choose a shorter scalebar_length.",
                    bar_label,
                    bar_pixels,
                    w,
                )

    return AnnotatedSequenceSource(
        source,
        time_texts=time_texts,
        bar_pixels=bar_pixels,
        bar_label=bar_label,
        placement=placement,
        band_height=band_height,
        font_size=font_size,
        color=color,
        band_color=band_color,
        margin=margin,
        timepoints=timepoints,
        pixel_size=pixel_size,
    )
