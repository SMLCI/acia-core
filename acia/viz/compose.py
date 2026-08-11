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
