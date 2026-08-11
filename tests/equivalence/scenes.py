"""Deterministic scenes and the filter battery for the equivalence harness.

The scenes deliberately cover the inputs that break geometry code -- cells on
each border, corner cells, single-pixel cells, fragmented masks, degenerate
outlines and the empty overlay -- in both overlay flavours, because
``Instance`` (mask-backed) and ``Contour`` (polygon-backed) reach the geometry
through completely different code.

Everything here is seeded and free of I/O, so a snapshot taken from one machine
is reproducible on another.
"""

from __future__ import annotations

import numpy as np

from acia import Q_
from acia.base import Contour, Instance, Overlay
from acia.segm.filter import (
    AreaFilter,
    BoundaryClosenessFilter,
    CircularityFilter,
    LengthFilter,
    WidthFilter,
)
from acia.segm.local import THWCSequenceSource

#: frame size for the mask-backed scenes (small enough for a fast test run,
#: large enough that a cell is far from filling the frame)
FRAME = 128

#: pixel calibration used by every scene; chosen so the synthetic cells land
#: inside the filter battery's ranges rather than being rejected by the first
#: filter (which would leave the later filters untested).
PIXEL_SIZE = "0.163 micrometer"
FRAME_INTERVAL = "5 minute"


def source(t: int = 1, h: int = FRAME, w: int = FRAME) -> THWCSequenceSource:
    """Blank calibrated source of shape ``(t, h, w, 1)``."""
    return THWCSequenceSource(
        np.zeros((t, h, w, 1), dtype=np.uint8),
        pixel_size=PIXEL_SIZE,
        frame_interval=FRAME_INTERVAL,
    )


def _ellipse(
    label_img: np.ndarray, cx: int, cy: int, a: int, b: int, angle: float, label: int
):
    """Draw a filled ellipse with ``label`` into ``label_img`` (in place)."""
    import cv2

    cv2.ellipse(
        label_img,
        (int(cx), int(cy)),
        (int(a), int(b)),
        float(angle),
        0,
        360,
        int(label),
        -1,
    )


def _rect_contour(
    x0: float, y0: float, w: float, h: float, *, frame: int, id
) -> Contour:
    """Axis-aligned rectangular contour."""
    return Contour(
        np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]),
        score=-1,
        frame=frame,
        id=id,
    )


# --------------------------------------------------------------------------
# scenes whose properties every extractor can compute today -- these carry the
# golden numbers
# --------------------------------------------------------------------------


def scene_instance_basic() -> tuple[Overlay, THWCSequenceSource]:
    """Rod-shaped cells over 3 frames, mask-backed, well inside the frame."""
    rng = np.random.default_rng(20260811)
    masks = []
    uid = 1
    conts: list[Instance] = []
    for frame in range(3):
        li = np.zeros((FRAME, FRAME), np.int32)
        for label in range(1, 13):
            cx = int(rng.integers(20, FRAME - 20))
            cy = int(rng.integers(20, FRAME - 20))
            _ellipse(li, cx, cy, 9, 4, float(rng.uniform(0, 180)), label)
        masks.append(li)
        for label in np.unique(li)[1:]:
            conts.append(Instance(mask=li, frame=frame, label=int(label), id=uid))
            uid += 1
    return Overlay(conts), source(t=len(masks))


def scene_instance_borders() -> tuple[Overlay, THWCSequenceSource]:
    """Cells touching each of the four borders and each of the four corners.

    Exercises ``BoundaryClosenessFilter`` at both extremes and the bbox-crop
    path where a cell's bounding box is clipped by the frame edge.
    """
    li = np.zeros((FRAME, FRAME), np.int32)
    e = FRAME - 1
    placements = [
        (FRAME // 2, 0),  # top edge
        (FRAME // 2, e),  # bottom edge
        (0, FRAME // 2),  # left edge
        (e, FRAME // 2),  # right edge
        (0, 0),  # corners
        (e, 0),
        (0, e),
        (e, e),
        (FRAME // 2, FRAME // 2),  # one safely interior, as a control
    ]
    for label, (cx, cy) in enumerate(placements, start=1):
        _ellipse(li, cx, cy, 8, 5, 0.0, label)

    conts = [
        Instance(mask=li, frame=0, label=int(label), id=int(label))
        for label in np.unique(li)[1:]
    ]
    return Overlay(conts), source()


def scene_instance_small_and_fragmented() -> tuple[Overlay, THWCSequenceSource]:
    """Single-pixel, two-pixel and disconnected (``MultiPolygon``) masks.

    A fragmented mask has no single outline, so ``coordinates``/``draw`` keep
    only its largest part (``acia.utils.largest_polygon``) while ``area`` counts
    every pixel -- an asymmetry any change to the geometry must preserve.
    """
    li = np.zeros((FRAME, FRAME), np.int32)
    li[10, 10] = 1  # single pixel
    li[20, 20:22] = 2  # two pixels
    li[30:34, 30:34] = 3  # small square
    li[50:54, 50:54] = 4  # fragmented: two disconnected blobs share label 4
    li[70:74, 70:74] = 4
    li[90:96, 90:100] = 5  # plain rectangle

    conts = [
        Instance(mask=li, frame=0, label=int(label), id=int(label))
        for label in np.unique(li)[1:]
    ]
    return Overlay(conts), source()


def scene_contour_basic() -> tuple[Overlay, THWCSequenceSource]:
    """Polygon-backed contours: ellipse outlines plus axis-aligned rectangles.

    This is the flavour the public workflow notebooks produce, and it reaches
    the geometry through ``Contour.polygon`` rather than mask polygonisation.
    """
    rng = np.random.default_rng(20260811)
    conts: list[Contour] = []
    uid = 1
    for frame in range(3):
        for _ in range(10):
            t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
            cx, cy = rng.uniform(20, FRAME - 20), rng.uniform(20, FRAME - 20)
            ang = rng.uniform(0, np.pi)
            x, y = 9.0 * np.cos(t), 4.0 * np.sin(t)
            coords = np.stack(
                [
                    x * np.cos(ang) - y * np.sin(ang) + cx,
                    x * np.sin(ang) + y * np.cos(ang) + cy,
                ],
                axis=1,
            )
            conts.append(Contour(coords, score=0.9, frame=frame, id=uid))
            uid += 1
        conts.append(_rect_contour(5, 5, 12, 6, frame=frame, id=uid))
        uid += 1
    return Overlay(conts), source(t=3)


def scene_contour_borders() -> tuple[Overlay, THWCSequenceSource]:
    """Polygon-backed contours flush against each border and corner."""
    e = float(FRAME)
    specs = [
        (0.0, 0.0, 10.0, 6.0),  # top-left corner
        (e - 10, 0.0, 10.0, 6.0),  # top-right
        (0.0, e - 6, 10.0, 6.0),  # bottom-left
        (e - 10, e - 6, 10.0, 6.0),  # bottom-right
        (e / 2, 0.0, 10.0, 6.0),  # top edge
        (e / 2, e - 6, 10.0, 6.0),  # bottom edge
        (e / 2, e / 2, 10.0, 6.0),  # interior control
    ]
    conts = [
        _rect_contour(x, y, w, h, frame=0, id=i)
        for i, (x, y, w, h) in enumerate(specs, start=1)
    ]
    return Overlay(conts), source()


#: scenes every extractor handles today -- the golden snapshot covers these
GOLDEN_SCENES = {
    "instance_basic": scene_instance_basic,
    "instance_borders": scene_instance_borders,
    "instance_small_and_fragmented": scene_instance_small_and_fragmented,
    "contour_basic": scene_contour_basic,
    "contour_borders": scene_contour_borders,
}


# --------------------------------------------------------------------------
# scenes today's extractors CANNOT handle -- see test_property_equivalence
# --------------------------------------------------------------------------


def scene_degenerate() -> tuple[Overlay, THWCSequenceSource]:
    """Outlines with no area: collinear points, and a repeated single point.

    ``LengthFilter``/``WidthFilter`` already treat these as a 0 measurement
    (``_rotated_rect_coords`` returns ``None``), but ``LengthEx``/``WidthEx``
    raise ``AttributeError`` because a degenerate minimum rotated rectangle is a
    ``LineString``/``Point`` with no ``exterior``. Kept as its own scene so the
    asymmetry is pinned rather than discovered again later.
    """
    return (
        Overlay(
            [
                Contour(
                    np.array([[0, 0], [1, 1], [2, 2]]),
                    score=-1,
                    frame=0,
                    id="collinear",
                ),
                Contour(
                    np.array([[5, 5], [5, 5], [5, 5]]), score=-1, frame=0, id="point"
                ),
                Contour(
                    np.array([[0, 0], [4, 0], [0, 4]]),
                    score=-1,
                    frame=0,
                    id="triangle",
                ),
            ]
        ),
        source(),
    )


def scene_absent_label() -> tuple[Overlay, THWCSequenceSource]:
    """An ``Instance`` whose label does not occur in its mask (empty polygon).

    ``PerimeterEx`` raises ``AttributeError`` on it today (``polygon`` is
    ``None``); the filters treat it as distance/measurement 0.
    """
    return (
        Overlay(
            [Instance(mask=np.zeros((FRAME, FRAME), np.int32), frame=0, label=9, id=9)]
        ),
        source(),
    )


def scene_empty() -> tuple[Overlay, THWCSequenceSource]:
    """Empty overlay -- the ``_id_indexed`` / typed-empty-frame path."""
    return Overlay([]), source()


EDGE_SCENES = {
    "degenerate": scene_degenerate,
    "absent_label": scene_absent_label,
    "empty": scene_empty,
}


# --------------------------------------------------------------------------
# filter battery
# --------------------------------------------------------------------------


def filter_battery() -> dict[str, list]:
    """Filter configurations exercised by the equivalence assertions.

    Includes two-sided, one-sided and wide-open bounds, because a ``None`` bound
    takes a different branch in ``CellFilter.accepts`` and round-trips
    differently through ``FilterExplorer``.
    """
    return {
        "area_two_sided": [AreaFilter(Q_(0.1, "um**2"), Q_(4.0, "um**2"))],
        "area_open_above": [AreaFilter(Q_(0.1, "um**2"), None)],
        "area_open_below": [AreaFilter(None, Q_(4.0, "um**2"))],
        "length_two_sided": [LengthFilter(Q_(0.5, "um"), Q_(4.0, "um"))],
        "width_two_sided": [WidthFilter(Q_(0.2, "um"), Q_(2.0, "um"))],
        "circularity_two_sided": [CircularityFilter(vmin=0.4, vmax=0.95)],
        "circularity_open_above": [CircularityFilter(vmin=0.4)],
        "boundary_only": [BoundaryClosenessFilter(Q_(0.5, "um"))],
        "boundary_strict": [BoundaryClosenessFilter(Q_(2.0, "um"))],
        "notebook_combo": [
            AreaFilter(Q_(0.1, "um**2"), Q_(4.0, "um**2")),
            LengthFilter(Q_(0.5, "um"), Q_(4.0, "um")),
            WidthFilter(Q_(0.2, "um"), Q_(2.0, "um")),
            BoundaryClosenessFilter(Q_(0.5, "um")),
        ],
        "nothing_survives": [AreaFilter(Q_(1e6, "um**2"), None)],
        "empty_filter_list": [],
    }


def extractors():
    """The extractor list the golden snapshot is built from.

    Mirrors the lab notebook's list (``01_Segment.ipynb``) minus ``LabelEx``,
    which needs tracking labels these untracked scenes do not carry. Order
    matters: ``CircularityEx`` reads the ``area``/``perimeter`` columns produced
    by the two extractors before it.
    """
    from acia.analysis import (
        AreaEx,
        CircularityEx,
        FrameEx,
        LengthEx,
        PerimeterEx,
        PositionEx,
        TimeEx,
        WidthEx,
    )

    return [
        AreaEx(),
        PerimeterEx(),
        LengthEx(),
        WidthEx(),
        CircularityEx(),
        PositionEx(),
        FrameEx(),
        TimeEx(),
    ]


def filter_extractors():
    """``extractors()`` plus the column the boundary filter needs.

    Kept separate from :func:`extractors` on purpose: the golden snapshot pins
    the column set that existed when it was taken, and ``boundary_closeness``
    was added afterwards. Filtering needs it, the golden must not gain a column
    it never had, so the two lists differ by exactly that one extractor.
    """
    from acia.analysis import BoundaryClosenessEx

    return [*extractors(), BoundaryClosenessEx()]


#: columns the golden snapshot pins, and the tolerance each is held to.
#: ``None`` means exact float equality is required.
COLUMN_TOLERANCE: dict[str, float | None] = {
    "area": None,
    "perimeter": None,
    "length": None,
    "width": None,
    "frame": None,
    "time": None,
    # a division, so float-noise is permitted -- measured at 4.4e-16 today
    "circularity": 1e-15,
    # Positions were originally computed in single precision: `Contour.center`
    # is float32 and pint carried that magnitude through the unit conversion.
    # Converting the column in float64 is more accurate and moves every value by
    # up to ~1e-7 relative. Accepted deliberately (2026-08-11): the column is in
    # micrometres, so this is sub-picometre and below any physical relevance.
    # This is the one tolerance here that permits a *changed* value rather than
    # float-noise around an unchanged one.
    "position_x": 1e-6,
    "position_y": 1e-6,
}
