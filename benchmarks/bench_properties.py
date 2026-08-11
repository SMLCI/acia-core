"""Benchmark for single-cell property extraction and cell filtering.

Run unchanged before and after each optimisation stage so the numbers stay
comparable:

    python benchmarks/bench_properties.py
    python benchmarks/bench_properties.py --cells 300 --frames 5 --size 1024

Both overlay flavours are measured, because they reach the geometry through
different code and an optimisation can help one and not the other:

* ``Instance`` -- mask-backed, what ``overlay_from_masks`` produces from a
  segmentation (the lab notebooks);
* ``Contour`` -- polygon-backed (the public workflow notebooks).

Filtering is measured on a *fresh* overlay, matching how a notebook runs it: a
warmed overlay hides the cost of the first geometry access behind the cache that
extraction happened to fill.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

from acia import Q_
from acia.analysis import (
    AreaEx,
    BoundaryClosenessEx,
    CircularityEx,
    ExtractorExecutor,
    FrameEx,
    LengthEx,
    PerimeterEx,
    PositionEx,
    WidthEx,
)
from acia.base import Contour, Overlay
from acia.segm.filter import (
    AreaFilter,
    BoundaryClosenessFilter,
    LengthFilter,
    WidthFilter,
    apply_cell_filters,
)
from acia.segm.formats import overlay_from_masks
from acia.segm.local import THWCSequenceSource

#: calibration chosen so the synthetic cells fall INSIDE the filter ranges --
#: otherwise the first filter rejects everything and the rest are never
#: exercised, which silently understates the filtering cost.
PIXEL_SIZE = "0.163 micrometer"

CELL_SEMI_MAJOR = 12
CELL_SEMI_MINOR = 5


def build_masks(frames: int, cells: int, size: int, seed: int = 0) -> np.ndarray:
    """Label-image stack with ``cells`` rod-shaped cells per frame."""
    import cv2

    rng = np.random.default_rng(seed)
    stack = []
    for _ in range(frames):
        label_img = np.zeros((size, size), np.int32)
        for label in range(1, cells + 1):
            cx = int(rng.integers(30, size - 30))
            cy = int(rng.integers(30, size - 30))
            cv2.ellipse(
                label_img,
                (cx, cy),
                (CELL_SEMI_MAJOR, CELL_SEMI_MINOR),
                float(rng.uniform(0, 180)),
                0,
                360,
                int(label),
                -1,
            )
        stack.append(label_img)
    return np.array(stack)


def build_contour_overlay(frames: int, cells: int, size: int, seed: int = 0) -> Overlay:
    """The same scene as polygon-backed contours (40-vertex ellipse outlines)."""
    rng = np.random.default_rng(seed)
    contours = []
    uid = 1
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    for frame in range(frames):
        for _ in range(cells):
            cx, cy = rng.uniform(30, size - 30), rng.uniform(30, size - 30)
            rot = rng.uniform(0, np.pi)
            x = CELL_SEMI_MAJOR * np.cos(angles)
            y = CELL_SEMI_MINOR * np.sin(angles)
            coords = np.stack(
                [
                    x * np.cos(rot) - y * np.sin(rot) + cx,
                    x * np.sin(rot) + y * np.cos(rot) + cy,
                ],
                axis=1,
            )
            contours.append(Contour(coords, score=0.9, frame=frame, id=uid))
            uid += 1
    return Overlay(contours)


def extractors():
    """The lab notebook's geometry extractors (``01_Segment.ipynb``).

    Includes ``BoundaryClosenessEx``, which backs ``BoundaryClosenessFilter``.
    Extraction therefore does slightly more work than it used to -- that work
    moved here out of the filtering step, which is exactly the change being
    measured, so both halves have to stay in the total.
    """
    return [
        AreaEx(),
        PerimeterEx(),
        LengthEx(),
        WidthEx(),
        CircularityEx(),
        PositionEx(),
        FrameEx(),
        BoundaryClosenessEx(),
    ]


def filters():
    """The lab notebook's default filter set.

    The width bound is widened from the notebook's ``1.75 um`` to ``2.0 um``:
    rasterising an ellipse of semi-minor axis 5 yields an 11 px wide mask
    (1.79 µm here), so the notebook's own bound would reject nearly every
    synthetic cell at the third filter and short-circuit the fourth --
    understating the cost of a filter set that in reality runs to completion.
    """
    return [
        AreaFilter(Q_(1, "um**2"), Q_(15, "um**2")),
        LengthFilter(Q_(1, "um"), Q_(8, "um")),
        WidthFilter(Q_(0.8, "um"), Q_(2.0, "um")),
        BoundaryClosenessFilter(Q_(0.5, "um")),
    ]


def _time(fn):
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def _apply_filters(overlay, filter_list, images, table):
    """Call ``apply_cell_filters`` across the signature change in stage A."""
    try:
        return apply_cell_filters(overlay, filter_list, properties=table)
    except TypeError:
        return apply_cell_filters(overlay, filter_list, images=images)


def run(frames: int, cells: int, size: int) -> None:
    total_cells = frames * cells
    images = THWCSequenceSource(
        np.zeros((frames, size, size, 1), np.uint8),
        pixel_size=PIXEL_SIZE,
        frame_interval="5 minute",
    )
    masks = build_masks(frames, cells, size)

    print(
        f"\n{frames} frames x {cells} cells = {total_cells} detections, "
        f"{size}x{size}, pixel_size={PIXEL_SIZE}"
    )
    print("=" * 74)

    for flavour, make_overlay in (
        ("Instance (mask-backed)", lambda: overlay_from_masks(masks)),
        (
            "Contour  (polygon-backed)",
            lambda: build_contour_overlay(frames, cells, size),
        ),
    ):
        print(f"\n{flavour}")

        build_time, overlay = _time(make_overlay)
        extract_time, table = _time(
            lambda ov=overlay: ExtractorExecutor().execute(ov, images, extractors())
        )

        # filtering on a FRESH overlay: the notebook's real situation
        fresh = make_overlay()
        fresh_table = table.copy()
        fresh_table.index = [c.id for c in fresh]
        filter_time, kept = _time(
            lambda ov=fresh, tbl=fresh_table: _apply_filters(ov, filters(), images, tbl)
        )

        total = extract_time + filter_time
        print(f"  build overlay        {build_time:8.3f} s")
        print(
            f"  extraction           {extract_time:8.3f} s   "
            f"{extract_time / total_cells * 1e3:8.3f} ms/cell"
        )
        print(
            f"  filtering (cold)     {filter_time:8.3f} s   "
            f"{filter_time / total_cells * 1e3:8.3f} ms/cell"
        )
        print(
            f"  TOTAL                {total:8.3f} s   "
            f"{total / total_cells * 1e3:8.3f} ms/cell"
            f"      ({len(kept.contours)}/{total_cells} cells kept)"
        )
        per_roi = total / total_cells * 150_000
        print(
            f"  -> extrapolated to a 150k-detection ROI: {per_roi / 60:6.1f} min"
            f"   |  107-ROI batch: {per_roi * 107 / 3600:6.1f} h"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--cells", type=int, default=300)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument(
        "--warnings", action="store_true", help="do not silence library warnings"
    )
    args = parser.parse_args()

    if not args.warnings:
        warnings.filterwarnings("ignore")

    run(args.frames, args.cells, args.size)


if __name__ == "__main__":
    main()
