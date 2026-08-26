"""Benchmark for Overlay -> label-mask rasterisation.

This is the conversion every tracking processor runs before it can hand the
segmentation to its backend (``TrackastraTracker``, ``LapTrack*``, ``PyUAT``,
``Ultrack`` all call ``overlay_to_masks``), and the same loop the CTC exporters
use on the way out.

Run unchanged before and after each optimisation stage so the numbers stay
comparable:

    python benchmarks/bench_overlay_to_masks.py
    python benchmarks/bench_overlay_to_masks.py --frames 20 --cells 200 --size 1024

Both overlay flavours are measured, because they reach the rasteriser through
different code and an optimisation can help one and not the other:

* ``Instance`` -- mask-backed, what ``overlay_from_masks`` produces from a
  segmentation (the lab notebooks); every instance of a frame shares one
  full-frame label mask, which a LUT remap can exploit;
* ``Contour`` -- polygon-backed, what ``load_segmentation`` returns; each cell
  has to be rasterised from its outline.

The ``Contour`` flavour is the one the user hits after ``load_segmentation``,
and the one the naive implementation punishes hardest: it rasterises every
polygon over the whole frame.
"""

from __future__ import annotations

import argparse
import time
import warnings

# reuse the scene generators so this benchmark measures the same synthetic
# movie as bench_properties.py -- the numbers are then directly comparable
from bench_properties import build_contour_overlay, build_masks

from acia.segm.formats import overlay_from_masks
from acia.tracking.processor.utils import overlay_to_masks


def _time(fn):
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def run(frames: int, cells: int, size: int) -> None:
    total_cells = frames * cells
    masks = build_masks(frames, cells, size)

    print(
        f"\n{frames} frames x {cells} cells = {total_cells} detections, {size}x{size}"
    )
    print("=" * 74)

    for flavour, make_overlay in (
        ("Instance (mask-backed)", lambda: overlay_from_masks(masks)),
        (
            "Contour  (polygon-backed, load_segmentation)",
            lambda: build_contour_overlay(frames, cells, size),
        ),
    ):
        print(f"\n{flavour}")

        build_time, overlay = _time(make_overlay)
        rasterise_time, stack = _time(
            lambda ov=overlay: overlay_to_masks(ov, height=size, width=size)
        )

        print(f"  build overlay        {build_time:8.3f} s")
        print(
            f"  overlay_to_masks     {rasterise_time:8.3f} s   "
            f"{rasterise_time / total_cells * 1e3:8.3f} ms/cell"
            f"      (stack {stack.shape} {stack.dtype})"
        )
        per_roi = rasterise_time / total_cells * 150_000
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
