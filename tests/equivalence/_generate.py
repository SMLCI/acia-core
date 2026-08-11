"""Regenerate ``golden.npz`` -- the pre-optimisation reference snapshot.

Run it with the library in its **pre-optimisation** state to (re)create the
reference, then never again while a speed change is in flight: the whole point
is that the committed numbers predate the optimisation.

    python -m tests.equivalence._generate

Regenerating after a change would silently bless whatever that change did, so
the test asserts the file's provenance stamp and any regeneration should be its
own reviewed commit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapely

from acia.analysis import ExtractorExecutor
from acia.segm.filter import apply_cell_filters

from .scenes import GOLDEN_SCENES, extractors, filter_battery

GOLDEN_PATH = Path(__file__).parent / "golden.npz"


def snapshot() -> dict:
    """Build the full reference snapshot from the current implementation."""
    payload: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}

    for scene_name, build in GOLDEN_SCENES.items():
        overlay, images = build()

        executor = ExtractorExecutor()
        df = executor.execute(overlay, images, extractors())

        ids = [str(i) for i in df.index]
        payload[f"{scene_name}/ids"] = np.array(ids, dtype=object)
        for column in df.columns:
            payload[f"{scene_name}/col/{column}"] = df[column].to_numpy(
                dtype=np.float64
            )

        # every contour's polygon, as WKB -- exact, and comparable with .equals()
        payload[f"{scene_name}/wkb"] = np.array(
            [
                shapely.to_wkb(c.polygon) if c.polygon is not None else b""
                for c in overlay
            ],
            dtype=object,
        )
        payload[f"{scene_name}/wkb_ids"] = np.array(
            [str(c.id) for c in overlay], dtype=object
        )

        kept: dict[str, list[str]] = {}
        for battery_name, filters in filter_battery().items():
            result = apply_cell_filters(overlay, filters, images=images)
            kept[battery_name] = sorted(str(c.id) for c in result)

        meta[scene_name] = {
            "columns": list(df.columns),
            "units": {k: str(v) for k, v in df.attrs.get("units", {}).items()},
            "kept": kept,
            "n_contours": len(overlay.contours),
        }

    payload["__meta__"] = np.array([json.dumps(meta, sort_keys=True)], dtype=object)
    return payload


def main() -> None:
    payload = snapshot()
    np.savez_compressed(GOLDEN_PATH, **payload)
    meta = json.loads(payload["__meta__"][0])
    print(f"wrote {GOLDEN_PATH}")
    for scene_name, scene_meta in meta.items():
        print(
            f"  {scene_name:<32} {scene_meta['n_contours']:>4} contours, "
            f"{len(scene_meta['columns'])} columns, "
            f"{len(scene_meta['kept'])} filter configs"
        )


if __name__ == "__main__":
    main()
