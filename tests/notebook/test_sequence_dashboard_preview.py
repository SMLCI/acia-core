"""The SequenceDashboard gallery's hover preview.

Rows keep a fixed height (hovering must never reflow the list), and the one
shared preview popover is placed beside the gallery -- never over it. The
placement is a pure JS function (``placePreview``), checked here with node when
it is available; jsdom has no layout, so the event wiring itself still needs a
real Jupyter session.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

pytest.importorskip("anywidget")

from acia.notebook import (  # noqa: E402
    _SEQUENCE_DASHBOARD_CSS_TEXT as CSS,
)
from acia.notebook import (  # noqa: E402
    _SEQUENCE_DASHBOARD_ESM_TEXT as ESM,
)


def test_rows_have_a_fixed_height():
    thumb = re.search(r"\.acia-sd \.sd-thumb\{[^}]*\}", CSS).group(0)
    assert "height:44px" in thumb
    assert "transition" not in thumb
    hover = re.search(r"\.acia-sd \.sd-thumb:hover\{([^}]*)\}", CSS).group(1)
    assert "height" not in hover
    assert "--exp" not in CSS and "computeExp" not in ESM


def test_one_popover_that_ignores_the_pointer():
    pop = re.search(r"\.acia-sd \.sd-pop\{([^}]*)\}", CSS).group(1)
    assert "position:absolute" in pop and "pointer-events:none" in pop
    assert ESM.count('pop.className = "sd-pop"') == 1
    assert '"pointerenter", () => showPreview(d)' in ESM


def _place(cases):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    fn = re.search(r"function placePreview\(.*?\n\}\n", ESM, re.S).group(0)
    script = (
        fn
        + f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(c => placePreview(...c))));"
    )
    out = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def _rect(left, top, right, bottom):
    return {"left": left, "top": top, "right": right, "bottom": bottom}


GAL = _rect(0, 50, 190, 650)  # gallery pane
PANE = _rect(195, 50, 900, 650)  # editor (canvas) pane


def _overlaps(a, b):
    return (
        a["left"] < b["right"]
        and b["left"] < a["right"]
        and a["top"] < b["bottom"]
        and b["top"] < a["bottom"]
    )


@pytest.mark.parametrize("aspect", [0.25, 1.0, 3.0])
def test_preview_never_covers_the_gallery_and_stays_in_the_pane(aspect):
    rows = [_rect(8, y, 182, y + 44) for y in range(50, 650, 50)]
    placed = _place([[GAL, row, PANE, aspect, 420] for row in rows])
    for p in placed:
        box = _rect(p["left"], p["top"], p["left"] + p["width"], p["top"] + p["height"])
        assert not _overlaps(box, GAL)
        assert box["left"] >= PANE["left"] and box["right"] <= PANE["right"]
        assert box["top"] >= PANE["top"] and box["bottom"] <= PANE["bottom"]
        assert p["width"] <= 420
        assert p["height"] == pytest.approx(p["width"] * aspect, abs=1)


def test_preview_is_centred_on_the_row_when_there_is_room():
    row = _rect(8, 300, 182, 344)
    (p,) = _place([[GAL, row, PANE, 0.5, 400]])
    assert p["top"] + p["height"] / 2 == pytest.approx(322, abs=1)
    assert p["left"] == GAL["right"] + 8


def test_no_room_gives_an_empty_box():
    narrow = _rect(195, 50, 200, 650)  # editor pane squeezed to nothing
    (p,) = _place([[GAL, _rect(8, 60, 182, 104), narrow, 1.0, 420]])
    assert p["width"] == 0  # showPreview hides it below 40 px
