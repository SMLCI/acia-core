"""Headless-browser verification of the ``ROICropper`` ESM JavaScript.

The widget's interactive layer (``acia.notebook._ROI_CROPPER_ESM``) is plain
JavaScript run by anywidget in the browser, so the Python unit tests in
``test_roi_cropper.py`` cannot exercise it. This module loads the *real* ESM in
a headless Chromium (Playwright), wires it to a minimal mock anywidget ``model``,
drives it with synthetic ``PointerEvent``\\s, and asserts the behaviours that the
deferred-work note said "needs a real run to confirm":

* rotate knob does NOT teleport the angle when grabbed at rest, and turns the
  box in the expected direction (left -> +angle, right -> -angle, CCW);
* corner resize anchors the diagonally OPPOSITE corner;
* a click adds a point everywhere -- including *inside* the box body -- and a
  sub-threshold jitter on press is treated as a click, not a drag;
* coordinates map to the right canvas pixel even when the host stretches the
  canvas with CSS (Fix C);
* a re-render does not stack duplicate canvases, and the returned disposer
  tears the widget down (Fix A).

Requires the ``widget`` extra (anywidget, for the ESM source) and a Playwright
Chromium browser (baked into the devcontainer image). Both are skipped cleanly
when absent so the suite still runs on a bare machine.

NOTE: the click -> ``cv2.minAreaRect`` *fit* runs in Python (the ``points``
observer), not in JS, so it is intentionally NOT covered here -- it is unit
tested in ``test_roi_cropper.py``. This module covers the pure-JS interactions.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("anywidget")
sync_api = pytest.importorskip("playwright.sync_api")

from acia.notebook import _ROI_CROPPER_ESM  # noqa: E402

# --- the in-page harness -----------------------------------------------------
# Sets up the ESM against a mock model and exposes helpers on ``window.__h``.
# The mock model mirrors anywidget's backbone-style API closely enough for the
# ESM: get/set/save_changes plus on/off where a combined "change:a change:b"
# subscription fires whenever any named trait is set (so the ESM's redraw runs).
_HARNESS = r"""
async ([esm, traits]) => {
  const blob = new Blob([esm], { type: "text/javascript" });
  const url = URL.createObjectURL(blob);
  const mod = await import(url);

  const data = JSON.parse(JSON.stringify(traits));
  const subs = [];                       // {events: string, cb}
  const model = {
    get: (k) => data[k],
    set: (k, v) => {
      data[k] = v;
      for (const s of subs) {
        if (s.events.split(/\s+/).includes("change:" + k)) s.cb();
      }
    },
    save_changes: () => {},
    on: (events, cb) => { subs.push({ events, cb }); },
    off: (events, cb) => {
      for (let i = subs.length - 1; i >= 0; i--) {
        if (subs[i].events === events && subs[i].cb === cb) subs.splice(i, 1);
      }
    },
  };

  const el = document.getElementById("widget");
  const cleanup = mod.default.render({ model, el });

  function canvas() { return el.querySelector("canvas"); }
  function clientFromCanvas(cx, cy) {
    const c = canvas();
    const rect = c.getBoundingClientRect();
    const sx = rect.width / c.width;   // CSS px per backing px
    const sy = rect.height / c.height;
    return [rect.left + cx * sx, rect.top + cy * sy];
  }
  function fire(type, cx, cy) {
    const [clientX, clientY] = clientFromCanvas(cx, cy);
    const ev = new PointerEvent(type, {
      clientX, clientY, pointerId: 1, isPrimary: true,
      bubbles: true, cancelable: true,
    });
    canvas().dispatchEvent(ev);
  }

  window.__h = {
    data, model, cleanup,
    subCount: () => subs.length,
    canvasCount: () => el.querySelectorAll("canvas").length,
    canvasGone: () => el.querySelector("canvas") === null,
    fire,
    // a full press-drag-release gesture through N intermediate move points
    gesture: (path) => {
      fire("pointerdown", path[0][0], path[0][1]);
      for (let i = 1; i < path.length; i++) fire("pointermove", path[i][0], path[i][1]);
      const last = path[path.length - 1];
      fire("pointerup", last[0], last[1]);
    },
    click: (x, y) => {            // down+up at the same spot (no move)
      fire("pointerdown", x, y);
      fire("pointerup", x, y);
    },
  };
  return true;
}
"""

# Image is 200x200 so the ESM's display ``scale`` is exactly 1 (MAX_W=640):
# canvas pixels == image pixels, which keeps the expected geometry simple.
_IMG = 200
# A 1x1 transparent PNG; the geometry never depends on the image content.
_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _traits(**over):
    t = {
        "center_x": 100.0,
        "center_y": 100.0,
        "width": 80,
        "height": 60,
        "angle": 0.0,
        "points": [],
        "image_b64": _PNG,
        "image_w": _IMG,
        "image_h": _IMG,
    }
    t.update(over)
    return t


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import Error as PWError

    try:
        with sync_api.sync_playwright() as p:
            try:
                b = p.chromium.launch()
            except PWError as exc:  # browser binary not installed
                pytest.skip(f"Chromium not available for Playwright: {exc}")
            yield b
            b.close()
    except PWError as exc:  # pragma: no cover - environment guard
        pytest.skip(f"Playwright could not start: {exc}")


@pytest.fixture
def page(browser):
    page = browser.new_page(viewport={"width": 900, "height": 700})
    page.set_content('<!doctype html><html><body><div id="widget"></div></body></html>')
    yield page
    page.close()


def _setup(page, **over):
    """Render the ESM with the given trait overrides; return the live trait dict."""
    page.evaluate(_HARNESS, [_ROI_CROPPER_ESM, _traits(**over)])
    return page.evaluate("() => window.__h.data")


# --- render / teardown -------------------------------------------------------


def test_render_creates_single_canvas(page):
    _setup(page)
    assert page.evaluate("() => window.__h.canvasCount()") == 1


def test_rerender_does_not_stack_canvases(page):
    """Fix A: render() wipes ``el`` first, so a re-render keeps exactly one canvas."""
    _setup(page)
    # render a second time onto the same el (as marimo would on a re-run)
    page.evaluate(_HARNESS, [_ROI_CROPPER_ESM, _traits()])
    assert page.evaluate("() => window.__h.canvasCount()") == 1


def test_disposer_removes_widget(page):
    """Fix A: the returned cleanup() detaches the node."""
    _setup(page)
    page.evaluate("() => window.__h.cleanup()")
    assert page.evaluate("() => window.__h.canvasGone()") is True


# --- clicks add points (incl. inside the box) --------------------------------


def test_click_outside_box_adds_point(page):
    _setup(page)
    page.evaluate("() => window.__h.click(10, 10)")
    pts = page.evaluate("() => window.__h.data.points")
    assert len(pts) == 1
    assert pts[0][0] == pytest.approx(10, abs=1)
    assert pts[0][1] == pytest.approx(10, abs=1)


def test_click_inside_box_body_adds_point(page):
    """Fix D: a click on the box body (here, the center) still adds a point."""
    _setup(page)
    page.evaluate("() => window.__h.click(100, 100)")  # dead center of the box
    pts = page.evaluate("() => window.__h.data.points")
    assert len(pts) == 1
    assert pts[0][0] == pytest.approx(100, abs=1)
    assert pts[0][1] == pytest.approx(100, abs=1)


def test_subthreshold_jitter_is_a_click_not_a_drag(page):
    """Fix D: a tiny move (<4px) on press is a click (adds a point), not a move."""
    _setup(page)
    # press inside the box, jitter 2px, release -> still a click; center unchanged
    page.evaluate("() => window.__h.gesture([[100,100],[102,101],[101,100]])")
    data = page.evaluate("() => window.__h.data")
    assert len(data["points"]) == 1
    assert data["center_x"] == pytest.approx(100, abs=1e-6)
    assert data["center_y"] == pytest.approx(100, abs=1e-6)


# --- move drag ---------------------------------------------------------------


def test_drag_body_moves_center(page):
    _setup(page)
    # press at center, drag +30/+20 px past the threshold, release
    page.evaluate("() => window.__h.gesture([[100,100],[130,120]])")
    data = page.evaluate("() => window.__h.data")
    assert data["center_x"] == pytest.approx(130, abs=1)
    assert data["center_y"] == pytest.approx(120, abs=1)
    assert data["points"] == []  # a drag must NOT add a point


# --- resize anchors the opposite corner --------------------------------------


def _corner(cx, cy, w, h, angle_deg, i):
    """Replicate the ESM ``corners()`` (CCW, screen-y-down) for corner ``i``."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    hw, hh = w / 2, h / 2
    local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)][i]
    lx, ly = local
    return (cx + lx * ca + ly * sa, cy - lx * sa + ly * ca)


def test_resize_anchors_opposite_corner(page):
    _setup(page)
    # corner 0 starts at (60,70); opposite corner 2 is (140,130).
    opp_before = _corner(100, 100, 80, 60, 0, 2)
    assert opp_before == pytest.approx((140, 130))
    # grab corner 0, drag it to (50,60)
    page.evaluate("() => window.__h.gesture([[60,70],[50,60]])")
    data = page.evaluate("() => window.__h.data")
    # opposite corner (index 2) must be unchanged
    opp_after = _corner(
        data["center_x"],
        data["center_y"],
        data["width"],
        data["height"],
        data["angle"],
        2,
    )
    assert opp_after[0] == pytest.approx(140, abs=1)
    assert opp_after[1] == pytest.approx(130, abs=1)
    # and the box grew to span pointer<->opposite (90 x 70, centered at 95,95)
    assert data["width"] == pytest.approx(90, abs=1)
    assert data["height"] == pytest.approx(70, abs=1)
    assert data["center_x"] == pytest.approx(95, abs=1)
    assert data["center_y"] == pytest.approx(95, abs=1)


# --- rotate: no teleport + correct direction ---------------------------------


def _rotate_handle(cx, cy, w, h, angle_deg, scale=1.0):
    """Replicate the ESM ``rotateHandle()`` image-px position."""
    off = h / 2 + 24 / scale
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # localToImg(r, 0, -off)
    return (cx + (-off) * sa, cy + (-off) * ca)


def test_rotate_no_teleport_at_rest(page):
    """Fix B: grabbing the knob and sliding it radially keeps the angle put."""
    _setup(page, angle=30.0)
    hx, hy = _rotate_handle(100, 100, 80, 60, 30.0)
    # move the pointer further out along the SAME ray from the center (pure
    # radial move): direction is unchanged, so the angle must stay ~30.
    far_x = 100 + 2 * (hx - 100)
    far_y = 100 + 2 * (hy - 100)
    page.evaluate(f"() => window.__h.gesture([[{hx},{hy}],[{far_x},{far_y}]])")
    angle = page.evaluate("() => window.__h.data.angle")
    assert angle == pytest.approx(30.0, abs=0.5)


def test_rotate_direction_left_is_positive(page):
    """Fix B: from the rest knob (box at 0deg), moving the pointer left turns the
    box CCW (+angle); moving right turns it CW (-angle)."""
    # knob rest for angle 0 is straight up from center: (100, 100-54).
    hx, hy = _rotate_handle(100, 100, 80, 60, 0.0)
    assert (hx, hy) == pytest.approx((100, 46))

    _setup(page, angle=0.0)
    page.evaluate(f"() => window.__h.gesture([[{hx},{hy}],[{hx - 10},{hy}]])")
    left_angle = page.evaluate("() => window.__h.data.angle")
    assert left_angle > 0.5

    _setup(page, angle=0.0)
    page.evaluate(f"() => window.__h.gesture([[{hx},{hy}],[{hx + 10},{hy}]])")
    right_angle = page.evaluate("() => window.__h.data.angle")
    assert right_angle < -0.5


# --- CSS scaling (Fix C) -----------------------------------------------------


def test_clicks_map_correctly_under_css_stretch(page):
    """Fix C: stretch the canvas 2x via CSS; a click still lands on the intended
    canvas pixel because localPos() rescales by canvas.width/rect.width."""
    _setup(page)
    # stretch the backing-store-200px canvas to 400 CSS px wide, 100 tall.
    page.evaluate(
        "() => { const c = window.__h.canvasCount && document.querySelector('#widget canvas');"
        " c.style.width = '400px'; c.style.height = '100px'; }"
    )
    # click at canvas pixel (10,10); clientFromCanvas accounts for the CSS size,
    # and the ESM inverts it the same way -> the stored point must be ~(10,10).
    page.evaluate("() => window.__h.click(10, 10)")
    pts = page.evaluate("() => window.__h.data.points")
    assert len(pts) == 1
    assert pts[0][0] == pytest.approx(10, abs=1.5)
    assert pts[0][1] == pytest.approx(10, abs=1.5)
