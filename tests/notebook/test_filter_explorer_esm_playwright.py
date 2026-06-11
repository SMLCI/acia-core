"""Headless-browser verification of the ``FilterExplorer`` ESM JavaScript.

Loads the real ``acia.notebook._FILTER_EXPLORER_ESM`` in a headless Chromium
(Playwright) against a mock anywidget ``model`` and drives the slider controls,
asserting the client-side live-filter behaviour that the Python tests cannot
reach:

* one (min, max) control row is built per filter spec;
* dragging a handle recomputes the kept set from the precomputed per-contour
  values and writes the ``selection`` trait -- with NO kernel round-trip;
* a handle parked at its track extreme keeps that side open;
* a re-render leaves a single canvas, and the disposer tears the widget down.

Skips cleanly without anywidget or a Playwright Chromium browser.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anywidget")
sync_api = pytest.importorskip("playwright.sync_api")

from acia.notebook import _FILTER_EXPLORER_ESM  # noqa: E402

# In-page harness: mounts the ESM against a mock model and exposes helpers on
# window.__h. The mock model fires a combined "change:a change:b" subscription
# whenever any named trait is set (mirrors anywidget closely enough for redraw).
_HARNESS = r"""
async ([esm, traits]) => {
  const blob = new Blob([esm], { type: "text/javascript" });
  const mod = await import(URL.createObjectURL(blob));

  const data = JSON.parse(JSON.stringify(traits));
  const subs = [];
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

  function sliders() { return Array.from(el.querySelectorAll('input[type=range]')); }

  window.__h = {
    data, model, cleanup,
    canvasCount: () => el.querySelectorAll("canvas").length,
    canvasGone: () => el.querySelector("canvas") === null,
    sliderCount: () => sliders().length,
    countText: () => el.querySelector("div div") ? el.querySelectorAll("div")[1].textContent : "",
    // set range input number `idx` to `value` and fire an 'input' event.
    setSlider: (idx, value) => {
      const s = sliders()[idx];
      s.value = String(value);
      s.dispatchEvent(new Event("input", { bubbles: true }));
    },
  };
  return true;
}
"""

_IMG = 200
_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _traits():
    """Two filters (area µm², circularity), three contours with known values."""
    specs = [
        {
            "name": "area",
            "unit": "micrometer ** 2",
            "lo": 4.0,
            "hi": 36.0,
            "step": 0.16,
            "vmin": 4.0,
            "vmax": 36.0,
        },
        {
            "name": "circularity",
            "unit": "dimensionless",
            "lo": 0.5,
            "hi": 1.0,
            "step": 0.0025,
            "vmin": 0.5,
            "vmax": 1.0,
        },
    ]
    contours = [
        {"points": [[10, 10], [20, 10], [20, 20], [10, 20]], "values": [4.0, 0.78]},
        {"points": [[30, 30], [50, 30], [50, 50], [30, 50]], "values": [16.0, 0.78]},
        {"points": [[60, 60], [90, 60], [90, 90], [60, 90]], "values": [36.0, 0.78]},
    ]
    selection = [{"vmin": s["vmin"], "vmax": s["vmax"]} for s in specs]
    return {
        "image_b64": _PNG,
        "image_w": _IMG,
        "image_h": _IMG,
        "filter_specs": specs,
        "contours": contours,
        "selection": selection,
    }


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import Error as PWError

    try:
        with sync_api.sync_playwright() as p:
            try:
                b = p.chromium.launch()
            except PWError as exc:
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


def _setup(page):
    page.evaluate(_HARNESS, [_FILTER_EXPLORER_ESM, _traits()])


# --- structure ---------------------------------------------------------------


def test_builds_two_sliders_per_filter(page):
    _setup(page)
    # one (min, max) pair per filter -> 2 filters * 2 = 4 range inputs
    assert page.evaluate("() => window.__h.sliderCount()") == 4
    assert page.evaluate("() => window.__h.canvasCount()") == 1


def test_initial_count_keeps_all(page):
    _setup(page)
    assert "kept 3 / 3" in page.evaluate("() => window.__h.countText()")


# --- live filtering ----------------------------------------------------------


def test_dragging_area_max_drops_big_contour(page):
    """Slider 1 is the area-MAX handle; lowering it to 20 drops the 36 µm² cell."""
    _setup(page)
    page.evaluate("() => window.__h.setSlider(1, 20)")
    sel = page.evaluate("() => window.__h.data.selection")
    assert sel[0]["vmax"] == pytest.approx(20.0, abs=0.2)
    assert "kept 2 / 3" in page.evaluate("() => window.__h.countText()")


def test_dragging_area_min_drops_small_contour(page):
    """Slider 0 is the area-MIN handle; raising it to 10 drops the 4 µm² cell."""
    _setup(page)
    page.evaluate("() => window.__h.setSlider(0, 10)")
    sel = page.evaluate("() => window.__h.data.selection")
    assert sel[0]["vmin"] == pytest.approx(10.0, abs=0.2)
    assert "kept 2 / 3" in page.evaluate("() => window.__h.countText()")


def test_nan_value_is_dropped_not_kept(page):
    """keep() drops a non-finite value (matches Python's >=/<=), not keeps it."""
    _setup(page)
    # inject a NaN area into the first contour, then nudge a slider to redraw.
    page.evaluate(
        "() => { window.__h.data.contours[0].values[0] = NaN; "
        "window.__h.model.set('contours', window.__h.data.contours); }"
    )
    page.evaluate("() => window.__h.setSlider(1, 36)")  # no-op move -> redraw
    assert "kept 2 / 3" in page.evaluate("() => window.__h.countText()")


def test_handles_at_extremes_keep_all(page):
    """With every handle at its track extreme, all contours stay kept."""
    _setup(page)
    page.evaluate("() => window.__h.setSlider(1, 36)")  # area max back to hi
    assert "kept 3 / 3" in page.evaluate("() => window.__h.countText()")


def test_min_does_not_exceed_max(page):
    """Pushing the MIN handle past the MAX clamps them equal (min <= max)."""
    _setup(page)
    page.evaluate("() => window.__h.setSlider(1, 16)")  # area max -> 16
    page.evaluate("() => window.__h.setSlider(0, 30)")  # area min -> 30 (> max)
    sel = page.evaluate("() => window.__h.data.selection")
    assert sel[0]["vmin"] <= sel[0]["vmax"] + 1e-6


# --- teardown ----------------------------------------------------------------


def test_rerender_does_not_stack_canvases(page):
    _setup(page)
    page.evaluate(_HARNESS, [_FILTER_EXPLORER_ESM, _traits()])
    assert page.evaluate("() => window.__h.canvasCount()") == 1


def test_disposer_removes_widget(page):
    _setup(page)
    page.evaluate("() => window.__h.cleanup()")
    assert page.evaluate("() => window.__h.canvasGone()") is True
