"""Jupyter notebook visualization mixin for image sequence sources."""

from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from acia.base import BaseImage


def normalize_to_uint8(image_array: np.ndarray) -> np.ndarray:
    """Normalize an image array to the uint8 ``[0, 255]`` range.

    A ``uint8`` array is passed through unchanged; any other dtype is min-max
    scaled to ``[0, 255]``. A flat array (``max == min``) maps to all zeros.

    Args:
        image_array: Source image array of any dtype/shape.

    Returns:
        np.ndarray: A ``uint8`` array of the same shape.
    """
    if image_array.dtype == np.uint8:
        return image_array

    # Normalize to 0-255 range
    min_val = np.min(image_array)
    max_val = np.max(image_array)

    normalized: np.ndarray
    if max_val > min_val:
        normalized = ((image_array - min_val) / (max_val - min_val) * 255).astype(
            np.uint8
        )
    else:
        normalized = np.zeros_like(image_array, dtype=np.uint8)

    return normalized


class JupyterVisualizationMixin:
    """Mixin providing interactive Jupyter notebook visualization for image sequences.

    This mixin expects the host class to implement the ImageSequenceSource
    interface with these properties/methods:
    - size_t: int - number of time frames
    - num_channels: int - number of channels
    - get_frame(frame: int) -> BaseImage

    Example usage:
        class MyImageSource(ImageSequenceSource, JupyterVisualizationMixin):
            ...
    """

    if TYPE_CHECKING:
        # `size_t`/`num_channels` are supplied by the host ImageSequenceSource.
        # They are declared here for the type checker only (guarded by
        # TYPE_CHECKING so no runtime descriptor is created) and as read-only
        # properties so concrete sources may override them with `@property`.
        @property
        def size_t(self) -> int: ...
        @property
        def num_channels(self) -> int: ...

    def get_frame(self, frame: int) -> BaseImage:
        """Get frame at given index."""
        raise NotImplementedError()

    def _repr_html_(self) -> str | None:
        """Jupyter notebook rich display integration with interactive ipywidgets viewer.

        Returns an interactive viewer with:
        - Time slider to navigate through frames (T dimension)
        - Channel toggle controls (C dimension)
        - Real-time image display updates

        Returns:
            str: HTML representation for Jupyter display, or None for non-Jupyter environments
        """
        try:
            # Try to import ipywidgets and IPython display
            import ipywidgets as widgets
            from IPython.display import HTML, display

            # Check if we're in a Jupyter environment
            try:
                get_ipython()  # type: ignore[name-defined]  # noqa: F821
            except NameError:
                # Not in Jupyter/IPython environment
                return None

        except ImportError:
            # ipywidgets not available
            logging.warning(
                "ipywidgets not installed. Install with: pip install ipywidgets>=8.0.0"
            )
            return None

        # Get dimensions
        try:
            num_frames = self.size_t
            num_channels = self.num_channels
        except (NotImplementedError, AttributeError):
            # If properties not implemented, try to get from __len__
            num_frames = len(self) if hasattr(self, "__len__") else 1  # type: ignore[arg-type]
            num_channels = 1

        # Create output widget for displaying images
        output = widgets.Output()

        # Create time slider (only if more than 1 frame)
        if num_frames > 1:
            time_slider = widgets.IntSlider(
                value=0,
                min=0,
                max=num_frames - 1,
                step=1,
                description="Frame:",
                continuous_update=False,  # Update only when slider is released
                layout=widgets.Layout(width="80%"),
            )
        else:
            time_slider = None

        # Create channel toggle buttons (only if more than 1 channel)
        if num_channels > 1:
            channel_toggles = [
                widgets.Checkbox(
                    value=True,
                    description=f"Channel {i}",
                    indent=False,
                )
                for i in range(num_channels)
            ]
        else:
            channel_toggles = []

        # Create overlay controls (only if overlay is provided)
        overlay = getattr(self, "overlay", None)
        if overlay is not None:
            overlay_checkbox = widgets.Checkbox(
                value=True,
                description="Overlay",
                indent=False,
            )
            opacity_slider = widgets.FloatSlider(
                value=0.8,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Opacity:",
                continuous_update=False,
                layout=widgets.Layout(width="80%"),
            )
        else:
            overlay_checkbox = None
            opacity_slider = None

        def render_image(frame_idx: int, active_channels: list[bool]) -> None:
            """Render image for given frame and active channels."""
            with output:
                output.clear_output(wait=True)

                try:
                    # Get the frame
                    frame = self.get_frame(frame_idx)
                    image_data = frame.raw

                    # Handle channel selection
                    if num_channels > 1 and len(active_channels) > 0:
                        # Get active channel indices
                        active_indices = [
                            i for i, active in enumerate(active_channels) if active
                        ]

                        if len(active_indices) == 0:
                            # No channels selected, show blank image
                            if len(image_data.shape) == 3:
                                height, width = image_data.shape[:2]
                            else:
                                height, width = image_data.shape
                            image_data = np.zeros((height, width, 3), dtype=np.uint8)
                        elif len(active_indices) == 1:
                            # Single channel - display as grayscale
                            channel_data = frame.get_channel(active_indices[0])
                            channel_data = normalize_to_uint8(channel_data)
                            # Convert to RGB by repeating channel
                            if len(channel_data.shape) == 2:
                                image_data = np.repeat(
                                    channel_data[:, :, np.newaxis], 3, axis=-1
                                )
                            else:
                                image_data = channel_data
                        else:
                            # Multiple channels - combine them
                            # For now, we'll overlay them as RGB if 3 channels,
                            # otherwise grayscale blend
                            combined = None
                            for idx in active_indices[
                                :3
                            ]:  # Take at most first 3 channels for RGB
                                channel_data = frame.get_channel(idx)
                                channel_data = normalize_to_uint8(channel_data)
                                if combined is None:
                                    if len(channel_data.shape) == 2:
                                        combined = np.zeros(
                                            (*channel_data.shape, 3), dtype=np.uint8
                                        )
                                    else:
                                        combined = np.zeros_like(
                                            channel_data, dtype=np.uint8
                                        )

                                if len(channel_data.shape) == 2:
                                    # Grayscale channel
                                    channel_idx = active_indices.index(idx) % 3
                                    combined[:, :, channel_idx] = channel_data
                                else:
                                    # RGB channel
                                    combined = np.maximum(combined, channel_data)

                            image_data = combined
                    else:
                        # Single channel or no channel selection
                        image_data = normalize_to_uint8(image_data)

                    # Ensure image is in correct format for PIL
                    if len(image_data.shape) == 2:
                        # Grayscale - convert to RGB
                        image_data = np.repeat(image_data[:, :, np.newaxis], 3, axis=-1)

                    # Convert to PIL Image
                    pil_image = Image.fromarray(image_data)

                    # Convert to base64-encoded PNG for display
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format="PNG")
                    buffer.seek(0)
                    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

                    # Display as HTML image
                    html = f'<img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto;" />'
                    display(HTML(html))

                except Exception as e:
                    logging.error(f"Error rendering image: {e}")
                    display(
                        HTML(f"<p style='color: red;'>Error displaying frame: {e}</p>")
                    )

        def on_update(*args) -> None:
            """Callback for widget updates."""
            frame_idx = time_slider.value if time_slider else 0
            active_channels = (
                [toggle.value for toggle in channel_toggles]
                if channel_toggles
                else [True]
            )
            render_image(frame_idx, active_channels)

        # Connect widgets to callback
        if time_slider:
            time_slider.observe(on_update, names="value")

        for toggle in channel_toggles:
            toggle.observe(on_update, names="value")

        # Build layout
        controls = []
        if time_slider:
            controls.append(time_slider)
        if overlay_checkbox is not None and opacity_slider is not None:
            controls.append(
                widgets.HBox(
                    [overlay_checkbox, opacity_slider],
                    layout=widgets.Layout(flex_flow="row wrap"),
                )
            )
        if channel_toggles:
            controls.append(
                widgets.HBox(
                    channel_toggles,
                    layout=widgets.Layout(flex_flow="row wrap"),
                )
            )

        if controls:
            viewer = widgets.VBox([*controls, output])
        else:
            viewer = widgets.VBox([output])

        # Render initial image
        on_update()

        # Display the widget
        display(viewer)

        # Return empty string to satisfy _repr_html_ protocol
        return ""


def _encode_frame_png(source, frame: int = 0, channel: int | None = None):
    """Encode one frame of ``source`` as a PNG data URL for an anywidget.

    Selects a single 2D display channel (channel ``0`` by default for a
    multi-channel frame), min-max normalizes it to ``uint8``, promotes
    grayscale to RGB, and returns a base64 PNG ``data:`` URL plus the frame
    ``(width, height)`` in pixels. Shared by the ``ROICropper`` /
    ``FilterExplorer`` widgets so frames travel to the browser as bytes.

    Args:
        source: An :class:`~acia.base.ImageSequenceSource` (uses ``get_frame``).
        frame: Frame index to encode.
        channel: Display channel for a multi-channel frame; ``None`` -> channel 0.

    Returns:
        tuple[str, int, int]: ``(data_url, width, height)``.

    Raises:
        ValueError: if ``channel`` is out of range for the frame.
    """
    raw = np.asarray(source.get_frame(frame).raw)

    num_channels = raw.shape[-1] if raw.ndim == 3 else 1
    if channel is not None and not (0 <= channel < num_channels):
        raise ValueError(f"channel must be in [0, {num_channels}); got {channel}.")

    if raw.ndim == 3:
        if raw.shape[-1] == 1:
            display = raw[..., 0]
        else:
            display = raw[..., 0 if channel is None else channel]
    else:
        display = raw

    display = normalize_to_uint8(display)
    if display.ndim == 2:
        display = np.repeat(display[:, :, np.newaxis], 3, axis=-1)

    frame_h, frame_w = int(raw.shape[0]), int(raw.shape[1])

    pil_image = Image.fromarray(display)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}", frame_w, frame_h


# ---------------------------------------------------------------------------
# ROICropper -- optional anywidget for drawing a rotated-rectangle ROI on
# frame 0. anywidget is an OPTIONAL dependency (pip install acia[widget]);
# acia.notebook MUST stay importable without it (it is imported by base.py /
# local.py / nd2_source.py). The anywidget subclass is therefore defined only
# when the import succeeds; otherwise ``ROICropper`` is bound to a stub that
# raises a clear ImportError on instantiation.
# ---------------------------------------------------------------------------

try:
    import anywidget
    import traitlets

    _HAS_ANYWIDGET = True
except ImportError:  # pragma: no cover - exercised only when extra is absent
    _HAS_ANYWIDGET = False


# The ESM render() is BEST-EFFORT and CANNOT be exercised by the headless Python
# test-suite. It is verified only by a real run in Jupyter/Colab/marimo. It is
# self-contained (no external imports). Keep it in sync with the synced traits.
#
# BEST-EFFORT / UNVERIFIED-HEADLESS NOTICE
# ----------------------------------------
# Everything below this notice is JavaScript executed in the browser by anywidget
# and is *not* covered by the Python test-suite. The fixes here come from a
# careful code review; the interactive feel still needs a real Jupyter/Colab/
# marimo run to confirm. A real run must still verify, by hand:
#   * the rotate knob turns the box in the direction the pointer moves (CCW feel)
#     and does not "teleport" the angle when first grabbed at its rest position;
#   * corner resize anchors the OPPOSITE corner (drag a corner; the diagonally
#     opposite corner should stay put);
#   * click-to-add-point works everywhere, including *inside* the box body, and
#     a small accidental movement on press does not get mis-read as a drag;
#   * the box lands on the right image pixels even when the notebook host
#     stretches the canvas with CSS.
_ROI_CROPPER_ESM = r"""
// ROICropper render() -- canvas draw + click-to-add-points + drag/resize/rotate.
// UNVERIFIED in CI: validated by a real Jupyter/Colab/marimo run only.
// render() returns a cleanup function (anywidget calls it on teardown; marimo
// re-renders frequently, so we must NOT stack duplicate canvases/listeners).
function render({ model, el }) {
  // Fix A: wipe any previous render output so re-renders don't stack canvases.
  el.innerHTML = "";

  const wrap = document.createElement("div");
  wrap.style.position = "relative";
  wrap.style.display = "inline-block";
  const canvas = document.createElement("canvas");
  canvas.style.touchAction = "none";
  wrap.appendChild(canvas);

  const hint = document.createElement("div");
  hint.style.font = "12px sans-serif";
  hint.style.marginTop = "4px";
  hint.textContent =
    "Click ≥3 points to fit a box (clicks inside the box add points too), " +
    "or drag the box / corners / rotate knob.";
  wrap.appendChild(hint);
  el.appendChild(wrap);

  const img = new Image();
  let imgReady = false;
  img.onload = () => { imgReady = true; layout(); draw(); };
  // Fix F: surface a broken/empty data URL instead of silently showing nothing.
  img.onerror = () => { imgReady = false; draw(); };

  // Display scale: fit within a max width while tracking image<->canvas px.
  const MAX_W = 640;
  let scale = 1;
  const MOVE_THRESHOLD = 4; // px in canvas space; below this a press is a CLICK

  function imgW() { return model.get("image_w") || img.naturalWidth || 1; }
  function imgH() { return model.get("image_h") || img.naturalHeight || 1; }

  function layout() {
    const w = imgW(), h = imgH();
    scale = Math.min(1, MAX_W / w);
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
  }

  // canvas px <-> image px
  function toImg(px, py) { return [px / scale, py / scale]; }
  function toCanvas(ix, iy) { return [ix * scale, iy * scale]; }

  function getRect() {
    return {
      cx: model.get("center_x"),
      cy: model.get("center_y"),
      w: model.get("width"),
      h: model.get("height"),
      angle: model.get("angle"),
    };
  }

  // Corner offsets in the rect's local frame (image px). angle is CCW degrees
  // (OpenCV getRotationMatrix2D convention). Screen y is down, so a positive
  // CCW angle rotates with -sin in the y component to match the Python warp.
  // Rotation matrix R (local -> image), used everywhere for consistency:
  //   ix = cx + lx*ca + ly*sa
  //   iy = cy - lx*sa + ly*ca
  function corners(r) {
    const a = (r.angle * Math.PI) / 180;
    const ca = Math.cos(a), sa = Math.sin(a);
    const hw = r.w / 2, hh = r.h / 2;
    const local = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
    return local.map(([lx, ly]) => [
      r.cx + lx * ca + ly * sa,
      r.cy - lx * sa + ly * ca,
    ]);
  }

  // Forward map local (lx, ly) -> image (ix, iy) using the SAME matrix.
  function localToImg(r, lx, ly) {
    const a = (r.angle * Math.PI) / 180;
    const ca = Math.cos(a), sa = Math.sin(a);
    return [r.cx + lx * ca + ly * sa, r.cy - lx * sa + ly * ca];
  }
  // Inverse map image (ix, iy) -> local (lx, ly) (transpose of R).
  function imgToLocal(r, ix, iy) {
    const a = (r.angle * Math.PI) / 180;
    const ca = Math.cos(a), sa = Math.sin(a);
    const dx = ix - r.cx, dy = iy - r.cy;
    // local = R^T * d ; from corners(): lx = dx*ca - dy*sa, ly = dx*sa + dy*ca
    return [dx * ca - dy * sa, dx * sa + dy * ca];
  }

  // Knob sits along the box's local -y axis (i.e. local (0, -off)).
  function rotateHandle(r) {
    const off = r.h / 2 + 24 / scale;
    return localToImg(r, 0, -off);
  }

  function draw() {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (imgReady) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // clicked points
    const pts = model.get("points") || [];
    ctx.fillStyle = "#00e5ff";
    for (const [ix, iy] of pts) {
      const [px, py] = toCanvas(ix, iy);
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    // rect
    const r = getRect();
    const cs = corners(r).map(([ix, iy]) => toCanvas(ix, iy));
    ctx.strokeStyle = "#ffeb3b";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cs[0][0], cs[0][1]);
    for (let i = 1; i < cs.length; i++) ctx.lineTo(cs[i][0], cs[i][1]);
    ctx.closePath();
    ctx.stroke();

    // corner handles
    ctx.fillStyle = "#ffeb3b";
    for (const [px, py] of cs) {
      ctx.fillRect(px - 4, py - 4, 8, 8);
    }

    // rotate knob
    const [rx, ry] = toCanvas(...rotateHandle(r));
    const [ccx, ccy] = toCanvas(r.cx, r.cy);
    ctx.strokeStyle = "#ff5252";
    ctx.beginPath();
    ctx.moveTo(ccx, ccy);
    ctx.lineTo(rx, ry);
    ctx.stroke();
    ctx.fillStyle = "#ff5252";
    ctx.beginPath();
    ctx.arc(rx, ry, 5, 0, 2 * Math.PI);
    ctx.fill();
  }

  let drag = null; // {mode, cornerIndex, r, ix, iy, startPx, startPy, moved}

  function hit(px, py) {
    const r = getRect();
    const cs = corners(r).map(([ix, iy]) => toCanvas(ix, iy));
    for (let i = 0; i < cs.length; i++) {
      if (Math.hypot(px - cs[i][0], py - cs[i][1]) <= 8) {
        return { mode: "resize", cornerIndex: i };
      }
    }
    const [rx, ry] = toCanvas(...rotateHandle(r));
    if (Math.hypot(px - rx, py - ry) <= 8) return { mode: "rotate" };
    // inside body?
    const [ix, iy] = toImg(px, py);
    const [lx, ly] = imgToLocal(r, ix, iy);
    if (Math.abs(lx) <= r.w / 2 && Math.abs(ly) <= r.h / 2) {
      return { mode: "move" };
    }
    return null;
  }

  // Fix C: convert a pointer event to canvas pixels robustly under CSS scaling.
  // The host may stretch the canvas via CSS, so getBoundingClientRect() can
  // differ from the canvas backing-store size. Scale client coords by
  // canvas.width/rect.width (and height) so clicks land on the right pixel.
  function localPos(ev) {
    const rect = canvas.getBoundingClientRect();
    const sx = rect.width ? canvas.width / rect.width : 1;
    const sy = rect.height ? canvas.height / rect.height : 1;
    return [(ev.clientX - rect.left) * sx, (ev.clientY - rect.top) * sy];
  }

  function addPoint(px, py) {
    const [ix, iy] = toImg(px, py);
    const pts = (model.get("points") || []).slice();
    pts.push([ix, iy]);
    model.set("points", pts);
    model.save_changes();
    draw();
  }

  function onPointerDown(ev) {
    const [px, py] = localPos(ev);
    const h = hit(px, py);
    const r = getRect();
    const [ix, iy] = toImg(px, py);
    // Fix D: record the press; do NOT enter a drag mode yet. A press becomes a
    // drag only once the pointer moves past MOVE_THRESHOLD; otherwise pointerup
    // treats it as a CLICK and appends a point -- even over the box body.
    drag = {
      ...(h || {}),
      candidateMode: h ? h.mode : null,
      mode: null, // activated on first significant move
      r,
      ix,
      iy,
      startPx: px,
      startPy: py,
      moved: false,
    };
    // Fix F: setPointerCapture can throw on some hosts; never let it break drag.
    try { canvas.setPointerCapture(ev.pointerId); } catch (e) {}
  }

  function onPointerMove(ev) {
    if (!drag) return;
    const [px, py] = localPos(ev);
    if (!drag.moved) {
      if (Math.hypot(px - drag.startPx, py - drag.startPy) < MOVE_THRESHOLD) {
        return; // still within click tolerance
      }
      drag.moved = true;
      // Only now commit to a manipulation mode (if the press hit a handle/body).
      drag.mode = drag.candidateMode;
    }
    if (!drag.mode) return; // moved on empty space -> ignore (no point yet)

    const [ix, iy] = toImg(px, py);
    const r = drag.r;
    if (drag.mode === "move") {
      model.set("center_x", r.cx + (ix - drag.ix));
      model.set("center_y", r.cy + (iy - drag.iy));
    } else if (drag.mode === "rotate") {
      // Fix B: derive the angle from the SAME rotation matrix corners() uses.
      // The knob's rest position is local (0, -off), which maps to image offset
      //   (dx, dy) = (-off*sa, -off*ca)   [from localToImg with lx=0, ly=-off]
      // We want angle(pointer) such that at rest it equals r.angle. With
      //   angle = atan2(-(ix-cx), -(iy-cy)) * 180/PI
      // at rest: atan2(-(-off*sa), -(-off*ca)) = atan2(off*sa, off*ca) = a (rad)
      // => angle == r.angle exactly (no teleport on grab). A small CCW pointer
      // move increases the angle, matching the Python CCW / warp convention.
      const angle = Math.atan2(-(ix - r.cx), -(iy - r.cy)) * 180 / Math.PI;
      model.set("angle", angle);
    } else if (drag.mode === "resize") {
      // Fix E: anchor the OPPOSITE corner (standard UX). Using the drag-start
      // snapshot r, compute the fixed opposite corner in image px, then derive
      // the new center as the midpoint of (pointer, opposite) and the new
      // width/height from the pointer-vs-opposite delta projected onto the
      // box's rotated local axes.
      const hw = r.w / 2, hh = r.h / 2;
      const localCorners = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
      const i = drag.cornerIndex;
      const opp = localCorners[(i + 2) % 4]; // diagonally opposite corner
      const [ox, oy] = localToImg(r, opp[0], opp[1]); // fixed anchor (image px)

      // new center = midpoint of dragged pointer and the fixed opposite corner.
      const ncx = (ix + ox) / 2;
      const ncy = (iy + oy) / 2;

      // project (pointer - opposite) onto the box's (unchanged) local axes to
      // get the new full width/height.
      const rAxes = { ...r, cx: ncx, cy: ncy };
      const [lx, ly] = imgToLocal(rAxes, ix, iy);
      const oLocal = imgToLocal(rAxes, ox, oy); // == -[lx, ly] by construction
      const nw = Math.max(1, Math.round(Math.abs(lx - oLocal[0])));
      const nh = Math.max(1, Math.round(Math.abs(ly - oLocal[1])));

      model.set("center_x", ncx);
      model.set("center_y", ncy);
      model.set("width", nw);
      model.set("height", nh);
    }
    model.save_changes();
    draw();
  }

  function onPointerEnd(ev) {
    if (!drag) return;
    // Fix D: a press that never crossed the threshold is a click -> add a point
    // (works inside the box body too, since we deferred entering "move").
    if (!drag.moved) {
      addPoint(drag.startPx, drag.startPy);
    }
    try { canvas.releasePointerCapture(ev.pointerId); } catch (e) {}
    drag = null;
  }

  // Fix A: named handlers + a disposer. anywidget calls the returned function
  // on teardown; we remove listeners, model observers, and the appended node.
  function onImageChange() {
    const b64 = model.get("image_b64");
    if (b64) { img.src = b64; } // Fix F: guard empty/None src
  }
  function onGeomChange() { draw(); }

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerEnd);
  canvas.addEventListener("pointercancel", onPointerEnd);

  model.on("change:image_b64", onImageChange);
  model.on(
    "change:center_x change:center_y change:width change:height change:angle change:points",
    onGeomChange,
  );

  layout();
  const initB64 = model.get("image_b64");
  if (initB64) { img.src = initB64; } // Fix F: guard empty/None src
  draw();

  // Disposer: undo everything this render() set up.
  return () => {
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerEnd);
    canvas.removeEventListener("pointercancel", onPointerEnd);
    model.off("change:image_b64", onImageChange);
    model.off(
      "change:center_x change:center_y change:width change:height change:angle change:points",
      onGeomChange,
    );
    if (wrap.parentNode) { wrap.parentNode.removeChild(wrap); }
  };
}
export default { render };
"""


# The FilterExplorer ESM is BEST-EFFORT JavaScript, exercised by the headless
# Playwright suite (tests/notebook/test_filter_explorer_esm_playwright.py) but
# not by the pure-Python tests. Live mask filtering happens entirely client-side
# from the precomputed per-contour values, so NO kernel round-trip is needed as
# the sliders move (the spec's "reactive, no observer wiring"). render() returns
# a disposer so marimo re-renders don't stack canvases/listeners.
_FILTER_EXPLORER_ESM = r"""
// FilterExplorer render() -- one (min,max) slider row per filter; live overlay
// recolouring kept=green / dropped=red as the handles move. Client-side only.
function render({ model, el }) {
  el.innerHTML = ""; // wipe prior render output (no stacked canvases on re-run)

  const wrap = document.createElement("div");
  wrap.style.font = "12px sans-serif";

  const canvas = document.createElement("canvas");
  canvas.style.display = "block";
  wrap.appendChild(canvas);

  const count = document.createElement("div");
  count.style.margin = "4px 0";
  wrap.appendChild(count);

  const controls = document.createElement("div");
  wrap.appendChild(controls);
  el.appendChild(wrap);

  const MAX_W = 480;
  let scale = 1;
  function imgW() { return model.get("image_w") || 1; }
  function imgH() { return model.get("image_h") || 1; }
  function layout() {
    const w = imgW(), h = imgH();
    scale = Math.min(1, MAX_W / w);
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
  }

  const img = new Image();
  let imgReady = false;
  img.onload = () => { imgReady = true; draw(); };
  img.onerror = () => { imgReady = false; draw(); };

  // local working copy of the handle values; written back to the model on input.
  let sel = JSON.parse(JSON.stringify(model.get("selection") || []));

  function fmt(x) {
    const a = Math.abs(x);
    if (a >= 100) return x.toFixed(0);
    if (a >= 1) return x.toFixed(2);
    return x.toFixed(3);
  }

  // a contour is kept iff every filter's value is within its [vmin, vmax].
  // The positive form (rather than `v < vmin || v > vmax`) also drops a
  // non-finite value, matching Python's `accepts` (>=/<= are false for NaN).
  function keep(rec) {
    const v = rec.values || [];
    for (let i = 0; i < sel.length; i++) {
      if (!(v[i] >= sel[i].vmin && v[i] <= sel[i].vmax)) return false;
    }
    return true;
  }

  function draw() {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (imgReady) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const conts = model.get("contours") || [];
    let kept = 0;
    for (const rec of conts) {
      const pts = rec.points || [];
      if (pts.length < 2) continue;
      const k = keep(rec);
      if (k) kept++;
      ctx.beginPath();
      ctx.moveTo(pts[0][0] * scale, pts[0][1] * scale);
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i][0] * scale, pts[i][1] * scale);
      }
      ctx.closePath();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = k ? "#2ecc40" : "#ff4136";
      ctx.fillStyle = k ? "rgba(46,204,64,0.25)" : "rgba(255,65,54,0.12)";
      ctx.fill();
      ctx.stroke();
    }
    count.textContent = "kept " + kept + " / " + conts.length;
  }

  function commit() {
    model.set("selection", JSON.parse(JSON.stringify(sel)));
    model.save_changes();
    draw();
  }

  // build one control row per filter spec.
  const specs = model.get("filter_specs") || [];
  const rowHandlers = [];
  specs.forEach((spec, i) => {
    const row = document.createElement("div");
    row.style.margin = "6px 0";

    const label = document.createElement("div");
    const unit = spec.unit ? " [" + spec.unit + "]" : "";
    label.textContent = spec.name + unit;
    label.style.fontWeight = "bold";
    row.appendChild(label);

    const mkSlider = () => {
      const s = document.createElement("input");
      s.type = "range";
      s.min = String(spec.lo);
      s.max = String(spec.hi);
      s.step = String(spec.step || (spec.hi - spec.lo) / 200 || 1);
      s.style.width = "240px";
      return s;
    };
    const lo = mkSlider(); lo.value = String(sel[i].vmin);
    const hi = mkSlider(); hi.value = String(sel[i].vmax);

    const readout = document.createElement("span");
    readout.style.marginLeft = "8px";

    function refresh() {
      let a = parseFloat(lo.value), b = parseFloat(hi.value);
      if (a > b) {                         // keep min <= max
        if (this === lo) { b = a; hi.value = String(b); }
        else { a = b; lo.value = String(a); }
      }
      sel[i] = { vmin: a, vmax: b };
      readout.textContent = "[" + fmt(a) + ", " + fmt(b) + "]";
      commit();
    }
    lo.addEventListener("input", refresh);
    hi.addEventListener("input", refresh);
    rowHandlers.push([lo, hi, refresh]);

    readout.textContent = "[" + fmt(sel[i].vmin) + ", " + fmt(sel[i].vmax) + "]";

    row.appendChild(document.createElement("br"));
    row.appendChild(lo);
    row.appendChild(hi);
    row.appendChild(readout);
    controls.appendChild(row);
  });

  function onSelectionChange() {
    sel = JSON.parse(JSON.stringify(model.get("selection") || []));
    specs.forEach((spec, i) => {
      const [lo, hi] = rowHandlers[i];
      // Only rewrite a handle whose value actually changed, so a slider the
      // user is dragging (already equal to sel[i]) is not snapped/reset by our
      // own committed change:selection echo.
      if (parseFloat(lo.value) !== sel[i].vmin) lo.value = String(sel[i].vmin);
      if (parseFloat(hi.value) !== sel[i].vmax) hi.value = String(sel[i].vmax);
    });
    draw();
  }
  function onDataChange() { draw(); }
  model.on("change:selection", onSelectionChange);
  model.on("change:contours change:filter_specs", onDataChange);

  layout();
  const b64 = model.get("image_b64");
  if (b64) { img.src = b64; }
  draw();

  return () => {
    for (const [lo, hi, refresh] of rowHandlers) {
      lo.removeEventListener("input", refresh);
      hi.removeEventListener("input", refresh);
    }
    model.off("change:selection", onSelectionChange);
    model.off("change:contours change:filter_specs", onDataChange);
    if (wrap.parentNode) { wrap.parentNode.removeChild(wrap); }
  };
}
export default { render };
"""


if _HAS_ANYWIDGET:

    class ROICropper(anywidget.AnyWidget):  # type: ignore[no-redef]
        """Interactive rotated-rectangle ROI selector over frame 0 (anywidget).

        Draw/drag/resize/rotate a rectangle over the first frame of an
        :class:`~acia.base.ImageSequenceSource` and emit a
        :class:`~acia.base.RotatedCropSpec`. Two ways to set the box, both
        feeding the same synced traits:

        1. Click >=3 points around the ROI (the ``points`` trait); an observer
           runs :meth:`fit_to_points` (``cv2.minAreaRect``) to seed the tightest
           oriented rectangle. The geometry lives in Python, so it is unit-tested.
        2. Drag the box / corner handles / rotate knob in the ESM ``render()``.

        ROI coordinates are kept in **parent image pixels** so :attr:`spec`
        plugs straight into :meth:`~acia.base.ImageSequenceSource.crop_rotated`.

        Works in Jupyter/Colab and in marimo via ``mo.ui.anywidget(cropper)``
        (it IS-A ipywidgets ``DOMWidget``). The ESM JavaScript is best-effort and
        is verified only by a real notebook run, not by the headless test-suite.
        """

        center_x = traitlets.Float(0.0).tag(sync=True)
        center_y = traitlets.Float(0.0).tag(sync=True)
        width = traitlets.Int(1).tag(sync=True)
        height = traitlets.Int(1).tag(sync=True)
        angle = traitlets.Float(0.0).tag(sync=True)
        points = traitlets.List().tag(sync=True)  # type: ignore[var-annotated]  # [[x, y], ...] image px
        image_b64 = traitlets.Unicode("").tag(sync=True)
        image_w = traitlets.Int(0).tag(sync=True)
        image_h = traitlets.Int(0).tag(sync=True)

        _esm = _ROI_CROPPER_ESM

        def __init__(
            self,
            source,
            *,
            width: int | None = None,
            height: int | None = None,
            channel: int | None = None,
            **kwargs,
        ) -> None:
            """Build the widget from a source's frame 0.

            Args:
                source: The :class:`~acia.base.ImageSequenceSource` to crop.
                width: Default ROI width (px). Defaults to ``frame_w // 2``.
                height: Default ROI height (px). Defaults to ``frame_h // 2``.
                channel: Display channel for a multi-channel frame. Defaults to
                    channel ``0``.
                **kwargs: Forwarded to ``anywidget.AnyWidget``.
            """
            self._source = source

            raw = np.asarray(source.get_frame(0).raw)

            # Validate an explicitly requested display channel against the frame.
            num_channels = raw.shape[-1] if raw.ndim == 3 else 1
            if channel is not None and not (0 <= channel < num_channels):
                raise ValueError(
                    f"channel must be in [0, {num_channels}); got {channel}."
                )
            self._channel = channel

            # Select a 2D display channel; grayscale stays grayscale.
            if raw.ndim == 3:
                if raw.shape[-1] == 1:
                    display = raw[..., 0]
                else:
                    display = raw[..., 0 if self._channel is None else self._channel]
            else:
                display = raw

            display = normalize_to_uint8(display)

            # Grayscale -> RGB for display.
            if display.ndim == 2:
                display = np.repeat(display[:, :, np.newaxis], 3, axis=-1)

            frame_h, frame_w = int(raw.shape[0]), int(raw.shape[1])

            pil_image = Image.fromarray(display)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            img_b64 = base64.b64encode(buffer.read()).decode("utf-8")

            default_w = int(width) if width is not None else max(1, frame_w // 2)
            default_h = int(height) if height is not None else max(1, frame_h // 2)

            super().__init__(
                center_x=frame_w / 2.0,
                center_y=frame_h / 2.0,
                width=default_w,
                height=default_h,
                angle=0.0,
                image_b64=f"data:image/png;base64,{img_b64}",
                image_w=frame_w,
                image_h=frame_h,
                **kwargs,
            )

        def fit_to_points(self, points=None):
            """Fit the tightest oriented rectangle to ``points`` and set traits.

            Uses ``cv2.minAreaRect`` on the given (or the ``points`` trait's)
            ``[x, y]`` image-px points, then normalizes the angle into
            ``(-45, 45]`` degrees (CCW, OpenCV ``getRotationMatrix2D`` /
            :class:`~acia.base.RotatedCropSpec` convention), swapping width and
            height with each 90-degree step so the round-trip
            ``fit_to_points -> crop_rotated`` straightens the region. Sizes are
            rounded to positive ints.

            Args:
                points: ``[[x, y], ...]`` image-px points. Defaults to the
                    current ``points`` trait when ``None``.

            Raises:
                ValueError: If fewer than 3 points are supplied, or if the
                    points are collinear/duplicate (degenerate rectangle).
            """
            import cv2

            pts = self.points if points is None else points
            pts_arr = np.asarray(pts, dtype=np.float32)
            if pts_arr.ndim != 2 or pts_arr.shape[0] < 3 or pts_arr.shape[1] != 2:
                raise ValueError(
                    "fit_to_points requires at least 3 [x, y] points; "
                    f"got {pts_arr.shape}."
                )

            (cx, cy), (w, h), angle = cv2.minAreaRect(pts_arr)
            # Collinear / duplicate points yield a degenerate rect (zero extent).
            # Reject BEFORE the max(1, ...) clamp so callers see a clear error
            # rather than a silently-clamped 1px box.
            if w == 0 or h == 0:
                raise ValueError("degenerate rectangle (collinear/duplicate points)")
            w = int(round(w))
            h = int(round(h))

            # Normalize the OpenCV angle into (-45, 45] degrees. The rectangle has
            # 90-degree symmetry, so each 90-degree step swaps width/height. This
            # keeps fit_to_points -> crop_rotated a faithful straightening and
            # maps an axis-aligned box to angle == 0.
            while angle > 45.0:
                angle -= 90.0
                w, h = h, w
            while angle <= -45.0:
                angle += 90.0
                w, h = h, w

            self.center_x = float(cx)
            self.center_y = float(cy)
            self.width = max(1, w)
            self.height = max(1, h)
            self.angle = float(angle)

        @traitlets.observe("points")
        def _on_points(self, change) -> None:
            """Re-fit the box whenever >=3 points are present.

            Degenerate / too-few-point states during interactive clicking are
            ignored (the box is only seeded once enough points arrive).
            """
            import cv2

            pts = change.get("new") if isinstance(change, dict) else change.new
            try:
                if pts is not None and len(pts) >= 3:
                    self.fit_to_points(pts)
            except (ValueError, cv2.error):
                # Only swallow the expected degenerate/too-few-point states
                # (raised by fit_to_points / cv2.minAreaRect). Real bugs such as
                # an ImportError must NOT be silently dropped.
                pass

        @property
        def spec(self):
            """Return the current ROI as a :class:`~acia.base.RotatedCropSpec`."""
            from acia.base import RotatedCropSpec

            return RotatedCropSpec(
                center=(self.center_x, self.center_y),
                size=(int(self.width), int(self.height)),
                angle=self.angle,
            )

        def cropped(self):
            """Return ``source.crop_rotated(self.spec)`` (a lazy crop source)."""
            return self._source.crop_rotated(self.spec)

        def save(self, dataset_dir, **kwargs):
            """Persist the crop via :func:`~acia.crop_capture.save_crop_capture`.

            Args:
                dataset_dir: Directory the capture is written to.
                **kwargs: Forwarded to ``save_crop_capture`` (``frame``,
                    ``channel``, ``clip_percentiles``, ``source_ref``).

            Returns:
                dict: The ``save_crop_capture`` result.
            """
            from acia.crop_capture import save_crop_capture

            return save_crop_capture(self._source, self.spec, dataset_dir, **kwargs)

        def _repr_html_(self) -> str:
            """Static fallback so a non-executed/persisted notebook shows frame 0."""
            return (
                f'<div><img src="{self.image_b64}" '
                'style="max-width: 100%; height: auto;" />'
                "<p style='font:12px sans-serif;color:#666;'>"
                "ROICropper (interactive widget renders when the notebook is run)."
                "</p></div>"
            )

    class FilterExplorer(anywidget.AnyWidget):  # type: ignore[no-redef]
        """Interactive cell-filter threshold explorer with live mask preview.

        Auto-builds **one (min, max) slider per filter** from the passed
        ``filters`` list (modular -- add a :class:`~acia.segm.filter.CellFilter`
        and it appears) and live-recolours the contour overlay (kept = green,
        dropped = red) as the handles move. The live filtering runs **entirely
        client-side**: each contour's value under each filter is precomputed once
        in Python (reusing the goal-E ``value()`` calibration) and shipped to the
        browser, so dragging a slider needs **no kernel round-trip** (the
        workflow's "reactive, no observer wiring"). All control ranges/handles are
        in each filter's **physical unit** (µm, µm², dimensionless).

        The widget previews a **single frame** (``frame=0`` by default), drawing
        only that frame's contours over that frame's image. :attr:`params` and
        :meth:`configured_filters` emit/restore frame-independent thresholds;
        :meth:`filtered_overlay` applies them across the **whole** overlay via
        :func:`~acia.segm.filter.apply_cell_filters`.

        Works in Jupyter/Colab and in marimo via ``mo.ui.anywidget(explorer)``.
        The ESM JavaScript is best-effort, verified by the headless Playwright
        suite (and a real notebook run), not by the pure-Python tests.
        """

        image_b64 = traitlets.Unicode("").tag(sync=True)
        image_w = traitlets.Int(0).tag(sync=True)
        image_h = traitlets.Int(0).tag(sync=True)
        # one spec per filter: {name, unit, lo, hi, step, vmin, vmax}
        filter_specs = traitlets.List().tag(sync=True)  # type: ignore[var-annotated]
        # one record per displayed contour: {points: [[x,y],...], values: [m0,m1,...]}
        contours = traitlets.List().tag(sync=True)  # type: ignore[var-annotated]
        # live handle values, aligned with filter_specs: [{vmin, vmax}, ...]
        selection = traitlets.List().tag(sync=True)  # type: ignore[var-annotated]

        _esm = _FILTER_EXPLORER_ESM

        def __init__(
            self,
            overlay,
            images,
            filters,
            *,
            frame: int = 0,
            channel: int | None = None,
            **kwargs,
        ) -> None:
            """Build the explorer from an overlay, a calibrated source and filters.

            Args:
                overlay: The :class:`~acia.base.Overlay` to filter.
                images: The calibrated :class:`~acia.base.ImageSequenceSource`
                    (must expose a non-``None`` ``pixel_size``).
                filters: A list of :class:`~acia.segm.filter.CellFilter` instances
                    -- one slider control is built per filter.
                frame: Frame to preview (image + its contours). Defaults to ``0``.
                channel: Display channel for a multi-channel frame. Defaults to 0.
                **kwargs: Forwarded to ``anywidget.AnyWidget``.

            Raises:
                ValueError: If ``images`` is ``None`` or its ``pixel_size`` is
                    ``None`` (physical-unit thresholds need calibration).
            """
            if images is None or getattr(images, "pixel_size", None) is None:
                raise ValueError(
                    "FilterExplorer requires a calibrated source (pixel_size); "
                    "physical-unit filtering cannot run on uncalibrated data."
                )

            self._overlay = overlay
            self._images = images
            self._filters = list(filters)
            self._frame = frame

            img_b64, frame_w, frame_h = _encode_frame_png(images, frame, channel)

            # contours shown for this frame (default frame attr -> include it)
            conts = [c for c in overlay.contours if getattr(c, "frame", frame) == frame]

            records = [
                {
                    "points": np.asarray(c.coordinates, dtype=float).tolist(),
                    "values": [],
                }
                for c in conts
            ]

            specs = []
            for f in self._filters:
                unit, mags = self._measure(f, conts)
                for rec, m in zip(records, mags, strict=False):
                    rec["values"].append(m)
                lo, hi, vmin, vmax = self._axis(f, unit, mags)
                step = (hi - lo) / 200.0 or 1.0
                specs.append(
                    {
                        "name": f.name,
                        "unit": unit,
                        "lo": lo,
                        "hi": hi,
                        "step": step,
                        "vmin": vmin,
                        "vmax": vmax,
                    }
                )

            selection = [{"vmin": s["vmin"], "vmax": s["vmax"]} for s in specs]

            super().__init__(
                image_b64=img_b64,
                image_w=frame_w,
                image_h=frame_h,
                filter_specs=specs,
                contours=records,
                selection=selection,
                **kwargs,
            )

        # --- construction helpers -------------------------------------------

        def _measure(self, f, conts):
            """Return ``(unit_str, [magnitude per contour])`` for filter ``f``.

            Reuses the goal-E ``CellFilter.value()`` (calibrated from the source
            ``pixel_size``); all contours of a filter share one unit. For an empty
            overlay the unit is inferred from the filter's existing bound, else
            falls back to dimensionless.
            """
            import math

            unit = None
            mags = []
            for c in conts:
                q = f.value(c, images=self._images)
                if unit is None:
                    unit = f"{q.units}"
                m = float(q.to(unit).magnitude)
                # a non-finite magnitude would serialize as invalid JSON (NaN /
                # Infinity) and break the browser trait sync; coerce to 0, the
                # same convention the built-in filters use for degenerate cells.
                mags.append(m if math.isfinite(m) else 0.0)
            if unit is None:
                unit = self._unit_of(f.vmin) or self._unit_of(f.vmax) or ""
            return unit, mags

        @staticmethod
        def _unit_of(bound) -> str | None:
            """Unit string of a pint bound, or ``None`` for plain numbers/``None``."""
            if bound is not None and hasattr(bound, "units"):
                return f"{bound.units}"
            return None

        def _axis(self, f, unit: str, mags):
            """Build a control axis: ``(lo, hi, vmin, vmax)`` for filter ``f``.

            The track ``[lo, hi]`` spans the data, then is **widened to include
            any explicitly-set bound** so seeding is lossless: a bound outside the
            data range is preserved exactly instead of being clamped to the data
            extreme (which would silently rewrite the threshold). A ``None`` bound
            opens that side (handle parked at the track extreme: ``vmin`` at ``lo``,
            ``vmax`` at ``hi``). An empty overlay falls back to a ``[0, 1]`` track.
            """
            raw_vmin = self._bound_magnitude(f.vmin, unit)
            raw_vmax = self._bound_magnitude(f.vmax, unit)

            bounds = [b for b in (raw_vmin, raw_vmax) if b is not None]
            if mags:
                lo, hi = float(min(mags)), float(max(mags))
            elif bounds:
                lo, hi = float(min(bounds)), float(max(bounds))
            else:
                lo, hi = 0.0, 1.0
            # widen so an out-of-range seed sits inside the track (lossless).
            for b in bounds:
                lo, hi = min(lo, b), max(hi, b)
            if hi <= lo:  # single-valued data / single bound -> non-zero width
                hi = lo + 1.0

            vmin = lo if raw_vmin is None else raw_vmin
            vmax = hi if raw_vmax is None else raw_vmax
            return lo, hi, vmin, vmax

        @staticmethod
        def _bound_magnitude(bound, unit: str):
            """Magnitude of ``bound`` in ``unit`` (``None`` stays ``None``)."""
            if bound is None:
                return None
            if hasattr(bound, "to") and hasattr(bound, "magnitude"):
                target = unit if unit else "dimensionless"
                return float(bound.to(target).magnitude)
            return float(bound)

        # --- outputs ---------------------------------------------------------

        @property
        def params(self):
            """Current thresholds as ``[{name, vmin, vmax}]`` pint ``Quantity``\\s.

            A handle parked at its track extreme is reported as ``None`` (open on
            that side), so a one-sided filter round-trips faithfully.
            """
            result = []
            for spec, sel in zip(self.filter_specs, self.selection, strict=False):
                unit = spec["unit"]
                vmin = self._as_quantity(sel["vmin"], spec["lo"], unit, lower=True)
                vmax = self._as_quantity(sel["vmax"], spec["hi"], unit, lower=False)
                result.append({"name": spec["name"], "vmin": vmin, "vmax": vmax})
            return result

        @staticmethod
        def _as_quantity(value: float, extreme: float, unit: str, *, lower: bool):
            """``Q_(value, unit)`` unless ``value`` is at the open extreme."""
            from acia import Q_

            if (lower and value <= extreme) or (not lower and value >= extreme):
                return None
            return Q_(value, unit) if unit else Q_(value)

        def configured_filters(self):
            """Update each passed filter's ``vmin``/``vmax`` from the sliders.

            Mutates the filter instances supplied to ``__init__`` in place (a
            handle at its extreme sets that bound to ``None``) and returns the
            list, ready for :func:`~acia.segm.filter.apply_cell_filters` or the
            scaled batch run.
            """
            for f, spec, sel in zip(
                self._filters, self.filter_specs, self.selection, strict=False
            ):
                unit = spec["unit"]
                f.vmin = self._as_quantity(sel["vmin"], spec["lo"], unit, lower=True)
                f.vmax = self._as_quantity(sel["vmax"], spec["hi"], unit, lower=False)
            return self._filters

        def filtered_overlay(self):
            """Return the whole overlay filtered by the current thresholds."""
            from acia.segm.filter import apply_cell_filters

            return apply_cell_filters(
                self._overlay, self.configured_filters(), images=self._images
            )

        def save(self, path):
            """Write the current thresholds to ``path`` as ``filter_params.json``.

            Serializes ``[{name, unit, vmin, vmax}]`` (magnitudes; ``None`` for an
            open side) -- a small spec the scaled batch run reloads to rebuild the
            filters.

            Args:
                path: Destination JSON file path.

            Returns:
                list[dict]: The serialized filter parameters.
            """
            import json
            from pathlib import Path

            data = []
            for spec, sel in zip(self.filter_specs, self.selection, strict=False):
                lo, hi = spec["lo"], spec["hi"]
                data.append(
                    {
                        "name": spec["name"],
                        "unit": spec["unit"],
                        "vmin": None if sel["vmin"] <= lo else sel["vmin"],
                        "vmax": None if sel["vmax"] >= hi else sel["vmax"],
                    }
                )
            Path(path).write_text(json.dumps({"filters": data}, indent=2))
            return data

        def _repr_html_(self) -> str:
            """Static fallback so a non-executed/persisted notebook shows frame 0."""
            return (
                f'<div><img src="{self.image_b64}" '
                'style="max-width: 100%; height: auto;" />'
                "<p style='font:12px sans-serif;color:#666;'>"
                "FilterExplorer (interactive widget renders when the notebook is run)."
                "</p></div>"
            )

else:

    def ROICropper(*args, **kwargs):  # type: ignore[no-redef]
        """Stub raised when the optional ``widget`` extra is not installed."""
        raise ImportError(
            "ROICropper requires the optional dependency: pip install acia[widget]"
        )

    def FilterExplorer(*args, **kwargs):  # type: ignore[no-redef]
        """Stub raised when the optional ``widget`` extra is not installed."""
        raise ImportError(
            "FilterExplorer requires the optional dependency: pip install acia[widget]"
        )
