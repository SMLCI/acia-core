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

else:

    def ROICropper(*args, **kwargs):  # type: ignore[no-redef]
        """Stub raised when the optional ``widget`` extra is not installed."""
        raise ImportError(
            "ROICropper requires the optional dependency: pip install acia[widget]"
        )
