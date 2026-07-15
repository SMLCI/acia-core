"""Jupyter notebook visualization mixin for image sequence sources."""

from __future__ import annotations

import base64
import dataclasses
import io
import logging
import os
import time
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable

    from acia.base import BaseImage, RotatedCropSpec
    from acia.registration import FrameTransform
    from acia.registration_persistence import RegistrationManifest, RegistrationRecord


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
                    if image_data.ndim == 3 and image_data.shape[-1] == 1:
                        # a raw single-channel (H, W, 1) frame -- drop the axis so
                        # the 2D branch below promotes it to RGB; PIL's fromarray
                        # has no mode for a trailing size-1 channel axis
                        image_data = image_data[..., 0]
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


def _fit_rotated_rect(points):
    """Fit the tightest oriented rectangle to ``points`` -> ``RotatedCropSpec``.

    Shared min-area-rectangle geometry (``cv2.minAreaRect``) used by the
    :class:`SequenceDashboard` point-fit path. Angle normalized into ``(-45, 45]``
    degrees (CCW / ``RotatedCropSpec`` convention), width/height swapped per
    90-degree step. ``cv2`` is a core dependency (always available).

    Raises:
        ValueError: If fewer than 3 points, or the points are degenerate.
    """
    import cv2

    from acia.base import RotatedCropSpec

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError(f"fit requires at least 3 [x, y] points; got {pts.shape}.")
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    if w == 0 or h == 0:
        raise ValueError("degenerate rectangle (collinear/duplicate points)")
    w = int(round(w))
    h = int(round(h))
    while angle > 45.0:
        angle -= 90.0
        w, h = h, w
    while angle <= -45.0:
        angle += 90.0
        w, h = h, w
    return RotatedCropSpec(
        center=(float(cx), float(cy)), size=(max(1, w), max(1, h)), angle=float(angle)
    )


# The 5 acia.registration.RegistrationMethod subclass names, in the order they
# appear in the RegistrationDashboard method picker. Kept as plain strings at
# module level (no import) so both the traitlets validator and the ESM's
# <select> options list can use them without touching acia.registration.
_REGISTRATION_METHOD_NAMES: tuple[str, ...] = (
    "PhaseCorrelationHighpass",
    "MaskedTemplateCorrelation",
    "HoughLineRigidFit",
    "FeatureRANSACEuclidean",
    "GradientECC",
)


# batch_apply's checkpoint cadence: the manifest is persisted after every this
# many newly-estimated frames *within* a position (not only after the
# position fully completes), bounding worst-case lost progress on interrupt
# to this many frames without rewriting the whole manifest every single frame
# across a long, multi-position run. See the
# registration-dashboard-progress-video spec's Design Notes.
CHECKPOINT_INTERVAL = 20


def _estimate_eta(
    *,
    elapsed: float,
    frames_done: int,
    frames_left_in_position: int,
    positions_remaining_after: int,
    position_frame_counts: list[int],
    current_position_num_frames: int,
) -> float | None:
    """Best-effort ETA (seconds) for the remainder of a batch-apply run.

    Returns ``None`` when there isn't yet enough signal (no elapsed time or no
    completed frames) to produce a rate -- the ESM only renders an ETA once
    this is non-``None``.

    Heuristic (approximate, not exact -- see the spec's Design Notes):
    ``rate = frames_done / elapsed``; ``remaining = frames_left_in_position +
    positions_remaining_after * average_frames_per_position`` where the
    average is over positions *completed so far in this run*
    (``position_frame_counts``), falling back to the current position's own
    frame count when no position has completed yet; ``eta = remaining / rate``.

    Args:
        elapsed: Seconds elapsed since the batch-apply run started.
        frames_done: Total frames estimated so far across the whole run.
        frames_left_in_position: Frames remaining in the position currently
            being processed.
        positions_remaining_after: Number of positions still to process after
            the current one.
        position_frame_counts: Frame counts of positions already fully
            completed in this run (for averaging).
        current_position_num_frames: Frame count of the position currently
            being processed (fallback average when nothing has completed yet).

    Returns:
        float | None: Estimated remaining seconds, or ``None`` if no rate can
            be computed yet.
    """
    if elapsed <= 0 or frames_done <= 0:
        return None
    rate = frames_done / elapsed
    avg_frames_per_position = (
        sum(position_frame_counts) / len(position_frame_counts)
        if position_frame_counts
        else current_position_num_frames
    )
    remaining = (
        frames_left_in_position + positions_remaining_after * avg_frames_per_position
    )
    return remaining / rate


def _registration_method_classes() -> dict[str, type]:
    """Lazy import of the 5 :class:`~acia.registration.RegistrationMethod` subclasses.

    Deliberately NOT a module-level import: ``acia.registration`` imports from
    ``acia.base``, which itself imports this module (``acia.notebook``) at load
    time for :class:`JupyterVisualizationMixin` -- a module-level import here
    would be circular.
    """
    from acia.registration import (
        FeatureRANSACEuclidean,
        GradientECC,
        HoughLineRigidFit,
        MaskedTemplateCorrelation,
        PhaseCorrelationHighpass,
    )

    return {
        "PhaseCorrelationHighpass": PhaseCorrelationHighpass,
        "MaskedTemplateCorrelation": MaskedTemplateCorrelation,
        "HoughLineRigidFit": HoughLineRigidFit,
        "FeatureRANSACEuclidean": FeatureRANSACEuclidean,
        "GradientECC": GradientECC,
    }


# The SequenceDashboard CSS + ESM are ported near-verbatim from the approved
# clickable mockup (three-pane curation UI). Like the other widgets here, the ESM
# is UNVERIFIED by the headless Python suite and validated only by a real
# Jupyter/marimo run (or the Playwright suite in the devcontainer). Images arrive
# from Python as PNG bytes over anywidget's buffer channel; only single frames and
# the small `selections` list cross the wire.
_SEQUENCE_DASHBOARD_CSS = _SEQUENCE_DASHBOARD_CSS_TEXT = r"""
.acia-sd{--bg:#e9edf1;--panel:#fff;--panel-2:#f4f7f9;--panel-3:#eef2f5;--border:#d5dce2;
  --border-strong:#c2ccd4;--text:#182028;--text-dim:#596a76;--text-faint:#8695a0;
  --accent:#0f8f82;--accent-ink:#fff;--accent-soft:rgba(15,143,130,.12);
  --roi-1:#e08a12;--roi-2:#d24d92;--roi-3:#2b7fd6;--roi-4:#25a06f;--roi-5:#8a6ef0;
  --font-ui:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --font-mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-family:var(--font-ui);color:var(--text);font-size:14px;
  border:1px solid var(--border);border-radius:10px;overflow:hidden;background:var(--bg);
  display:flex;flex-direction:column;height:640px;}
.acia-sd:fullscreen{height:100vh;border-radius:0;}
@media (prefers-color-scheme:dark){.acia-sd{--bg:#0c1216;--panel:#141c22;--panel-2:#1a232a;
  --panel-3:#202b33;--border:#28333c;--border-strong:#34424c;--text:#e7eef3;--text-dim:#8b9aa6;
  --text-faint:#657481;--accent:#2fd4c1;--accent-ink:#062521;--accent-soft:rgba(47,212,193,.14);
  --roi-1:#f0a83a;--roi-2:#ec6fac;--roi-3:#5aa6f0;--roi-4:#3cc78d;--roi-5:#a78bfa;}}
.acia-sd *{box-sizing:border-box}
.acia-sd .sd-src{display:flex;gap:10px;align-items:center;padding:9px 12px;background:var(--panel-2);
  border-bottom:1px solid var(--border);flex-wrap:wrap;font-family:var(--font-mono);font-size:11.5px;}
.acia-sd .sd-src input{flex:1;min-width:160px;background:var(--panel);border:1px solid var(--border);
  color:var(--text);border-radius:7px;padding:6px 9px;font-family:var(--font-mono);font-size:12px;}
.acia-sd .sd-meta{color:var(--text-dim);width:100%;}
.acia-sd .sd-meta b{color:var(--text)}
.acia-sd .sd-main{flex:1;display:grid;grid-template-columns:var(--lw,190px) 5px minmax(0,1fr) 5px var(--rw,250px);
  gap:0;background:var(--border);min-height:0;}
.acia-sd .sd-pane{background:var(--panel);display:flex;flex-direction:column;min-height:0;min-width:0;}
.acia-sd .sd-head{padding:8px 11px;border-bottom:1px solid var(--border);font-family:var(--font-mono);
  font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--text-dim);display:flex;
  justify-content:space-between;align-items:center;gap:8px;}
.acia-sd .sd-rz{background:var(--border);cursor:col-resize;}
.acia-sd .sd-rz:hover{background:var(--accent);}
.acia-sd .sd-gal{overflow-y:auto;padding:8px;display:flex;flex-direction:column;gap:6px;}
.acia-sd .sd-thumb{position:relative;height:44px;flex:none;border-radius:7px;overflow:hidden;cursor:pointer;
  border:1.5px solid transparent;background:var(--panel-3);
  transition:height .3s cubic-bezier(.2,.75,.2,1);}
.acia-sd .sd-thumb:hover{height:var(--exp,300px);}
.acia-sd .sd-thumb.sel{border-color:var(--accent);}
.acia-sd .sd-thumb img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;}
.acia-sd .sd-thumb .ix{position:absolute;left:6px;bottom:4px;font-family:var(--font-mono);font-size:11px;
  font-weight:600;color:#fff;text-shadow:0 1px 3px #000;}
.acia-sd .sd-thumb .dot{position:absolute;right:6px;top:6px;width:8px;height:8px;border-radius:50%;
  background:var(--accent);box-shadow:0 0 0 1.5px rgba(0,0,0,.4);}
.acia-sd .sd-editor{display:flex;flex-direction:column;}
.acia-sd .sd-ebody{flex:1;display:flex;gap:14px;padding:14px;overflow:auto;align-items:flex-start;}
.acia-sd .sd-cwrap{position:relative;border-radius:8px;overflow:hidden;background:#9aa0a4;flex:none;touch-action:none;}
.acia-sd .sd-cwrap.pick{cursor:crosshair;}
.acia-sd .sd-cwrap img{display:block;width:100%;height:100%;}
.acia-sd .sd-cstatus{position:absolute;inset:0;display:none;align-items:center;justify-content:center;
  text-align:center;padding:12px;font-family:var(--font-mono);font-size:12px;color:#fff;
  background:rgba(0,0,0,.35);pointer-events:none;}
.acia-sd .sd-cstatus.show{display:flex;}
.acia-sd .sd-cstatus.err{background:rgba(140,30,20,.55);}
.acia-sd .roi{position:absolute;border:2px solid var(--rc,#e08a12);cursor:move;
  background:color-mix(in srgb,var(--rc,#e08a12) 12%,transparent);}
.acia-sd .roi.active{box-shadow:0 0 0 3px color-mix(in srgb,var(--rc) 30%,transparent);}
.acia-sd .roi .tag{position:absolute;left:0;top:-18px;font-family:var(--font-mono);font-size:10px;
  background:var(--rc);color:#fff;padding:0 5px;border-radius:4px;white-space:nowrap;}
.acia-sd .roi .knob{position:absolute;left:50%;top:-22px;width:11px;height:11px;margin-left:-5.5px;
  border-radius:50%;background:var(--panel);border:2px solid var(--rc);cursor:grab;}
.acia-sd .roi .rz{position:absolute;width:12px;height:12px;border-radius:3px;
  background:var(--panel);border:2px solid var(--rc);}
.acia-sd .roi .rz[data-c="tl"]{left:-6px;top:-6px;cursor:nwse-resize;}
.acia-sd .roi .rz[data-c="tr"]{right:-6px;top:-6px;cursor:nesw-resize;}
.acia-sd .roi .rz[data-c="bl"]{left:-6px;bottom:-6px;cursor:nesw-resize;}
.acia-sd .roi .rz[data-c="br"]{right:-6px;bottom:-6px;cursor:nwse-resize;}
.acia-sd .ptmark{position:absolute;width:11px;height:11px;margin:-5.5px;border-radius:50%;
  background:var(--accent);border:2px solid #fff;pointer-events:none;}
.acia-sd .sd-readout{flex:1;min-width:150px;display:flex;flex-direction:column;gap:12px;
  position:sticky;right:0;background:var(--bg);padding-left:4px;}
.acia-sd .sd-card{background:var(--panel-2);border:1px solid var(--border);border-radius:9px;padding:10px;
  font-family:var(--font-mono);font-size:12px;}
.acia-sd .sd-card h3{margin:0 0 7px;font-size:10px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--text-faint);display:flex;align-items:center;gap:6px;}
.acia-sd .sd-hint{font-size:12px;letter-spacing:normal;text-transform:none;cursor:help;
  color:var(--text-dim);}
.acia-sd .sd-crop{width:100%;height:auto;display:block;border-radius:6px;background:var(--panel-3);}
.acia-sd .sd-frbar{display:flex;align-items:center;gap:8px;padding:7px 14px;flex:none;
  border-top:1px solid var(--border);background:var(--panel-2);
  font-family:var(--font-mono);font-size:11px;color:var(--text-dim);}
.acia-sd .sd-frbar input[type=range]{vertical-align:middle;}
.acia-sd .sd-tools{display:flex;gap:8px;align-items:center;padding:9px 12px;border-top:1px solid var(--border);
  background:var(--panel-2);flex-wrap:wrap;}
.acia-sd button{border:1px solid var(--border);background:var(--panel);color:var(--text);border-radius:7px;
  padding:6px 10px;cursor:pointer;font-size:12.5px;}
.acia-sd button.primary{border-color:var(--accent);color:var(--accent);}
.acia-sd button.accent{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);font-weight:600;}
.acia-sd .sd-list{flex:1;overflow-y:auto;padding:8px;}
.acia-sd .sd-poshd{font-family:var(--font-mono);font-size:11px;color:var(--text-dim);padding:5px 8px;
  cursor:pointer;border-radius:6px;user-select:none;}
.acia-sd .sd-poshd:hover{background:var(--panel-3);color:var(--text);}
.acia-sd .sd-row{display:flex;gap:8px;align-items:center;padding:6px 8px;border-radius:7px;cursor:pointer;}
.acia-sd .sd-row.active{background:var(--accent-soft);}
.acia-sd .sd-row .sw{width:11px;height:11px;border-radius:3px;flex:none;}
.acia-sd .sd-row .nm{flex:1;font-size:13px;}
.acia-sd .sd-foot{border-top:1px solid var(--border);padding:11px;display:flex;flex-direction:column;gap:9px;}
.acia-sd .sd-mode{display:flex;background:var(--panel-3);border-radius:7px;padding:3px;gap:3px;}
.acia-sd .sd-mode button{flex:1;border:0;background:none;font-family:var(--font-mono);font-size:12px;}
.acia-sd .sd-mode button.on{background:var(--panel);box-shadow:0 1px 3px rgba(0,0,0,.15);font-weight:600;}
.acia-sd .sd-saverow{display:flex;align-items:center;gap:10px;}
.acia-sd .sd-saverow .sd-save{flex:1;}
.acia-sd .sd-auto{display:flex;align-items:center;gap:5px;font-family:var(--font-mono);font-size:11px;
  color:var(--text-dim);white-space:nowrap;cursor:pointer;}
.acia-sd .sd-view{display:flex;align-items:center;gap:6px;font-family:var(--font-mono);font-size:10px;
  color:var(--text-faint);}
.acia-sd .sd-toast{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);background:var(--text);
  color:var(--bg);padding:8px 14px;border-radius:8px;font-size:12.5px;opacity:0;transition:.25s;pointer-events:none;}
.acia-sd .sd-toast.show{opacity:1;}
"""

_SEQUENCE_DASHBOARD_ESM = _SEQUENCE_DASHBOARD_ESM_TEXT = r"""
// SequenceDashboard render() -- three-pane curation UI (accordion gallery,
// resizable panes, ROI editor with draw + point-fit). Images arrive from Python
// as PNG bytes (model.send/on). UNVERIFIED headless; validated by a real run.
function render({ model, el }) {
  el.innerHTML = "";
  const root = document.createElement("div");
  root.className = "acia-sd";
  el.appendChild(root);

  const md = model.get("metadata") || {};
  const dims = (md.sizes && md.sizes.Y && md.sizes.X) ? [md.sizes.X, md.sizes.Y] : [1, 1];
  const ASPECT = dims[1] / dims[0];
  const NPOS = md.num_positions || (model.get("positions") || []).length || 1;
  const NT = md.num_timepoints || 1;
  const PX = md.pixel_size_um || null;
  const COLORS = ["--roi-1","--roi-2","--roi-3","--roi-4","--roi-5"];
  const cvar = (v) => getComputedStyle(root).getPropertyValue(v).trim();
  const hexOf = (ci) => cvar(COLORS[((ci % 5) + 5) % 5]);

  const metaLine =
    NPOS + " positions · " + NT + " T · " + dims[0] + "×" + dims[1] +
    " " + (md.dtype || "") + (PX ? " · " + PX.toFixed(4) + " µm/px" : "") +
    " · " + ((md.channels || []).join(", "));

  root.innerHTML =
    "<div class='sd-src'><span>Source</span>" +
    "<input class='sd-path' value='" + (md.path || "") + "' spellcheck='false'>" +
    "<div class='sd-meta'>" + metaLine + "</div></div>" +
    "<div class='sd-main'>" +
      "<div class='sd-pane'><div class='sd-head'><span>Positions</span>" +
        "<span class='sd-galn'>" + NPOS + "</span></div><div class='sd-gal'></div></div>" +
      "<div class='sd-rz' data-side='l'></div>" +
      "<div class='sd-pane sd-editor'><div class='sd-head'>" +
        "<span class='sd-epos'>pos 000</span>" +
        "<span style='display:flex;gap:10px;align-items:center'>" +
        "<span class='sd-view'>size <input type='range' class='sd-vs' min='300' max='2400' value='" +
          model.get("view_size") + "'></span>" +
        "<button class='sd-full' title='Toggle fullscreen (Ctrl/Cmd+scroll over the image also zooms)'>⛶ Fullscreen</button>" +
        "</span></div>" +
        "<div class='sd-ebody'><div class='sd-cwrap'><div class='sd-cstatus'></div></div>" +
        "<div class='sd-readout'><div class='sd-card'><h3>Active ROI → RotatedCropSpec" +
        "<span class='sd-hint' title='Delete/Backspace removes the active ROI; Ctrl/Cmd+C duplicates it (multi mode) -- hover the widget for shortcuts to apply'>⌨</span></h3>" +
        "<div class='sd-spec'>no ROI selected</div></div>" +
        "<div class='sd-card sd-crop-card' hidden><h3>Crop Preview</h3>" +
        "<canvas class='sd-crop'></canvas></div></div></div>" +
        "<div class='sd-frbar'>frame <input type='range' class='sd-frame' min='0' max='" + (NT - 1) + "' value='0'>" +
        " <span class='sd-frlbl'>0 / " + NT + "</span> (view only)</div>" +
        "<div class='sd-tools'><button class='sd-draw'>✎ Draw ROI</button>" +
        "<button class='sd-point primary' title='Click the 4 corners of the rectangle'>✛ Point-fit ROI</button>" +
        "<button class='sd-del' title='Delete the active ROI (Delete/Backspace)'>🗑 Delete</button>" +
        "<span style='margin-left:auto;display:flex;gap:6px;align-items:center'>" +
        "<span class='sd-sw' style='width:12px;height:12px;border-radius:3px'></span>" +
        "<input class='sd-lbl' placeholder='label' style='width:120px'></span></div></div>" +
      "<div class='sd-rz' data-side='r'></div>" +
      "<div class='sd-pane'><div class='sd-head'><span>Selections</span>" +
        "<span class='sd-seln'>0</span></div><div class='sd-list'></div>" +
        "<div class='sd-foot'><div class='sd-mode'>" +
          "<button data-m='single'>single</button><button data-m='multi'>multi</button></div>" +
          "<div class='sd-saverow'>" +
            "<button class='sd-save accent'>💾 Save selection.json</button>" +
            "<label class='sd-auto' title='Automatically write selection.json a moment after each change'>" +
              "<input type='checkbox' class='sd-autochk'> auto-save</label>" +
          "</div></div></div>" +
    "</div><div class='sd-toast'></div>";

  const $ = (s) => root.querySelector(s);
  const gal = $(".sd-gal"), wrap = $(".sd-cwrap"), main = $(".sd-main");
  const cstatus = wrap.querySelector(".sd-cstatus");
  function showStatus(text, isErr) {
    cstatus.textContent = text;
    cstatus.classList.toggle("err", !!isErr);
    cstatus.classList.add("show");
  }
  function hideStatus() { cstatus.classList.remove("show", "err"); }

  // editor geometry: ROIs stored in IMAGE px; s = display px per image px
  let EH = model.get("view_size") || 430;
  let EW = Math.round(EH / ASPECT), s = EW / dims[0];
  wrap.style.width = EW + "px"; wrap.style.height = EH + "px";
  const D = (v) => v * s, I = (v) => v / s;
  // shared by the size slider, ctrl/cmd+scroll zoom, and fullscreen enter/exit
  function applyViewSize(v) {
    EH = Math.max(150, Math.round(v)); EW = Math.round(EH / ASPECT); s = EW / dims[0];
    wrap.style.width = EW + "px"; wrap.style.height = EH + "px";
    $(".sd-vs").value = EH;
    renderEditor();
  }

  let selections = (model.get("selections") || []).map((x) => Object.assign({}, x));
  let currentPos = 0, activeId = null, mode = model.get("roi_mode") || "single";
  const collapsedPos = new Set(); // position headers collapsed in the selections list
  // start past the highest id already present (e.g. resumed from a saved
  // selection.json) so newly drawn/fitted ROIs never collide with those ids
  let frame = 0, uid = selections.reduce((m, x) => Math.max(m, +x.id || 0), 0) + 1, picking = false, points = [];
  const frameImg = new Image();
  // Which position/frame `frameImg` currently holds -- lets the crop preview
  // tell "the loaded pixels match the active ROI's position" apart from
  // "still showing the previous position while the new one is in flight",
  // instead of drawing a confidently-wrong crop from stale pixels.
  let frameImgPos = -1;
  const roisAt = (p) => selections.filter((x) => x.position === p);

  // ---- image requests over the wire ----
  const pending = {};
  model.on("msg:custom", (msg, buffers) => {
    if (!msg) return;
    if (msg.type === "thumb") {
      const cb = pending["t" + msg.pos]; delete pending["t" + msg.pos];
      if (cb && buffers && buffers[0]) cb(blobUrl(buffers[0]));
      thumbInFlight--; pumpThumbQueue();
    } else if (msg.type === "frame") {
      clearTimeout(frameTimer);
      if (buffers && buffers[0]) { frameImgPos = msg.pos; frameImg.src = blobUrl(buffers[0]); }
    } else if (msg.type === "error") {
      if (msg.kind === "thumb") {
        delete pending["t" + msg.pos];
        thumbInFlight--; pumpThumbQueue();
        return;
      }
      if (msg.kind === "frame" && msg.pos === currentPos) {
        clearTimeout(frameTimer);
        showStatus("Error loading pos " + String(msg.pos).padStart(3, "0") + ": " + (msg.message || "failed to load"), true);
        frameImgPos = -1; renderSpec();
        return;
      }
      if (msg.kind === "fit" && picking) {
        // surfaced persistently in the readout (not a fleeting toast) and
        // reset for a fresh 4-point attempt, instead of leaving the counter
        // stuck with no way forward.
        fitting = false; points = []; drawPick();
        pickError = msg.message || "fit failed"; renderSpec();
        return;
      }
      toast("Error: " + (msg.message || "something went wrong"));
    } else if (msg.type === "fit" && msg.roi) {
      finishFit(msg.roi);
    } else if (msg.type === "saved") {
      toast("Saved " + (msg.path || "selection.json"));
    }
  });
  function blobUrl(buf) {
    const arr = buf instanceof DataView ? new Uint8Array(buf.buffer) : new Uint8Array(buf);
    return URL.createObjectURL(new Blob([arr], { type: "image/png" }));
  }
  frameImg.onload = () => { wrap.style.backgroundImage = "url(" + frameImg.src + ")";
    wrap.style.backgroundSize = "cover"; hideStatus(); renderSpec(); };
  frameImg.onerror = () => { clearTimeout(frameTimer); showStatus("Failed to decode frame image", true); };

  // ---- gallery (accordion + lazy thumb) ----
  // A Jupyter kernel processes comm messages one at a time on a single
  // thread, and each thumb read is a blocking SMB call -- scrolling the
  // gallery can bring many rows into view almost at once, and firing a
  // request per row would queue dozens of slow reads ahead of anything else
  // (e.g. a point-fit or Save the user triggers moments later). Throttle to a
  // small number in flight so the wire queue stays short.
  const MAX_THUMB_INFLIGHT = 2;
  let thumbInFlight = 0;
  const thumbQueue = [];
  function pumpThumbQueue() {
    while (thumbInFlight < MAX_THUMB_INFLIGHT && thumbQueue.length) {
      const ix = thumbQueue.shift();
      thumbInFlight++;
      model.send({ type: "thumb", pos: ix, downscale: 8 });
    }
  }
  const io = new IntersectionObserver((ents) => {
    ents.forEach((e) => {
      if (e.isIntersecting) {
        const t = e.target, ix = +t.dataset.ix;
        pending["t" + ix] = (url) => {
          const img = document.createElement("img"); img.src = url; t.insertBefore(img, t.firstChild);
        };
        thumbQueue.push(ix);
        io.unobserve(t);
      }
    });
    pumpThumbQueue();
  }, { root: gal, rootMargin: "150px" });
  function computeExp() {
    const w = gal.clientWidth - 16;
    root.querySelectorAll(".sd-thumb").forEach((t) => t.style.setProperty("--exp", Math.max(150, Math.round(w * ASPECT)) + "px"));
  }
  function buildGallery() {
    gal.innerHTML = "";
    for (let i = 0; i < NPOS; i++) {
      const d = document.createElement("div");
      d.className = "sd-thumb" + (i === currentPos ? " sel" : "");
      d.dataset.ix = i;
      d.innerHTML = "<span class='ix'>pos " + String(i).padStart(3, "0") + "</span>" +
        (roisAt(i).length ? "<span class='dot'></span>" : "");
      d.onclick = () => selectPos(i);
      gal.appendChild(d); io.observe(d);
    }
    computeExp();
  }

  // ---- editor ----
  let frameTimer = null;
  function requestFrame() {
    clearTimeout(frameTimer);
    hideStatus();
    frameTimer = setTimeout(() => showStatus("Loading pos " + String(currentPos).padStart(3, "0") + "…", false), 150);
    model.send({ type: "frame", pos: currentPos, t: frame });
  }
  function renderEditor() {
    wrap.querySelectorAll(".roi:not(.preview)").forEach((n) => n.remove());
    roisAt(currentPos).forEach((sel) => {
      const eln = document.createElement("div");
      eln.className = "roi" + (sel.id === activeId ? " active" : "");
      eln.style.cssText = "--rc:" + hexOf(sel.ci) + ";left:" + D(sel.x) + "px;top:" + D(sel.y) +
        "px;width:" + D(sel.w) + "px;height:" + D(sel.h) +
        "px;transform:translate(-50%,-50%) rotate(" + sel.angle + "deg)";
      eln.innerHTML = "<span class='tag'>" + sel.label + "</span><span class='knob'></span>" +
        "<span class='rz' data-c='tl'></span><span class='rz' data-c='tr'></span>" +
        "<span class='rz' data-c='bl'></span><span class='rz' data-c='br'></span>";
      eln.addEventListener("pointerdown", (ev) => startDrag(ev, sel, "move", eln));
      eln.querySelector(".knob").addEventListener("pointerdown", (ev) => startDrag(ev, sel, "rotate", eln));
      eln.querySelectorAll(".rz").forEach((h) => h.addEventListener(
        "pointerdown", (ev) => startDrag(ev, sel, "resize", eln, h.dataset.c)));
      wrap.appendChild(eln);
    });
    $(".sd-epos").textContent = "pos " + String(currentPos).padStart(3, "0");
    renderSpec(); renderLabel();
  }
  function renderSpec() {
    const box = $(".sd-spec");
    if (picking) { box.textContent = pickHint(); renderCropPreview(null); return; }
    const sel = selections.find((x) => x.id === activeId);
    if (!sel || sel.position !== currentPos) { box.textContent = "no ROI selected"; renderCropPreview(null); return; }
    const um = PX ? " (" + (sel.w * PX).toFixed(1) + "×" + (sel.h * PX).toFixed(1) + " µm)" : "";
    box.innerHTML = "center " + Math.round(sel.x) + ", " + Math.round(sel.y) + " px<br>" +
      "size " + Math.round(sel.w) + "×" + Math.round(sel.h) + " px" + um + "<br>angle " + sel.angle.toFixed(1) + "°";
    renderCropPreview(sel);
  }
  // Renders exactly what crop_rotated(RotatedCropSpec(...)) would produce for
  // the active ROI: sample the already-loaded frame image into a small
  // un-rotated canvas, de-rotating by the SAME angle (and sense) the ROI box
  // itself is drawn with (CSS `rotate(sel.angle + "deg")` in renderEditor),
  // so the preview always matches what's on screen without a kernel round-trip.
  const MAX_PREVIEW_PX = 220;
  function renderCropPreview(sel) {
    const card = $(".sd-crop-card");
    if (!sel || frameImgPos !== currentPos || !frameImg.complete || !frameImg.naturalWidth) {
      card.hidden = true;
      return;
    }
    card.hidden = false;
    const canvas = $(".sd-crop");
    const w = Math.max(1, Math.round(sel.w)), h = Math.max(1, Math.round(sel.h));
    const scalePrev = Math.min(1, MAX_PREVIEW_PX / Math.max(w, h));
    const cw = Math.max(1, Math.round(w * scalePrev)), ch = Math.max(1, Math.round(h * scalePrev));
    if (canvas.width !== cw) canvas.width = cw;
    if (canvas.height !== ch) canvas.height = ch;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, cw, ch);
    ctx.save();
    ctx.translate(cw / 2, ch / 2);
    ctx.rotate(-(sel.angle * Math.PI) / 180);
    ctx.scale(scalePrev, scalePrev);
    ctx.drawImage(frameImg, -sel.x, -sel.y);
    ctx.restore();
  }
  function renderLabel() {
    const sel = selections.find((x) => x.id === activeId);
    const inp = $(".sd-lbl"), sw = $(".sd-sw");
    if (sel && sel.position === currentPos) { inp.value = sel.label; inp.disabled = false; sw.style.background = hexOf(sel.ci); }
    else { inp.value = ""; inp.disabled = true; sw.style.background = "var(--border)"; }
  }
  function selectPos(p) {
    if (picking) exitPick();
    currentPos = p; const h = roisAt(p); activeId = h.length ? h[0].id : null;
    root.querySelectorAll(".sd-thumb").forEach((t) => t.classList.toggle("sel", +t.dataset.ix === p));
    frame = 0; $(".sd-frame").value = 0; $(".sd-frlbl").textContent = "0 / " + NT;
    requestFrame(); renderEditor(); renderList();
  }

  // ---- selections ----
  let autosave = false, autoSaveTimer = null;
  function pushSelections() {
    model.set("selections", selections.map((x) => ({
      id: x.id, position: x.position, label: x.label, ci: x.ci,
      roi: { center: [x.x, x.y], size: [Math.round(x.w), Math.round(x.h)], angle: x.angle },
    })));
    model.save_changes();
    if (autosave) {
      // debounced so rapid edits (dragging, typing a label) trigger one
      // disk write shortly after things settle, not one per keystroke/pixel
      clearTimeout(autoSaveTimer);
      autoSaveTimer = setTimeout(() => model.send({ type: "save" }), 800);
    }
  }
  function renderList() {
    const list = $(".sd-list"); list.innerHTML = "";
    const byPos = {}; selections.forEach((x) => { (byPos[x.position] = byPos[x.position] || []).push(x); });
    Object.keys(byPos).map(Number).sort((a, b) => a - b).forEach((p) => {
      const collapsed = collapsedPos.has(p);
      const hd = document.createElement("div");
      hd.className = "sd-poshd";
      hd.textContent = (collapsed ? "▸" : "▾") + " pos " + String(p).padStart(3, "0") +
        " (" + byPos[p].length + ")";
      hd.onclick = () => {
        if (collapsedPos.has(p)) collapsedPos.delete(p); else collapsedPos.add(p);
        renderList();
      };
      list.appendChild(hd);
      if (collapsed) return;
      byPos[p].forEach((sel) => {
        const row = document.createElement("div");
        row.className = "sd-row" + (sel.id === activeId ? " active" : "");
        row.innerHTML = "<span class='sw' style='background:" + hexOf(sel.ci) + "'></span>" +
          "<span class='nm'>" + sel.label + "</span>" +
          "<span style='font-family:var(--font-mono);font-size:10px;color:var(--text-faint)'>" +
          Math.round(sel.w) + "×" + Math.round(sel.h) + "</span>";
        row.onclick = () => { currentPos = sel.position; activeId = sel.id;
          root.querySelectorAll(".sd-thumb").forEach((t) => t.classList.toggle("sel", +t.dataset.ix === currentPos));
          requestFrame(); renderEditor(); renderList(); };
        list.appendChild(row);
      });
    });
    $(".sd-seln").textContent = selections.length;
    root.querySelectorAll(".sd-thumb").forEach((t) => {
      const has = roisAt(+t.dataset.ix).length;
      let dot = t.querySelector(".dot");
      if (has && !dot) { dot = document.createElement("span"); dot.className = "dot"; t.appendChild(dot); }
      if (!has && dot) dot.remove();
    });
  }
  function addRoi() {
    if (mode === "single") selections = selections.filter((x) => x.position !== currentPos);
    const h = roisAt(currentPos), ci = h.length ? Math.max.apply(null, h.map((x) => x.ci)) + 1 : 0;
    const sel = { id: uid++, position: currentPos, x: dims[0] / 2, y: dims[1] / 2,
      w: Math.round(dims[0] / 3), h: Math.round(dims[1] / 3), angle: 0,
      label: "roi_" + String(ci + 1).padStart(2, "0"), ci: ci };
    selections.push(sel); activeId = sel.id;
    renderEditor(); renderList(); pushSelections();
  }
  function duplicateActive() {
    const src = selections.find((x) => x.id === activeId);
    if (!src || src.position !== currentPos) return;
    const h = roisAt(currentPos), ci = h.length ? Math.max.apply(null, h.map((x) => x.ci)) + 1 : 0;
    // Offset so the copy doesn't sit exactly on top of the source (which would
    // otherwise look like nothing happened) but stays inside the frame. Flip
    // the offset's sign per-axis when the source is already flush against
    // that edge, so a corner-positioned source doesn't clamp both axes back
    // to the same point as the source.
    const OFFSET = 24;
    const offX = (src.x + OFFSET <= dims[0] - 5) ? OFFSET : -OFFSET;
    const offY = (src.y + OFFSET <= dims[1] - 5) ? OFFSET : -OFFSET;
    const sel = { id: uid++, position: currentPos,
      x: Math.max(5, Math.min(dims[0] - 5, src.x + offX)),
      y: Math.max(5, Math.min(dims[1] - 5, src.y + offY)),
      w: src.w, h: src.h, angle: src.angle,
      label: "roi_" + String(ci + 1).padStart(2, "0"), ci: ci };
    selections.push(sel); activeId = sel.id;
    renderEditor(); renderList(); pushSelections(); toast("Duplicated ROI -- drag it into place");
  }
  function removeSel(id) {
    selections = selections.filter((x) => x.id !== id);
    if (activeId === id) { const h = roisAt(currentPos); activeId = h.length ? h[0].id : null; }
    renderEditor(); renderList(); pushSelections();
  }

  // ---- drag / rotate / resize ----
  let drag = null;
  function startDrag(ev, sel, kind, eln, corner) {
    if (picking) return;
    ev.preventDefault(); ev.stopPropagation();
    if (activeId !== sel.id) { activeId = sel.id; renderEditor(); renderList(); }
    drag = { sel, kind, eln, rect: wrap.getBoundingClientRect(),
      px: ev.clientX, py: ev.clientY, ox: sel.x, oy: sel.y };
    if (kind === "resize") {
      // Anchor the corner diagonally opposite the one being dragged: it stays
      // fixed in image space for the whole drag, so only the two dimensions
      // toward the dragged corner change (w/h + the center shift that keeps
      // the anchor put), instead of resizing symmetrically about the center.
      const signX = (corner === "tr" || corner === "br") ? 1 : -1;
      const signY = (corner === "bl" || corner === "br") ? 1 : -1;
      const theta = sel.angle * Math.PI / 180, cos = Math.cos(theta), sin = Math.sin(theta);
      const alx = -signX * sel.w / 2, aly = -signY * sel.h / 2;
      drag.anchorX = sel.x + (alx * cos - aly * sin);
      drag.anchorY = sel.y + (alx * sin + aly * cos);
      drag.signX = signX; drag.signY = signY; drag.rAngle = theta;
    }
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp, { once: true });
  }
  function onMove(ev) {
    if (!drag) return;
    const { sel, kind, rect } = drag, mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    if (kind === "move") {
      sel.x = Math.max(5, Math.min(dims[0] - 5, drag.ox + I(ev.clientX - drag.px)));
      sel.y = Math.max(5, Math.min(dims[1] - 5, drag.oy + I(ev.clientY - drag.py)));
    } else if (kind === "rotate") {
      sel.angle = Math.round((Math.atan2(my - D(sel.y), mx - D(sel.x)) * 180 / Math.PI + 90) * 10) / 10;
    } else if (kind === "resize") {
      const cos = Math.cos(drag.rAngle), sin = Math.sin(drag.rAngle);
      const vx = I(mx) - drag.anchorX, vy = I(my) - drag.anchorY;
      // anchor->pointer vector, expressed in the box's own (unrotated) frame
      const lx = vx * cos + vy * sin, ly = -vx * sin + vy * cos;
      sel.w = Math.max(20, Math.round(Math.abs(lx)));
      sel.h = Math.max(20, Math.round(Math.abs(ly)));
      const hx = drag.signX * sel.w / 2, hy = drag.signY * sel.h / 2;
      sel.x = drag.anchorX + (hx * cos - hy * sin);
      sel.y = drag.anchorY + (hx * sin + hy * cos);
    }
    const eln = drag.eln;
    eln.style.left = D(sel.x) + "px"; eln.style.top = D(sel.y) + "px";
    eln.style.width = D(sel.w) + "px"; eln.style.height = D(sel.h) + "px";
    eln.style.transform = "translate(-50%,-50%) rotate(" + sel.angle + "deg)";
    renderSpec();
  }
  function onUp() { window.removeEventListener("pointermove", onMove); drag = null; renderList(); pushSelections(); }

  // ---- point-fit (via Python cv2) ----
  const FIT_POINTS = 4; // one click per corner of the intended rectangle
  let fitting = false, pickError = null; // fitting: request in flight; pickError: last failure (persistent, not a fleeting toast)
  function pickHint() {
    if (fitting) return "Fitting…";
    if (pickError) return "Error: " + pickError + " -- click " + FIT_POINTS + " corners again (0 / " + FIT_POINTS + ")";
    return "Click the " + FIT_POINTS + " corners (" + points.length + " / " + FIT_POINTS + ")";
  }
  function enterPick() { picking = true; points = []; fitting = false; pickError = null; activeId = null;
    wrap.classList.add("pick"); renderEditor(); drawPick(); $(".sd-point").classList.add("on"); }
  function exitPick() { picking = false; points = []; fitting = false;
    wrap.classList.remove("pick"); wrap.querySelectorAll(".ptmark").forEach((n) => n.remove());
    $(".sd-point").classList.remove("on"); }
  wrap.addEventListener("click", (ev) => {
    if (!picking || fitting) return;
    const r = wrap.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    if (x < 0 || y < 0 || x > EW || y > EH) return;
    pickError = null;
    points.push([I(x), I(y)]); drawPick();
    if (points.length >= FIT_POINTS) { fitting = true; model.send({ type: "fit", points: points }); }
    renderSpec();
  });
  function drawPick() {
    wrap.querySelectorAll(".ptmark").forEach((n) => n.remove());
    points.forEach((p) => { const d = document.createElement("div"); d.className = "ptmark";
      d.style.left = D(p[0]) + "px"; d.style.top = D(p[1]) + "px"; wrap.appendChild(d); });
  }
  function finishFit(roi) {
    if (!picking) return;
    // commit immediately as a normal, adjustable ROI -- renderEditor() below
    // draws it with the usual drag/rotate/resize handles right away.
    const cx = roi.center[0], cy = roi.center[1], w = roi.size[0], h = roi.size[1];
    if (mode === "single") selections = selections.filter((x) => x.position !== currentPos);
    const hh = roisAt(currentPos), ci = hh.length ? Math.max.apply(null, hh.map((x) => x.ci)) + 1 : 0;
    const sel = { id: uid++, position: currentPos, x: cx, y: cy, w: w, h: h, angle: roi.angle,
      label: "roi_" + String(ci + 1).padStart(2, "0"), ci: ci };
    selections.push(sel); activeId = sel.id; exitPick();
    renderEditor(); renderList(); pushSelections(); toast("Fitted ROI from points");
  }

  // ---- controls ----
  $(".sd-vs").addEventListener("input", (e) => {
    applyViewSize(+e.target.value); model.set("view_size", EH); model.save_changes();
  });
  // Ctrl/Cmd+scroll zooms the editor image; plain scroll passes through to pan
  // the surrounding (overflow:auto) pane, so it never fights normal scrolling.
  wrap.addEventListener("wheel", (e) => {
    if (!(e.ctrlKey || e.metaKey)) return;
    e.preventDefault();
    applyViewSize(EH * (e.deltaY < 0 ? 1.12 : 1 / 1.12));
    model.set("view_size", EH); model.save_changes();
  }, { passive: false });
  // ---- fullscreen ----
  let preFsViewSize = EH;
  const fsBtn = $(".sd-full");
  const requestFs = root.requestFullscreen || root.webkitRequestFullscreen;
  const exitFs = document.exitFullscreen || document.webkitExitFullscreen;
  fsBtn.onclick = () => {
    if (!requestFs) { toast("Fullscreen not supported in this browser"); return; }
    if (document.fullscreenElement === root) exitFs.call(document);
    else requestFs.call(root);
  };
  function onFsChange() {
    const isFs = document.fullscreenElement === root;
    fsBtn.textContent = isFs ? "⤢ Exit fullscreen" : "⛶ Fullscreen";
    if (isFs) { preFsViewSize = EH; applyViewSize(Math.round(window.innerHeight * 0.82)); }
    else applyViewSize(preFsViewSize);
  }
  document.addEventListener("fullscreenchange", onFsChange);
  $(".sd-frame").addEventListener("input", (e) => { frame = +e.target.value;
    $(".sd-frlbl").textContent = frame + " / " + NT; requestFrame(); });
  $(".sd-draw").onclick = () => { if (picking) exitPick(); addRoi(); };
  $(".sd-point").onclick = () => { picking ? exitPick() : enterPick(); if (!picking) renderEditor(); };
  $(".sd-del").onclick = () => { if (activeId) removeSel(activeId); };
  $(".sd-lbl").addEventListener("input", (e) => { const sel = selections.find((x) => x.id === activeId);
    if (sel) { sel.label = e.target.value; renderEditor(); renderList(); pushSelections(); } });
  root.querySelectorAll(".sd-mode button").forEach((b) => { b.onclick = () => {
    mode = b.dataset.m; model.set("roi_mode", mode); model.save_changes();
    root.querySelectorAll(".sd-mode button").forEach((x) => x.classList.toggle("on", x === b));
    if (mode === "single") { const seen = {}; selections = selections.filter((x) => {
      if (seen[x.position]) return false; seen[x.position] = 1; return true; });
      renderEditor(); renderList(); pushSelections(); } }; });
  root.querySelectorAll(".sd-mode button").forEach((x) => x.classList.toggle("on", x.dataset.m === mode));
  $(".sd-save").onclick = () => model.send({ type: "save" });
  $(".sd-autochk").addEventListener("change", (e) => { autosave = e.target.checked; });

  // ---- keyboard shortcuts (Delete/Backspace, Ctrl/Cmd+C) ----
  // Scoped to "mouse is over this widget instance" (not tab/window focus) so
  // multiple dashboards in one notebook, or typing elsewhere on the page,
  // don't cross-trigger each other -- same reasoning as the wheel-zoom above.
  let hovering = false;
  root.addEventListener("pointerenter", () => { hovering = true; });
  root.addEventListener("pointerleave", () => { hovering = false; });
  function onKeyDown(ev) {
    if (!hovering || picking) return;
    const t = ev.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    if (!activeId) return;
    if (ev.key === "Delete" || ev.key === "Backspace") {
      ev.preventDefault(); removeSel(activeId);
    } else if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "c") {
      // Only steal the shortcut when we're actually going to act: leave native
      // copy alone in single mode (duplicate is meaningless there -- at most
      // one ROI per position) and when the user has text selected (they're
      // copying that, not asking to duplicate the ROI).
      if (mode !== "multi") return;
      const textSel = window.getSelection();
      if (textSel && textSel.toString()) return;
      ev.preventDefault(); duplicateActive();
    }
  }
  document.addEventListener("keydown", onKeyDown);

  // ---- splitters ----
  root.querySelectorAll(".sd-rz").forEach((rz) => {
    rz.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      const side = rz.dataset.side, sx = e.clientX;
      const pane = side === "l" ? main.children[0] : main.children[4];
      const sw = pane.getBoundingClientRect().width;
      const mv = (ev) => { const d = ev.clientX - sx;
        if (side === "l") { main.style.setProperty("--lw", Math.max(140, Math.min(360, sw + d)) + "px"); computeExp(); }
        else main.style.setProperty("--rw", Math.max(180, Math.min(420, sw - d)) + "px"); };
      const up = () => { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); };
      window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up, { once: true });
    });
  });

  let toastT;
  function toast(m) { const t = root.querySelector(".sd-toast"); t.textContent = m; t.classList.add("show");
    clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("show"), 2200); }

  buildGallery(); selectPos(0); renderList();
  return () => { io.disconnect(); document.removeEventListener("fullscreenchange", onFsChange);
    document.removeEventListener("keydown", onKeyDown); };
}
export default { render };
"""


# The RegistrationDashboard ESM: method/position picker, a verify view (drift
# trajectory + before/after toggle), a mask-rect editor for
# MaskedTemplateCorrelation (porting ROICropper's click-to-fit + drag/resize/
# rotate interaction model -- same corner/rotate-handle math, own canvas/model
# keys so ROICropper itself is untouched), and a batch-apply panel with a live
# progress bar fed by the widget's "progress" messages. Like the other widgets
# here, this is BEST-EFFORT JavaScript, unverified by the headless Python
# suite -- validated only by a real Jupyter/Colab/marimo run (no ESM/Playwright
# suite for this widget in v1, per the spec's Never section).
_REGISTRATION_DASHBOARD_ESM = r"""
// RegistrationDashboard render() -- method/position/verify controls, a mask-rect
// editor (shown only for MaskedTemplateCorrelation), a drift-trajectory +
// before/after verify view, and a batch-apply panel with a live progress bar.
// UNVERIFIED in CI: validated by a real Jupyter/Colab/marimo run only.
function render({ model, el }) {
  el.innerHTML = "";
  const root = document.createElement("div");
  root.style.font = "13px sans-serif";
  root.style.border = "1px solid #ccc";
  root.style.borderRadius = "8px";
  root.style.padding = "10px";
  root.style.maxWidth = "720px";
  el.appendChild(root);

  const METHODS = [
    "PhaseCorrelationHighpass",
    "MaskedTemplateCorrelation",
    "HoughLineRigidFit",
    "FeatureRANSACEuclidean",
    "GradientECC",
  ];

  const md = model.get("metadata") || {};
  const numPositions = md.num_positions || (model.get("positions") || []).length || 1;

  root.innerHTML =
    "<div class='rd-head' style='display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px;'>" +
      "<label>Method <select class='rd-method'>" +
        METHODS.map((m) => "<option value='" + m + "'>" + m + "</option>").join("") +
      "</select></label>" +
      "<label>Position <input class='rd-pos' type='number' min='0' max='" + (numPositions - 1) +
        "' value='0' style='width:60px'></label>" +
      "<label>Samples <input class='rd-nsamp' type='number' min='1' value='" +
        model.get("n_sample_frames") + "' style='width:50px'></label>" +
      "<button class='rd-verify'>Verify</button>" +
    "</div>" +
    "<div class='rd-mask' style='display:none;margin-bottom:10px;'>" +
      "<div style='font-size:11px;color:#666;margin-bottom:4px;'>Mask rect for " +
      "MaskedTemplateCorrelation: click &ge;3 points around a static landmark " +
      "(frame 0 of the position above), or drag the box / corners / rotate knob.</div>" +
      "<canvas class='rd-mask-canvas' style='touch-action:none;border:1px solid #999;'></canvas>" +
    "</div>" +
    "<div class='rd-verify-out' style='display:none;margin-bottom:10px;'>" +
      "<canvas class='rd-traj' width='640' height='140' style='border:1px solid #ddd;'></canvas>" +
      "<div class='rd-player' style='margin-top:8px;'>" +
        "<div style='display:flex;gap:8px;'>" +
          "<canvas class='rd-player-before' width='260' height='260' style='border:1px solid #ddd;max-width:48%;background:#111;'></canvas>" +
          "<canvas class='rd-player-after' width='260' height='260' style='border:1px solid #ddd;max-width:48%;background:#111;'></canvas>" +
        "</div>" +
        "<div style='display:flex;gap:8px;align-items:center;margin-top:4px;'>" +
          "<button type='button' class='rd-play-btn'>Play</button>" +
          "<input type='range' class='rd-scrubber' min='0' max='0' value='0' step='1' style='flex:1;'>" +
          "<span class='rd-player-label' style='font-size:11px;color:#666;white-space:nowrap;'></span>" +
        "</div>" +
      "</div>" +
    "</div>" +
    "<div class='rd-status' style='font-size:11px;color:#a33;margin-bottom:6px;'></div>" +
    "<div class='rd-batch' style='border-top:1px solid #ddd;padding-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;'>" +
      "<input class='rd-dir' placeholder='(current working directory)' style='flex:1;min-width:160px;'>" +
      "<button class='rd-batch-btn'>Batch Apply</button>" +
      "<button class='rd-save-btn'>Save</button>" +
    "</div>" +
    "<div class='rd-progress-wrap' style='display:none;margin-top:8px;'>" +
      "<div style='background:#eee;border-radius:4px;height:10px;overflow:hidden;'>" +
        "<div class='rd-progress-bar' style='background:#0f8f82;height:100%;width:0%;'></div>" +
      "</div>" +
      "<div class='rd-progress-label' style='font-size:11px;color:#666;margin-top:2px;'></div>" +
    "</div>";

  const $ = (s) => root.querySelector(s);
  const methodSel = $(".rd-method"), posInput = $(".rd-pos"), nsampInput = $(".rd-nsamp");
  const maskWrap = $(".rd-mask"), maskCanvas = $(".rd-mask-canvas");
  const verifyOut = $(".rd-verify-out"), trajCanvas = $(".rd-traj");
  const playerBeforeCanvas = $(".rd-player-before"), playerAfterCanvas = $(".rd-player-after");
  const playBtn = $(".rd-play-btn"), scrubber = $(".rd-scrubber"), playerLabel = $(".rd-player-label");
  const statusEl = $(".rd-status");
  const dirInput = $(".rd-dir"), progWrap = $(".rd-progress-wrap");
  const progBar = $(".rd-progress-bar"), progLabel = $(".rd-progress-label");

  methodSel.value = model.get("method_name");

  function updateMaskVisibility() {
    maskWrap.style.display = methodSel.value === "MaskedTemplateCorrelation" ? "block" : "none";
  }
  updateMaskVisibility();

  function requestMaskFrame() {
    model.send({ type: "mask_frame", position: parseInt(posInput.value, 10) || 0 });
  }

  methodSel.addEventListener("change", () => {
    model.set("method_name", methodSel.value);
    model.save_changes();
    updateMaskVisibility();
    if (methodSel.value === "MaskedTemplateCorrelation") {
      layoutMask();
      requestMaskFrame();
    }
  });
  nsampInput.addEventListener("change", () => {
    model.set("n_sample_frames", Math.max(1, parseInt(nsampInput.value, 10) || 1));
    model.save_changes();
  });
  posInput.addEventListener("change", () => {
    if (methodSel.value === "MaskedTemplateCorrelation") requestMaskFrame();
  });

  function showStatus(msg, isErr) {
    statusEl.textContent = msg || "";
    statusEl.style.color = isErr ? "#a33" : "#666";
  }

  function blobUrl(buf) {
    const arr = buf instanceof DataView ? new Uint8Array(buf.buffer) : new Uint8Array(buf);
    return URL.createObjectURL(new Blob([arr], { type: "image/png" }));
  }

  // ---- mask-rect editor: click-to-fit + drag/resize/rotate, porting
  // ROICropper's interaction model (same corner/rotate-handle geometry) onto
  // this widget's own mask_* traits/canvas -- ROICropper itself is untouched.
  const maskImg = new Image();
  let maskReady = false;
  maskImg.onload = () => { maskReady = true; layoutMask(); drawMask(); };
  maskImg.onerror = () => { maskReady = false; drawMask(); };
  let mscale = 1;
  function maskImgW() { return model.get("mask_image_w") || 1; }
  function maskImgH() { return model.get("mask_image_h") || 1; }
  function layoutMask() {
    const w = maskImgW(), h = maskImgH();
    const MAX_W = 480;
    mscale = Math.min(1, MAX_W / w);
    maskCanvas.width = Math.round(w * mscale);
    maskCanvas.height = Math.round(h * mscale);
  }
  function mToImg(px, py) { return [px / mscale, py / mscale]; }
  function mToCanvas(ix, iy) { return [ix * mscale, iy * mscale]; }
  function getMaskRect() {
    return {
      cx: model.get("mask_center_x"), cy: model.get("mask_center_y"),
      w: model.get("mask_width"), h: model.get("mask_height"), angle: model.get("mask_angle"),
    };
  }
  function maskCorners(r) {
    const a = (r.angle * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a);
    const hw = r.w / 2, hh = r.h / 2;
    const local = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
    return local.map(([lx, ly]) => [r.cx + lx * ca + ly * sa, r.cy - lx * sa + ly * ca]);
  }
  function maskLocalToImg(r, lx, ly) {
    const a = (r.angle * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a);
    return [r.cx + lx * ca + ly * sa, r.cy - lx * sa + ly * ca];
  }
  function maskImgToLocal(r, ix, iy) {
    const a = (r.angle * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a);
    const dx = ix - r.cx, dy = iy - r.cy;
    return [dx * ca - dy * sa, dx * sa + dy * ca];
  }
  function maskRotateHandle(r) {
    const off = r.h / 2 + 24 / mscale;
    return maskLocalToImg(r, 0, -off);
  }
  function drawMask() {
    const ctx = maskCanvas.getContext("2d");
    ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    if (maskReady) ctx.drawImage(maskImg, 0, 0, maskCanvas.width, maskCanvas.height);
    const pts = model.get("mask_points") || [];
    ctx.fillStyle = "#00e5ff";
    for (const [ix, iy] of pts) {
      const [px, py] = mToCanvas(ix, iy);
      ctx.beginPath(); ctx.arc(px, py, 3, 0, 2 * Math.PI); ctx.fill();
    }
    const r = getMaskRect();
    if (r.w > 0 && r.h > 0) {
      const cs = maskCorners(r).map(([ix, iy]) => mToCanvas(ix, iy));
      ctx.strokeStyle = "#ffeb3b"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(cs[0][0], cs[0][1]);
      for (let i = 1; i < cs.length; i++) ctx.lineTo(cs[i][0], cs[i][1]);
      ctx.closePath(); ctx.stroke();
      ctx.fillStyle = "#ffeb3b";
      for (const [px, py] of cs) ctx.fillRect(px - 4, py - 4, 8, 8);
      const [rx, ry] = mToCanvas(...maskRotateHandle(r));
      const [ccx, ccy] = mToCanvas(r.cx, r.cy);
      ctx.strokeStyle = "#ff5252";
      ctx.beginPath(); ctx.moveTo(ccx, ccy); ctx.lineTo(rx, ry); ctx.stroke();
      ctx.fillStyle = "#ff5252";
      ctx.beginPath(); ctx.arc(rx, ry, 5, 0, 2 * Math.PI); ctx.fill();
    }
  }
  let mdrag = null;
  const MOVE_THRESHOLD = 4;
  function maskLocalPos(ev) {
    const rect = maskCanvas.getBoundingClientRect();
    const sx = rect.width ? maskCanvas.width / rect.width : 1;
    const sy = rect.height ? maskCanvas.height / rect.height : 1;
    return [(ev.clientX - rect.left) * sx, (ev.clientY - rect.top) * sy];
  }
  function maskHit(px, py) {
    const r = getMaskRect();
    if (r.w > 0 && r.h > 0) {
      const cs = maskCorners(r).map(([ix, iy]) => mToCanvas(ix, iy));
      for (let i = 0; i < cs.length; i++) {
        if (Math.hypot(px - cs[i][0], py - cs[i][1]) <= 8) return { mode: "resize", cornerIndex: i };
      }
      const [rx, ry] = mToCanvas(...maskRotateHandle(r));
      if (Math.hypot(px - rx, py - ry) <= 8) return { mode: "rotate" };
      const [ix, iy] = mToImg(px, py);
      const [lx, ly] = maskImgToLocal(r, ix, iy);
      if (Math.abs(lx) <= r.w / 2 && Math.abs(ly) <= r.h / 2) return { mode: "move" };
    }
    return null;
  }
  function addMaskPoint(px, py) {
    const [ix, iy] = mToImg(px, py);
    const pts = (model.get("mask_points") || []).slice();
    pts.push([ix, iy]);
    model.set("mask_points", pts);
    model.save_changes();
    drawMask();
  }
  function onMaskDown(ev) {
    const [px, py] = maskLocalPos(ev);
    const h = maskHit(px, py);
    const r = getMaskRect();
    const [ix, iy] = mToImg(px, py);
    mdrag = {
      ...(h || {}), candidateMode: h ? h.mode : null, mode: null,
      r, ix, iy, startPx: px, startPy: py, moved: false,
    };
    try { maskCanvas.setPointerCapture(ev.pointerId); } catch (e) {}
  }
  function onMaskMove(ev) {
    if (!mdrag) return;
    const [px, py] = maskLocalPos(ev);
    if (!mdrag.moved) {
      if (Math.hypot(px - mdrag.startPx, py - mdrag.startPy) < MOVE_THRESHOLD) return;
      mdrag.moved = true;
      mdrag.mode = mdrag.candidateMode;
    }
    if (!mdrag.mode) return;
    const [ix, iy] = mToImg(px, py);
    const r = mdrag.r;
    if (mdrag.mode === "move") {
      model.set("mask_center_x", r.cx + (ix - mdrag.ix));
      model.set("mask_center_y", r.cy + (iy - mdrag.iy));
    } else if (mdrag.mode === "rotate") {
      const angle = Math.atan2(-(ix - r.cx), -(iy - r.cy)) * 180 / Math.PI;
      model.set("mask_angle", angle);
    } else if (mdrag.mode === "resize") {
      const hw = r.w / 2, hh = r.h / 2;
      const localCorners = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
      const i = mdrag.cornerIndex;
      const opp = localCorners[(i + 2) % 4];
      const [ox, oy] = maskLocalToImg(r, opp[0], opp[1]);
      const ncx = (ix + ox) / 2, ncy = (iy + oy) / 2;
      const rAxes = { ...r, cx: ncx, cy: ncy };
      const [lx, ly] = maskImgToLocal(rAxes, ix, iy);
      const oLocal = maskImgToLocal(rAxes, ox, oy);
      const nw = Math.max(1, Math.round(Math.abs(lx - oLocal[0])));
      const nh = Math.max(1, Math.round(Math.abs(ly - oLocal[1])));
      model.set("mask_center_x", ncx); model.set("mask_center_y", ncy);
      model.set("mask_width", nw); model.set("mask_height", nh);
    }
    model.save_changes();
    drawMask();
  }
  function onMaskUp(ev) {
    if (!mdrag) return;
    if (!mdrag.moved) addMaskPoint(mdrag.startPx, mdrag.startPy);
    try { maskCanvas.releasePointerCapture(ev.pointerId); } catch (e) {}
    mdrag = null;
  }
  maskCanvas.addEventListener("pointerdown", onMaskDown);
  maskCanvas.addEventListener("pointermove", onMaskMove);
  maskCanvas.addEventListener("pointerup", onMaskUp);
  maskCanvas.addEventListener("pointercancel", onMaskUp);

  function onMaskImageChange() {
    const b64 = model.get("mask_image_b64");
    if (b64) { maskImg.src = b64; }
  }
  function onMaskGeomChange() { drawMask(); }
  model.on("change:mask_image_b64", onMaskImageChange);
  model.on(
    "change:mask_center_x change:mask_center_y change:mask_width change:mask_height " +
      "change:mask_angle change:mask_points",
    onMaskGeomChange,
  );

  // ---- verify: drift trajectory (dx/dy/theta) + a play/pause+scrubber
  // side-by-side comparison player over every sampled frame ----
  $(".rd-verify").addEventListener("click", () => {
    showStatus("");
    progWrap.style.display = "block";
    progBar.style.width = "0%";
    progLabel.textContent = "starting verify...";
    model.send({
      type: "verify",
      position: parseInt(posInput.value, 10) || 0,
      method: methodSel.value,
    });
  });

  function fmtDuration(seconds) {
    const s = Math.max(0, Math.round(seconds));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return m + "m " + r + "s";
  }

  // ---- comparison player: preloads each sampled frame's uncorrected/
  // corrected PNGs as Image objects from blob URLs (same blobUrl() helper
  // used elsewhere), then a scrubber + optional setInterval-driven autoplay
  // redraws the two side-by-side canvases per tick -- no new dependency, same
  // PNG-over-comm-buffer delivery mechanism already in use.
  let player = [];
  let playTimer = null;

  // loadImage() creates one object URL per PNG (via blobUrl()) that stays
  // alive until explicitly revoked -- each frame keeps its uncorrImgUrl/
  // corrImgUrl alongside the decoded Image so revokePlayerUrls() can release
  // them once a player array is no longer displayed (superseded by a new
  // verify_result, or the widget itself is torn down).
  function loadImage(buf) {
    return new Promise((resolve) => {
      if (!buf) { resolve({ img: null, url: null }); return; }
      const url = blobUrl(buf);
      const img = new Image();
      img.onload = () => resolve({ img, url });
      img.onerror = () => resolve({ img: null, url });
      img.src = url;
    });
  }

  function revokePlayerUrls(frames) {
    (frames || []).forEach((f) => {
      if (f.uncorrImgUrl) URL.revokeObjectURL(f.uncorrImgUrl);
      if (f.corrImgUrl) URL.revokeObjectURL(f.corrImgUrl);
    });
  }

  async function buildPlayer(msg, buffers) {
    const frameIndices = msg.frame_indices || [];
    const hasCorrection = msg.has_correction || [];
    const bufs = buffers || [];
    let cursor = 1; // bufs[0] is the reference frame, not part of the player
    const frames = [];
    for (let i = 0; i < frameIndices.length; i++) {
      const uncorr = await loadImage(bufs[cursor++]);
      let corr = { img: null, url: null };
      if (hasCorrection[i]) corr = await loadImage(bufs[cursor++]);
      frames.push({
        frameIndex: frameIndices[i],
        uncorrImg: uncorr.img,
        uncorrImgUrl: uncorr.url,
        corrImg: corr.img,
        corrImgUrl: corr.url,
        hasCorrection: !!hasCorrection[i],
      });
    }
    return frames;
  }

  function drawHalf(canvas, img) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (img) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  }

  function drawPlayerFrame(idx) {
    const f = player[idx];
    if (!f) return;
    drawHalf(playerBeforeCanvas, f.uncorrImg);
    drawHalf(playerAfterCanvas, f.hasCorrection ? f.corrImg : f.uncorrImg);
    playerLabel.textContent = "frame " + f.frameIndex + " (" + (idx + 1) + "/" + player.length + ")" +
      (f.hasCorrection ? "" : " -- no correction available");
  }

  function stopPlayback() {
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    playBtn.textContent = "Play";
  }
  function startPlayback() {
    if (player.length < 2) return;
    playBtn.textContent = "Pause";
    playTimer = setInterval(() => {
      let idx = (parseInt(scrubber.value, 10) || 0) + 1;
      if (idx >= player.length) idx = 0;
      scrubber.value = String(idx);
      drawPlayerFrame(idx);
    }, 500);
  }
  playBtn.addEventListener("click", () => {
    if (playTimer) stopPlayback(); else startPlayback();
  });
  scrubber.addEventListener("input", () => {
    stopPlayback();
    drawPlayerFrame(parseInt(scrubber.value, 10) || 0);
  });

  function drawTrajectory(frameIndices, transforms) {
    const ctx = trajCanvas.getContext("2d");
    ctx.clearRect(0, 0, trajCanvas.width, trajCanvas.height);
    const n = frameIndices.length;
    if (n === 0) return;
    const dx = transforms.map((t) => (t ? t.dx : null));
    const dy = transforms.map((t) => (t ? t.dy : null));
    const theta = transforms.map((t) => (t ? t.theta : null));
    const nums = [].concat(dx, dy).filter((v) => v !== null && v !== undefined);
    const maxAbs = Math.max(1, ...nums.map((v) => Math.abs(v)));
    const midY = trajCanvas.height / 2;
    const scaleY = (trajCanvas.height / 2 - 10) / maxAbs;
    const stepX = n > 1 ? (trajCanvas.width - 20) / (n - 1) : 0;
    function plot(series, color) {
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.5;
      let started = false, prevX = 0, prevY = 0;
      series.forEach((v, i) => {
        const x = 10 + i * stepX;
        if (v === null || v === undefined) { started = false; return; }
        const y = midY - v * scaleY;
        ctx.beginPath();
        if (started) ctx.moveTo(prevX, prevY);
        else ctx.moveTo(x, y);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath(); ctx.arc(x, y, 2, 0, 2 * Math.PI); ctx.fill();
        started = true; prevX = x; prevY = y;
      });
    }
    ctx.strokeStyle = "#ccc"; ctx.beginPath();
    ctx.moveTo(0, midY); ctx.lineTo(trajCanvas.width, midY); ctx.stroke();
    plot(dx, "#2b7fd6");
    plot(dy, "#d24d92");
    plot(theta, "#25a06f");
    ctx.fillStyle = "#555"; ctx.font = "10px sans-serif";
    ctx.fillText("dx (blue)   dy (pink)   theta (green)", 10, 12);
  }

  model.on("msg:custom", (msg, buffers) => {
    if (!msg) return;
    if (msg.type === "verify_result") {
      verifyOut.style.display = "block";
      drawTrajectory(msg.frame_indices, msg.transforms);
      stopPlayback();
      buildPlayer(msg, buffers).then((frames) => {
        revokePlayerUrls(player);
        player = frames;
        scrubber.max = String(Math.max(0, frames.length - 1));
        scrubber.value = "0";
        drawPlayerFrame(0);
      });
    } else if (msg.type === "progress") {
      progWrap.style.display = "block";
      if (msg.phase === "verify") {
        const frac = msg.num_frames ? (msg.frame + 1) / msg.num_frames : 0;
        progBar.style.width = Math.min(100, Math.round(frac * 100)) + "%";
        progLabel.textContent = "verify: frame " + (msg.frame + 1) + "/" + msg.num_frames;
      } else {
        const frac = msg.num_frames ? msg.frame / msg.num_frames : 0;
        const posFrac = msg.num_positions ? (msg.position + frac) / msg.num_positions : 0;
        progBar.style.width = Math.min(100, Math.round(posFrac * 100)) + "%";
        let label = "position " + msg.position + "/" + msg.num_positions +
          " · frame " + msg.frame + "/" + msg.num_frames;
        if (msg.elapsed_seconds !== undefined && msg.elapsed_seconds !== null) {
          label += " · elapsed " + fmtDuration(msg.elapsed_seconds);
          if (msg.eta_seconds !== undefined && msg.eta_seconds !== null) {
            label += " · ETA ~" + fmtDuration(msg.eta_seconds);
          }
        }
        progLabel.textContent = label;
      }
    } else if (msg.type === "batch_done") {
      progBar.style.width = "100%";
      const nFailed = msg.failed_positions ? msg.failed_positions.length : 0;
      progLabel.textContent = "done: " + msg.completed.length + " completed, " +
        msg.skipped.length + " skipped, " + nFailed + " failed -- saved to " + msg.path;
      showStatus("");
    } else if (msg.type === "saved") {
      showStatus("saved " + msg.path);
    } else if (msg.type === "error") {
      showStatus("Error (" + msg.kind + "): " + msg.message, true);
    }
  });

  // ---- batch-apply ----
  $(".rd-batch-btn").addEventListener("click", () => {
    showStatus("");
    progWrap.style.display = "block";
    progBar.style.width = "0%";
    progLabel.textContent = "starting...";
    model.send({ type: "batch_apply", directory: dirInput.value || null });
  });
  $(".rd-save-btn").addEventListener("click", () => {
    model.send({ type: "save" });
  });

  if (methodSel.value === "MaskedTemplateCorrelation") {
    layoutMask();
    const initB64 = model.get("mask_image_b64");
    if (initB64) { maskImg.src = initB64; } else { requestMaskFrame(); }
  }
  drawMask();

  return () => {
    stopPlayback();
    revokePlayerUrls(player);
    maskCanvas.removeEventListener("pointerdown", onMaskDown);
    maskCanvas.removeEventListener("pointermove", onMaskMove);
    maskCanvas.removeEventListener("pointerup", onMaskUp);
    maskCanvas.removeEventListener("pointercancel", onMaskUp);
    model.off("change:mask_image_b64", onMaskImageChange);
    model.off(
      "change:mask_center_x change:mask_center_y change:mask_width change:mask_height " +
        "change:mask_angle change:mask_points",
      onMaskGeomChange,
    );
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

    class SequenceDashboard(anywidget.AnyWidget):  # type: ignore[no-redef]
        """Curate positions + ROIs across a multi-position acquisition (anywidget).

        A three-pane UI (position gallery / ROI editor / selection list) over a
        :class:`~acia.segm.open.SequenceFile`. Browse positions, mark ROIs (draw or
        point-fit), and emit a :class:`~acia.selection.SelectionManifest`. Frames
        are read lazily from the source and pushed to the browser as PNG bytes; the
        widget never loads the whole (possibly hundreds-of-GB) file.

        Works in Jupyter/Colab and in marimo via ``mo.ui.anywidget(dash)``. The ESM
        is best-effort and verified only by a real notebook run (or the Playwright
        suite in the devcontainer), not by the headless Python test-suite.
        """

        metadata = traitlets.Dict().tag(sync=True)  # type: ignore[var-annotated]
        positions = traitlets.List().tag(sync=True)  # type: ignore[var-annotated]
        selections = traitlets.List().tag(sync=True)  # type: ignore[var-annotated]
        roi_mode = traitlets.Unicode("single").tag(sync=True)
        view_size = traitlets.Int(430).tag(sync=True)

        _esm = _SEQUENCE_DASHBOARD_ESM
        _css = _SEQUENCE_DASHBOARD_CSS

        def __init__(self, source, *, roi_mode: str = "single", **kwargs) -> None:
            """Build the dashboard from a source (no pixel reads at construction).

            Args:
                source: A :class:`~acia.segm.open.SequenceFile`, or a path/str that
                    is opened via :func:`~acia.segm.open.open_sequence`.
                roi_mode: ``"single"`` (<=1 ROI/position) or ``"multi"``.
                **kwargs: Forwarded to ``anywidget.AnyWidget``.
            """
            from acia.segm.open import open_sequence

            if isinstance(source, (str, os.PathLike)):
                source = open_sequence(source)
            self._file = source

            meta = source.metadata
            positions = [
                {"index": p.index, "name": p.name, "has_roi": False}
                for p in source.positions
            ]
            super().__init__(
                metadata=meta.to_dict(),
                positions=positions,
                selections=[],
                roi_mode=roi_mode,
                **kwargs,
            )
            self.on_msg(self._on_custom_msg)

        def _on_custom_msg(self, _widget, content, buffers) -> None:
            """Serve lazy frames/thumbnails and run point-fit for the ESM.

            Named to avoid colliding with ``ipywidgets.Widget._handle_msg``,
            the internal method the base class uses to dispatch comm
            messages to callbacks registered via ``on_msg`` -- reusing that
            name here silently shadowed the real dispatcher and broke every
            custom message (frame/thumb/fit/save) for this widget.
            """
            kind = content.get("type") if isinstance(content, dict) else None
            # Diagnostic timing (temporary): a "fit" request appearing to hang
            # has been reported repeatedly against real SMB data; these lines
            # print to the kernel's terminal so it's visible whether a given
            # message is received promptly and how long its handler took --
            # the two things needed to tell "queued behind a slow read" apart
            # from "stuck inside the handler itself".
            t0 = time.time()
            logging.warning(
                "SequenceDashboard: received %r at %.3f (content=%s)",
                kind,
                t0,
                {k: v for k, v in content.items() if k != "points"}
                if isinstance(content, dict)
                else content,
            )
            if kind == "thumb":
                pos = int(content["pos"])
                try:
                    png = self._file.thumbnail_png(
                        pos, downscale=int(content.get("downscale", 8))
                    )
                except Exception as exc:  # noqa: BLE001 - report to the frontend
                    self.send(
                        {"type": "error", "kind": kind, "pos": pos, "message": str(exc)}
                    )
                    return
                self.send({"type": "thumb", "pos": pos}, buffers=[png])
                logging.warning(
                    "SequenceDashboard: thumb %d done in %.3fs", pos, time.time() - t0
                )
            elif kind == "frame":
                pos, t = int(content["pos"]), int(content.get("t", 0))
                try:
                    png = self._frame_png(pos, t)
                except Exception as exc:  # noqa: BLE001 - report to the frontend
                    self.send(
                        {"type": "error", "kind": kind, "pos": pos, "message": str(exc)}
                    )
                    return
                self.send({"type": "frame", "pos": pos, "t": t}, buffers=[png])
                logging.warning(
                    "SequenceDashboard: frame %d/%d done in %.3fs",
                    pos,
                    t,
                    time.time() - t0,
                )
            elif kind == "fit":
                try:
                    spec = _fit_rotated_rect(content["points"])
                except Exception as exc:  # noqa: BLE001 - report to the frontend
                    # too-few / degenerate points, or anything else -- the UI
                    # always sends exactly 4 points, so any failure here is
                    # unexpected and must be visible, never swallowed silently.
                    self.send({"type": "error", "kind": "fit", "message": str(exc)})
                    return
                self.send({"type": "fit", "roi": spec.to_dict()})
                logging.warning(
                    "SequenceDashboard: fit done in %.3fs", time.time() - t0
                )
            elif kind == "save":
                try:
                    path = self.save()
                except Exception as exc:  # noqa: BLE001 - report to the frontend
                    self.send({"type": "error", "kind": "save", "message": str(exc)})
                    return
                self.send({"type": "saved", "path": path})
                logging.warning(
                    "SequenceDashboard: save done in %.3fs", time.time() - t0
                )

        def _frame_png(self, pos: int, t: int) -> bytes:
            """PNG bytes of one lazily-read frame (display channel, normalized)."""
            raw = np.asarray(self._file.position(pos).get_frame(t).raw)
            plane = raw[..., 0] if raw.ndim == 3 else raw
            rgb = np.repeat(normalize_to_uint8(plane)[:, :, np.newaxis], 3, axis=-1)
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="PNG")
            return buffer.getvalue()

        @property
        def manifest(self):
            """Build a :class:`~acia.selection.SelectionManifest` from current state."""
            from acia.base import RotatedCropSpec
            from acia.selection import (
                RoiSelection,
                SelectionManifest,
                make_source_block,
            )

            sels = []
            for item in self.selections:
                roi = item["roi"]
                spec = RotatedCropSpec(
                    center=(float(roi["center"][0]), float(roi["center"][1])),
                    size=(int(roi["size"][0]), int(roi["size"][1])),
                    angle=float(roi["angle"]),
                )
                sels.append(
                    RoiSelection(
                        position=int(item["position"]),
                        roi=spec,
                        label=item.get("label", ""),
                        id=str(item.get("id", "")),
                    )
                )
            return SelectionManifest(
                source=make_source_block(self._file),
                selections=sels,
                roi_mode=self.roi_mode,
            )

        def save(self, directory=None) -> str:
            """Write ``selection.json`` (+ previews) via :func:`save_selection`.

            Args:
                directory: Output dir; defaults to the current working directory
                    (the notebook's dir at run time).

            Returns:
                The path to the written ``selection.json``.
            """
            from acia.selection import save_selection

            directory = os.getcwd() if directory is None else directory
            return save_selection(self.manifest, directory)

        @classmethod
        def resume(cls, manifest_or_path, source=None, **kwargs) -> SequenceDashboard:
            """Reopen a dashboard pre-populated from a saved ``selection.json``.

            Lets a curation session be saved with :meth:`save` and continued
            later in a fresh dashboard, instead of starting over.

            Args:
                manifest_or_path: A path to a ``selection.json`` file (or its
                    containing directory), or an already-loaded
                    :class:`~acia.selection.SelectionManifest`.
                source: ``None`` to reopen the manifest's own source path, a
                    path/str to apply the selections to a *different* file, or
                    an already-open :class:`~acia.segm.open.SequenceFile` --
                    same convention as :func:`~acia.selection.load_selection`.
                **kwargs: Forwarded to the constructor (e.g. ``roi_mode`` to
                    override the manifest's saved mode).

            Returns:
                A :class:`SequenceDashboard` with ``.selections`` restored.

            Raises:
                ValueError: If a selection's position is out of range for
                    ``source``.
            """
            from acia.segm.open import open_sequence
            from acia.selection import SelectionManifest

            if isinstance(manifest_or_path, SelectionManifest):
                manifest = manifest_or_path
            else:
                path = os.fspath(manifest_or_path)
                if os.path.isdir(path):
                    path = os.path.join(path, "selection.json")
                manifest = SelectionManifest.load(path)

            if source is None:
                source = open_sequence(manifest.source_path)
            elif isinstance(source, (str, os.PathLike)):
                source = open_sequence(source)

            kwargs.setdefault("roi_mode", manifest.roi_mode)
            dash = cls(source, **kwargs)

            num_positions = dash.metadata.get("num_positions", 0)
            next_ci: dict[int, int] = {}
            restored = []
            for i, sel in enumerate(manifest.selections):
                if not 0 <= sel.position < num_positions:
                    raise ValueError(
                        f"selection position {sel.position} out of range "
                        f"for source with {num_positions} positions"
                    )
                ci = next_ci.get(sel.position, 0)
                next_ci[sel.position] = ci + 1
                restored.append(
                    {
                        "id": i + 1,
                        "position": sel.position,
                        "label": sel.label,
                        "ci": ci,
                        "roi": sel.roi.to_dict(),
                    }
                )
            dash.selections = restored
            return dash

        def _repr_html_(self) -> str:
            """Static fallback for a non-executed/persisted notebook."""
            n = self.metadata.get("num_positions", "?")
            return (
                "<div style='font:12px sans-serif;color:#666;'>"
                f"SequenceDashboard — {n} positions "
                "(interactive widget renders when the notebook is run).</div>"
            )

    class RegistrationDashboard(anywidget.AnyWidget):  # type: ignore[no-redef]
        """Pick + verify + batch-apply a drift-correction method (anywidget).

        Pick one of the 5 :class:`~acia.registration.RegistrationMethod`
        implementations (default ``"GradientECC"``), verify it on sampled
        frames of a single position (drift trajectory + before/after), then
        batch-apply it across every position/frame of the acquisition with
        live progress and resumability. Frames are read lazily from the
        source; the widget never loads a whole (possibly hundreds-of-GB) file,
        and batch-apply holds at most one position's frames in memory at a
        time.

        ``MaskedTemplateCorrelation`` additionally needs a ``mask_rect``
        (:class:`~acia.base.RotatedCropSpec`): the ``mask_*`` traits and the
        ESM's mask editor port :class:`ROICropper`'s click-to-fit +
        drag/resize/rotate interaction model (:func:`_fit_rotated_rect` is the
        same geometry helper ``SequenceDashboard``'s point-fit tool uses) --
        ``ROICropper`` itself is not touched.

        Works in Jupyter/Colab and in marimo via ``mo.ui.anywidget(dash)``. The
        ESM is best-effort and verified only by a real notebook run, not by
        the headless Python test-suite (no ESM/Playwright suite for this
        widget in v1, per the spec).
        """

        metadata = traitlets.Dict().tag(sync=True)  # type: ignore[var-annotated]
        positions = traitlets.List().tag(sync=True)  # type: ignore[var-annotated]
        method_name = traitlets.Unicode("GradientECC").tag(sync=True)
        n_sample_frames = traitlets.Int(8).tag(sync=True)

        mask_center_x = traitlets.Float(0.0).tag(sync=True)
        mask_center_y = traitlets.Float(0.0).tag(sync=True)
        mask_width = traitlets.Int(0).tag(sync=True)
        mask_height = traitlets.Int(0).tag(sync=True)
        mask_angle = traitlets.Float(0.0).tag(sync=True)
        mask_points = traitlets.List().tag(sync=True)  # type: ignore[var-annotated]
        mask_image_b64 = traitlets.Unicode("").tag(sync=True)
        mask_image_w = traitlets.Int(0).tag(sync=True)
        mask_image_h = traitlets.Int(0).tag(sync=True)

        batch_running = traitlets.Bool(False).tag(sync=True)

        _esm = _REGISTRATION_DASHBOARD_ESM

        def __init__(
            self, source, *, method_name: str = "GradientECC", **kwargs
        ) -> None:
            """Build the dashboard from a source (no pixel reads at construction).

            Args:
                source: A :class:`~acia.segm.open.SequenceFile`, or a path/str
                    that is opened via :func:`~acia.segm.open.open_sequence`.
                method_name: The initially-selected
                    :class:`~acia.registration.RegistrationMethod` name; one of
                    :data:`_REGISTRATION_METHOD_NAMES`.
                **kwargs: Forwarded to ``anywidget.AnyWidget``.
            """
            from acia.segm.open import open_sequence

            if isinstance(source, (str, os.PathLike)):
                source = open_sequence(source)
            self._file = source
            self._records: dict[int, RegistrationRecord] = {}

            meta = source.metadata
            positions = [{"index": p.index, "name": p.name} for p in source.positions]
            super().__init__(
                metadata=meta.to_dict(),
                positions=positions,
                method_name=method_name,
                **kwargs,
            )
            self.on_msg(self._on_custom_msg)

        @traitlets.validate("method_name")
        def _validate_method_name(self, proposal):
            value = proposal["value"]
            if value not in _REGISTRATION_METHOD_NAMES:
                raise traitlets.TraitError(
                    f"method_name must be one of {_REGISTRATION_METHOD_NAMES}, "
                    f"got {value!r}"
                )
            return value

        @traitlets.validate("n_sample_frames")
        def _validate_n_sample_frames(self, proposal):
            value = int(proposal["value"])
            if value < 1:
                raise traitlets.TraitError(
                    f"n_sample_frames must be >= 1, got {value}."
                )
            return value

        @traitlets.observe("mask_points")
        def _on_mask_points(self, change) -> None:
            """Re-fit the mask box whenever >=3 points are present.

            Mirrors :meth:`ROICropper._on_points` exactly (same
            :func:`_fit_rotated_rect` geometry helper); degenerate/too-few-point
            states during interactive clicking are ignored.
            """
            import cv2

            pts = change.get("new") if isinstance(change, dict) else change.new
            try:
                if pts is not None and len(pts) >= 3:
                    spec = _fit_rotated_rect(pts)
                    self.mask_center_x, self.mask_center_y = spec.center
                    self.mask_width, self.mask_height = spec.size
                    self.mask_angle = spec.angle
            except (ValueError, cv2.error):
                # Only swallow the expected degenerate/too-few-point states;
                # a real bug such as an ImportError must not be silently
                # dropped.
                pass

        def _on_custom_msg(self, _widget, content, buffers) -> None:
            """Serve the mask frame, run verify, and drive batch-apply/save.

            Named to avoid colliding with ``ipywidgets.Widget._handle_msg``,
            same reasoning as ``SequenceDashboard._on_custom_msg``.
            """
            kind = content.get("type") if isinstance(content, dict) else None
            if kind == "mask_frame":
                pos = int(content.get("position", 0))
                try:
                    data_url, w, h = _encode_frame_png(
                        self._file.position(pos), frame=0
                    )
                except Exception as exc:  # noqa: BLE001 - report to the frontend
                    self.send({"type": "error", "kind": kind, "message": str(exc)})
                    return
                self.mask_image_b64 = data_url
                self.mask_image_w = w
                self.mask_image_h = h
                self.send({"type": "mask_frame", "position": pos})
            elif kind == "verify":
                pos = int(content.get("position", 0))
                method_name = str(content.get("method", self.method_name))
                try:
                    payload, verify_buffers = self._run_verify(pos, method_name)
                except Exception as exc:  # noqa: BLE001 - report to the frontend
                    self.send({"type": "error", "kind": kind, "message": str(exc)})
                    return
                self.send(payload, buffers=verify_buffers)
            elif kind == "batch_apply":
                directory = content.get("directory") or None
                subset = content.get("positions")
                try:
                    summary = self.batch_apply(directory=directory, positions=subset)
                except Exception as exc:  # noqa: BLE001 - report to the frontend
                    self.send({"type": "error", "kind": kind, "message": str(exc)})
                    return
                self.send({"type": "batch_done", **summary})
            elif kind == "save":
                try:
                    path = self.save()
                except Exception as exc:  # noqa: BLE001 - report to the frontend
                    self.send({"type": "error", "kind": kind, "message": str(exc)})
                    return
                self.send({"type": "saved", "path": path})

        def _build_method(
            self, method_name: str, mask_rect: RotatedCropSpec | None = None
        ):
            """Construct a fresh :class:`~acia.registration.RegistrationMethod`.

            Args:
                method_name: One of :data:`_REGISTRATION_METHOD_NAMES`.
                mask_rect: The mask rect to use for ``MaskedTemplateCorrelation``;
                    defaults to :attr:`mask_rect` when ``None``. Ignored for the
                    other 4 methods, which run directly on raw frame pairs.

            Raises:
                ValueError: If ``method_name`` is unknown, or if
                    ``MaskedTemplateCorrelation`` is requested without a mask
                    rect available.
            """
            classes = _registration_method_classes()
            cls = classes[method_name]
            if method_name == "MaskedTemplateCorrelation":
                rect = mask_rect if mask_rect is not None else self.mask_rect
                if rect is None:
                    raise ValueError(
                        "MaskedTemplateCorrelation requires a mask rect -- draw "
                        "one (click >=3 points on the mask editor) first."
                    )
                return cls(mask_rect=rect)
            return cls()

        @property
        def mask_rect(self) -> RotatedCropSpec | None:
            """The current mask rect, or ``None`` if none has been drawn yet."""
            from acia.base import RotatedCropSpec

            if self.mask_width <= 0 or self.mask_height <= 0:
                return None
            return RotatedCropSpec(
                center=(self.mask_center_x, self.mask_center_y),
                size=(int(self.mask_width), int(self.mask_height)),
                angle=self.mask_angle,
            )

        @staticmethod
        def _array_png_bytes(raw: np.ndarray) -> bytes:
            """PNG-encode a single in-memory frame array (channel 0, normalized).

            Sibling of ``SequenceDashboard._frame_png``, but for a frame that
            already lives in memory (e.g. an :func:`~acia.registration.apply_correction`
            result) rather than one read fresh from a source -- ``_encode_frame_png``
            only accepts the latter.
            """
            plane = raw[..., 0] if raw.ndim == 3 else raw
            rgb = np.repeat(normalize_to_uint8(plane)[:, :, np.newaxis], 3, axis=-1)
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="PNG")
            return buffer.getvalue()

        def _run_verify(self, pos: int, method_name: str):
            """Run verify for one position: drift trajectory + full-range before/after.

            Sends a ``"progress"`` message (``phase="verify"``) via
            :meth:`send` after each sampled frame is compared (wired through
            :func:`~acia.registration.run_comparison`'s ``on_progress``
            callback), so the widget shows visible progress while verify
            runs -- previously this method computed silently.

            Every sampled ``frame_indices`` entry gets an uncorrected PNG
            buffer, plus a corrected one when a transform estimate is
            available for it (mirrors, per-frame, what the single compare
            frame used to do) -- the ESM's comparison player cycles through
            all of them instead of showing one static toggle image.

            Returns:
                tuple[dict, list[bytes]]: The ``"verify_result"`` message
                content and its PNG buffers: ``[reference, uncorrected_0,
                (corrected_0)?, uncorrected_1, (corrected_1)?, ...]`` -- one
                uncorrected buffer per sampled frame, plus a corrected buffer
                only where ``has_correction[i]`` is true, so the ESM can walk
                the flat buffer array in lock-step with ``frame_indices``/
                ``has_correction``.
            """
            from acia.registration import (
                apply_correction,
                build_sample_frame_indices,
                run_comparison,
            )

            method = self._build_method(method_name)
            source = self._file.position(pos)
            frame_indices = build_sample_frame_indices(
                source.size_t, 0, self.n_sample_frames
            )
            reference = np.asarray(source.get_frame(0).raw)

            # Cache each comparison frame as it's read during run_comparison
            # so the buffer-encoding pass below doesn't re-read it from the
            # (possibly slow) source a second time.
            frame_cache: dict[int, np.ndarray] = {}

            def get_frame(t: int) -> np.ndarray:
                frame = np.asarray(source.get_frame(t).raw)
                frame_cache[t] = frame
                return frame

            total = len(frame_indices)

            def on_progress(i: int, _total: int) -> None:
                self.send(
                    {
                        "type": "progress",
                        "phase": "verify",
                        "frame": i,
                        "num_frames": total,
                    }
                )

            results = run_comparison(
                {method_name: method},
                reference,
                get_frame,
                frame_indices,
                on_progress=on_progress,
            )
            transforms = results[method_name]

            buffers = [self._array_png_bytes(reference)]
            has_correction: list[bool] = []
            for t, transform in zip(frame_indices, transforms, strict=True):
                frame = frame_cache[t]
                buffers.append(self._array_png_bytes(frame))
                available = transform is not None
                has_correction.append(available)
                if transform is not None:
                    corrected = apply_correction(frame, transform)
                    buffers.append(self._array_png_bytes(corrected))

            payload = {
                "type": "verify_result",
                "position": pos,
                "method": method_name,
                "reference_frame": 0,
                "frame_indices": frame_indices,
                "transforms": [t.to_dict() if t else None for t in transforms],
                "has_correction": has_correction,
            }
            return payload, buffers

        def _register_position(
            self,
            pos: int,
            method_name: str,
            mask_rect: RotatedCropSpec | None,
            num_positions: int,
            *,
            source=None,
            existing_record: RegistrationRecord | None = None,
            positions_remaining_after: int = 0,
            progress_state: dict | None = None,
            on_checkpoint: Callable[[RegistrationRecord], None] | None = None,
        ) -> RegistrationRecord:
            """Estimate a per-frame transform for every not-yet-computed frame.

            Resumable: frames are always processed in order (``0, 1, 2, ...``)
            and checkpointed periodically, so "how many frames are already in
            ``existing_record``" is always the index of the first uncomputed
            frame -- ``existing_record`` (when given) seeds ``transforms``/
            ``failed_frames`` and estimation resumes right after it instead of
            redoing the whole position.

            Reads (and releases) exactly one frame at a time -- never more than
            one position's frames in memory at once. A per-frame failure is
            caught and recorded in ``failed_frames``; it never aborts the rest
            of the position. Sends a ``"progress"`` message (with best-effort
            ``elapsed_seconds``/``eta_seconds``, see :func:`_estimate_eta`)
            after every frame, and invokes ``on_checkpoint`` with the
            record-so-far every :data:`CHECKPOINT_INTERVAL` newly-estimated
            frames so an interrupted run loses at most that many.

            Args:
                pos: Position index to register.
                method_name: One of :data:`_REGISTRATION_METHOD_NAMES`.
                mask_rect: Mask rect for ``MaskedTemplateCorrelation``; ignored
                    otherwise.
                num_positions: Total position count, forwarded into progress
                    messages unchanged.
                source: Optional :class:`~acia.base.ImageSequenceSource` to
                    register instead of ``self._file.position(pos)`` -- e.g. a
                    lazily-sliced ``self._file.position(pos)[:30]`` to limit
                    registration to the first 30 frames. Its own ``size_t``
                    (not the full position's) drives the frame loop, so a
                    resumed record's "already done" count is compared against
                    the *sliced* length. ``None`` (default) reproduces the
                    prior always-whole-position behavior.
                existing_record: A partial (or empty) prior result to resume
                    from; ``None`` is equivalent to a from-scratch position.
                    Assumed to already be for ``method_name`` -- callers
                    (e.g. :meth:`batch_apply`) are responsible for not
                    passing a record recorded under a different method.
                positions_remaining_after: Positions still to process after
                    this one in the current batch-apply run (for the ETA
                    heuristic).
                progress_state: Mutable dict shared across the whole
                    batch-apply run (``batch_start``, ``frames_done``,
                    ``position_frame_counts``) driving :func:`_estimate_eta`;
                    a fresh one is created if ``None`` (single-position use).
                on_checkpoint: Optional callback invoked with the
                    record-so-far every ``CHECKPOINT_INTERVAL`` frames.
            """
            from acia.registration_persistence import RegistrationRecord

            method = self._build_method(method_name, mask_rect)
            source = source if source is not None else self._file.position(pos)
            num_frames = source.size_t
            reference = np.asarray(source.get_frame(0).raw)

            transforms: dict[int, FrameTransform] = (
                dict(existing_record.transforms) if existing_record else {}
            )
            failed: dict[int, str] = (
                dict(existing_record.failed_frames) if existing_record else {}
            )
            start_frame = len(transforms) + len(failed)

            state = (
                progress_state
                if progress_state is not None
                else {"batch_start": time.monotonic(), "position_frame_counts": []}
            )
            state.setdefault("frames_done", 0)

            since_checkpoint = 0
            for t in range(start_frame, num_frames):
                try:
                    frame = np.asarray(source.get_frame(t).raw)
                    transforms[t] = method.estimate(reference, frame)
                except Exception as exc:  # noqa: BLE001 -- isolate per-frame failures
                    failed[t] = f"{type(exc).__name__}: {exc}"

                state["frames_done"] += 1
                since_checkpoint += 1

                if (
                    on_checkpoint is not None
                    and since_checkpoint >= CHECKPOINT_INTERVAL
                ):
                    on_checkpoint(
                        RegistrationRecord(
                            position=pos,
                            method=method_name,
                            transforms=dict(transforms),
                            reference_frame=0,
                            failed_frames=dict(failed),
                        )
                    )
                    since_checkpoint = 0

                elapsed = time.monotonic() - state["batch_start"]
                eta = _estimate_eta(
                    elapsed=elapsed,
                    frames_done=state["frames_done"],
                    frames_left_in_position=num_frames - (t + 1),
                    positions_remaining_after=positions_remaining_after,
                    position_frame_counts=state["position_frame_counts"],
                    current_position_num_frames=num_frames,
                )
                self.send(
                    {
                        "type": "progress",
                        "position": pos,
                        "num_positions": num_positions,
                        "frame": t,
                        "num_frames": num_frames,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                    }
                )

            state["position_frame_counts"].append(num_frames)

            return RegistrationRecord(
                position=pos,
                method=method_name,
                transforms=transforms,
                reference_frame=0,
                failed_frames=failed,
            )

        def batch_apply(self, directory=None, positions=None, sources=None) -> dict:
            """Estimate transforms for every (or a subset of) position, live.

            For the currently-selected :attr:`method_name`, processes every
            position in ``positions`` (default: all), one at a time, estimating
            a :class:`~acia.registration.FrameTransform` per frame against that
            position's own frame 0. Sends a ``"progress"`` message (with
            best-effort ``elapsed_seconds``/``eta_seconds``) after every frame,
            and persists the manifest both after every position *and*
            periodically within a position (every :data:`CHECKPOINT_INTERVAL`
            newly-estimated frames), so an interrupted run can be resumed. A
            position already fully complete (every frame accounted for in
            ``transforms``/``failed_frames``) is skipped; a partial one
            resumes from its first uncomputed frame instead of being
            re-skipped or fully redone. A prior record recorded under a
            *different* ``method_name`` is never treated as resume/skip data
            for the currently-selected method -- the position is processed
            from scratch instead of silently merging frames across methods.
            A whole-position failure (whether from ``_register_position``
            itself or from the resume/skip bookkeeping above it, e.g. a
            ``size_t`` lookup) never aborts the rest of the run, and never
            discards progress already checkpointed for that position in this
            or a prior run -- the failure is recorded as a note on top of
            whatever record (checkpointed or pre-existing) is already known,
            not as a fresh empty one.

            Args:
                directory: Output directory for ``registration_transforms.json``;
                    defaults to the current working directory. Also the path
                    consulted for already-completed positions to skip.
                positions: Optional subset of position indices to process;
                    defaults to every position in the acquisition.
                sources: Optional ``{position: ImageSequenceSource}`` override --
                    when a position has an entry, that source is registered
                    instead of ``self._file.position(position)``, and its own
                    ``size_t`` (not the full position's) is what "already
                    complete" is checked against. Lets a caller limit
                    registration to a sub-range via the lazy numpy-style
                    indexing every ``ImageSequenceSource`` already supports,
                    e.g. ``sources={2: seqfile.position(2)[:30]}`` to register
                    only the first 30 frames of position 2. A position absent
                    from ``sources`` (or when ``sources`` is ``None``) falls
                    back to the whole position, unchanged from before.

            Returns:
                dict: ``{"num_positions", "completed", "skipped",
                "failed_positions", "path"}``.

            Raises:
                RuntimeError: If a batch-apply run is already in progress.
                ValueError: If ``method_name`` is unknown, or
                    ``MaskedTemplateCorrelation`` is selected without a mask
                    rect.
            """
            if self.batch_running:
                raise RuntimeError("batch-apply is already running")
            method_name = self.method_name
            if method_name not in _REGISTRATION_METHOD_NAMES:
                raise ValueError(f"unknown method_name {method_name!r}")

            mask_rect = None
            if method_name == "MaskedTemplateCorrelation":
                mask_rect = self.mask_rect
                if mask_rect is None:
                    raise ValueError(
                        "MaskedTemplateCorrelation requires a mask rect -- draw "
                        "one (click >=3 points on the mask editor) before "
                        "running batch-apply."
                    )

            from acia.registration_persistence import (
                RegistrationManifest,
                save_registration,
            )

            directory = os.getcwd() if directory is None else os.fspath(directory)
            target_path = os.path.join(directory, "registration_transforms.json")

            records: dict[int, RegistrationRecord] = dict(self._records)
            if os.path.exists(target_path):
                try:
                    existing = RegistrationManifest.load(target_path)
                except Exception:  # noqa: BLE001 - a corrupt manifest must not block a run
                    existing = None
                if existing is not None:
                    for rec in existing.records:
                        records.setdefault(rec.position, rec)

            num_positions = int(self.metadata.get("num_positions", len(self.positions)))
            target_positions = (
                list(range(num_positions))
                if positions is None
                else [int(p) for p in positions]
            )

            self.batch_running = True
            completed: list[int] = []
            skipped: list[int] = []
            failed_positions: list[int] = []
            progress_state: dict = {
                "batch_start": time.monotonic(),
                "frames_done": 0,
                "position_frame_counts": [],
            }
            try:
                for idx, i in enumerate(target_positions):
                    pos_source = (sources or {}).get(i)
                    existing_record = records.get(i)
                    if (
                        existing_record is not None
                        and existing_record.method != method_name
                    ):
                        # Progress recorded under a different method is not valid
                        # resume/skip data for the currently-selected method --
                        # treat the position as if it had no prior record at all
                        # rather than silently merging frames across methods.
                        existing_record = None
                    if existing_record is not None:
                        # A failure looking up size_t must not abort the whole
                        # batch-apply run -- fall through to "not complete" so
                        # this position is attempted below, where the
                        # per-position try/except (which re-derives num_frames
                        # via _register_position) records it as a per-position
                        # failure instead.
                        try:
                            num_frames_i = (
                                pos_source.size_t
                                if pos_source is not None
                                else self._file.position(i).size_t
                            )
                            already_done = len(existing_record.transforms) + len(
                                existing_record.failed_frames
                            )
                            already_complete = (
                                num_frames_i > 0 and already_done >= num_frames_i
                            )
                        except Exception:  # noqa: BLE001 -- see comment above
                            already_complete = False
                        if already_complete:
                            skipped.append(i)
                            continue

                    positions_remaining_after = len(target_positions) - idx - 1

                    def _checkpoint(record: RegistrationRecord, _pos: int = i) -> None:
                        records[_pos] = record
                        self._records = records
                        save_registration(self.manifest, directory)

                    try:
                        record = self._register_position(
                            i,
                            method_name,
                            mask_rect,
                            num_positions,
                            source=pos_source,
                            existing_record=existing_record,
                            positions_remaining_after=positions_remaining_after,
                            progress_state=progress_state,
                            on_checkpoint=_checkpoint,
                        )
                    except Exception as exc:  # noqa: BLE001 -- isolate whole-position failures
                        from acia.registration_persistence import RegistrationRecord

                        note = f"position failed: {type(exc).__name__}: {exc}"
                        # `_register_position`'s on_checkpoint callback (above) may
                        # already have persisted a partial record into records[i]
                        # (and to disk) before this exception fired; records.get(i)
                        # also reflects the original existing_record when no
                        # checkpoint fired yet this run. Preserve whatever progress
                        # is already there instead of clobbering it with an empty
                        # record -- only synthesize an empty one when there's truly
                        # no prior record at all.
                        prior = records.get(i)
                        record = (
                            dataclasses.replace(prior, notes=note)
                            if prior is not None
                            else RegistrationRecord(
                                position=i,
                                method=method_name,
                                transforms={},
                                reference_frame=0,
                                notes=note,
                            )
                        )
                        failed_positions.append(i)
                    records[i] = record
                    completed.append(i)
                    self._records = records
                    save_registration(self.manifest, directory)
            finally:
                self.batch_running = False

            self._records = records
            saved_path = save_registration(self.manifest, directory)
            return {
                "num_positions": num_positions,
                "completed": completed,
                "skipped": skipped,
                "failed_positions": failed_positions,
                "path": saved_path,
            }

        @property
        def manifest(self) -> RegistrationManifest:
            """Build a :class:`~acia.registration_persistence.RegistrationManifest`
            from the accumulated :class:`~acia.registration_persistence.RegistrationRecord`
            results (mirrors ``SequenceDashboard.manifest`` building a
            ``SelectionManifest``).
            """
            from acia.registration_persistence import RegistrationManifest
            from acia.selection import make_source_block

            return RegistrationManifest(
                source=make_source_block(self._file),
                records=sorted(self._records.values(), key=lambda r: r.position),
                method=self.method_name,
            )

        def save(self, directory=None) -> str:
            """Write ``registration_transforms.json`` via :func:`save_registration`.

            Args:
                directory: Output dir; defaults to the current working
                    directory (the notebook's dir at run time).

            Returns:
                The path to the written ``registration_transforms.json``.
            """
            from acia.registration_persistence import save_registration

            directory = os.getcwd() if directory is None else directory
            return save_registration(self.manifest, directory)

        def _repr_html_(self) -> str:
            """Static fallback for a non-executed/persisted notebook."""
            n = self.metadata.get("num_positions", "?")
            return (
                "<div style='font:12px sans-serif;color:#666;'>"
                f"RegistrationDashboard — {n} positions, method="
                f"{self.method_name} (interactive widget renders when the "
                "notebook is run).</div>"
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

    def SequenceDashboard(*args, **kwargs):  # type: ignore[no-redef]
        """Stub raised when the optional ``widget`` extra is not installed."""
        raise ImportError(
            "SequenceDashboard requires the optional dependency: pip install acia[widget]"
        )

    def RegistrationDashboard(*args, **kwargs):  # type: ignore[no-redef]
        """Stub raised when the optional ``widget`` extra is not installed."""
        raise ImportError(
            "RegistrationDashboard requires the optional dependency: pip install acia[widget]"
        )
