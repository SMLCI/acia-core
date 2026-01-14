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

    # Type hints for expected interface (duck typing)
    size_t: int
    num_channels: int

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

        def normalize_to_uint8(image_array: np.ndarray) -> np.ndarray:
            """Normalize image array to uint8 [0, 255] range."""
            if image_array.dtype == np.uint8:
                return image_array

            # Normalize to 0-255 range
            min_val = np.min(image_array)
            max_val = np.max(image_array)

            if max_val > min_val:
                normalized = (
                    (image_array - min_val) / (max_val - min_val) * 255
                ).astype(np.uint8)
            else:
                normalized = np.zeros_like(image_array, dtype=np.uint8)

            return normalized

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
                        image_data = np.repeat(
                            image_data[:, :, np.newaxis], 3, axis=-1
                        )

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
