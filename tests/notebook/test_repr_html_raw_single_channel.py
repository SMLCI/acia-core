"""Regression test: _repr_html_() must render a raw (H, W, 1) single-channel frame.

LocalSequenceSource(normalize_image=False), ND2SequenceSource, and
CZISequenceSource all represent a genuinely single-channel frame as (H, W, 1)
(not a bare (H, W) 2D array, and not artificially triplicated to 3 channels).
PIL's Image.fromarray has no mode for a trailing size-1 channel axis and raises
TypeError("Cannot handle this data type: (1, 1, 1), ...") -- render_image must
squeeze that axis before handing the array to PIL.
"""

import builtins
import unittest
from unittest.mock import patch

import numpy as np

from acia.notebook import JupyterVisualizationMixin


class _RawFrame:
    """Minimal frame stand-in exposing only .raw, like BaseImage subclasses."""

    def __init__(self, raw_data: np.ndarray):
        self.raw = raw_data


class _SingleChannelRawSource(JupyterVisualizationMixin):
    """A source whose frames are genuinely (H, W, 1), num_channels == 1."""

    def __init__(self, height=8, width=10):
        self.size_t = 1
        self.num_channels = 1
        self.overlay = None
        self._frame = _RawFrame(
            np.arange(height * width, dtype=np.uint16).reshape(height, width, 1)
        )

    def get_frame(self, frame_idx: int) -> _RawFrame:
        return self._frame


class TestReprHtmlSingleChannelRawFrame(unittest.TestCase):
    def test_repr_html_renders_without_error(self):
        source = _SingleChannelRawSource()

        # get_ipython() is normally injected as a builtin by IPython; outside a
        # kernel it's undefined, and _repr_html_ short-circuits to None before
        # ever reaching render_image -- inject a stand-in so the real render
        # path (and the bug it used to trip) actually runs under pytest.
        with (
            patch.object(builtins, "get_ipython", lambda: object(), create=True),
            patch("acia.notebook.logging.error") as mock_log_error,
        ):
            html = source._repr_html_()

        self.assertEqual(html, "")
        mock_log_error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
