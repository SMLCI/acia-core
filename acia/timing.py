"""Helpers for physical calibration (time and pixel size) in pint units.

Sequences and overlays carry an optional physical calibration:

* ``timepoints`` -- a per-frame pint ``Quantity`` array (time of each frame),
  specified at load either explicitly or via a scalar ``frame_interval``;
* ``pixel_size`` -- a pint length per pixel (isotropic scalar, or a 2-element
  ``[y, x]`` Quantity for anisotropic pixels).

These helpers normalise user-supplied values into pint quantities.
"""

from __future__ import annotations

import numpy as np

from acia import Q_, ureg


def to_quantity(value):
    """Coerce a value to a pint ``Quantity`` (parse strings like ``"15 minute"``)."""
    if value is None:
        return None
    if isinstance(value, ureg.Quantity):
        return value
    return Q_(value)


def resolve_timepoints(n: int, *, timepoints=None, frame_interval=None):
    """Return a length-``n`` pint ``Quantity`` of per-frame timepoints, or ``None``.

    Args:
        n: number of frames.
        timepoints: explicit per-frame times as a pint ``Quantity`` array.
        frame_interval: scalar time between frames (pint ``Quantity`` or a string
            like ``"15 minute"``); yields ``arange(n) * interval``.
    """
    if timepoints is not None:
        if not isinstance(timepoints, ureg.Quantity):
            raise TypeError(
                "timepoints must be a pint Quantity array, e.g. "
                "Q_(np.array([0, 15, 30]), 'minute')"
            )
        if len(timepoints) != n:
            raise ValueError(
                f"timepoints length {len(timepoints)} does not match the number "
                f"of frames {n}"
            )
        return timepoints

    if frame_interval is not None:
        interval = to_quantity(frame_interval)
        return np.arange(n) * interval

    return None


def pixel_input_unit(pixel_size, dim: int):
    """Derive the per-pixel input unit for a length (``dim=1``) or area (``dim=2``).

    Accepts a scalar pint ``Quantity`` (isotropic), a 2-element pint array, or a
    ``(x, y)`` tuple/list of quantities (e.g. from OMERO). Returns ``None`` if
    ``pixel_size`` is ``None``. For anisotropic input, area uses the product of
    the two axes and length their geometric mean (axis order does not matter).
    """
    if pixel_size is None:
        return None

    if isinstance(pixel_size, tuple | list):
        comps = [to_quantity(p) for p in pixel_size]
    elif isinstance(pixel_size, ureg.Quantity) and np.ndim(pixel_size.magnitude) >= 1:
        comps = list(pixel_size)
    else:
        comps = [to_quantity(pixel_size)]

    if len(comps) == 1:
        return comps[0] ** dim

    px_a, px_b = comps[0], comps[1]
    if dim == 2:
        return px_a * px_b
    return (px_a * px_b) ** 0.5
