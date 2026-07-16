"""Named channel colors for `ImageSequenceSource.to_rgb(colors=...)` composites.

This is a minimal starter palette (see `color-constants.md` in the
`spec-to-rgb` implementation artifact), not a comprehensive named-color
library -- expanding it is future work.
"""

from __future__ import annotations

import matplotlib.colors as mcolors

CHANNEL_COLORS: dict[str, str] = {
    "DAPI": "#0000FF",
    "FITC": "#00FF00",
    "GFP": "#00FF00",
    "TRITC": "#FF0000",
    "RFP": "#FF0000",
    "CY5": "#FF00FF",
    "BRIGHTFIELD": "#FFFFFF",
}


def resolve_channel_color(value: str) -> tuple[float, float, float]:
    """Resolve a channel color to an RGB triple in ``[0, 1]``.

    Looks ``value`` up in :data:`CHANNEL_COLORS` case-insensitively first;
    if it is not a known channel name, ``value`` is passed straight to
    :func:`matplotlib.colors.to_rgb` (so hex strings like ``"#00FF00"`` and
    any matplotlib-recognized color name work too).

    Args:
        value: A key of :data:`CHANNEL_COLORS` (case-insensitive), a hex
            color string, or any color accepted by
            :func:`matplotlib.colors.to_rgb`.

    Returns:
        tuple[float, float, float]: RGB triple with each component in
        ``[0, 1]``.

    Raises:
        ValueError: if ``value`` is neither a known channel name nor a color
            :func:`matplotlib.colors.to_rgb` can parse.
    """
    resolved = (
        CHANNEL_COLORS.get(value.upper(), value) if isinstance(value, str) else value
    )

    try:
        r, g, b = mcolors.to_rgb(resolved)
    except ValueError as exc:
        raise ValueError(f"Unknown or invalid channel color: {value!r}") from exc
    return (r, g, b)
