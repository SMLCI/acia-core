"""Global utilities"""

from __future__ import annotations

import numpy as np
import rasterio
import shapely
from rasterio import features
from shapely.geometry import MultiPolygon, Polygon


def lut_mapping(image, in_min, in_max, out_min, out_max, dtype=None):
    mapped_data = (np.clip(image, in_min, in_max) - in_min) / (in_max - in_min) * (
        out_max - out_min
    ) + out_min

    if dtype:
        mapped_data = mapped_data.astype(dtype)

    return mapped_data


def pairwise_distances(points: np.ndarray) -> list[float]:
    distances: list[float] = []

    if len(points) == 0:
        return distances

    for a, b in zip(points, points[1:], strict=False):
        distances.append(float(np.linalg.norm(a - b)))

    return distances


def mask_to_polygons(mask: np.ndarray) -> Polygon | MultiPolygon | None:
    """Convert a mask to a Polygon or Multipolygon

    Args:
        mask (np.ndarray): Binary mask for an object

    Returns:
        shapely.geometry.Polygon | shapely.geometry.MultiPolygon | None:
            Extracted polygon structure, or None if the mask is empty
    """
    all_polygons = []
    for shape, _ in features.shapes(mask.astype(np.int16), mask=(mask > 0)):
        all_polygons.append(shapely.geometry.shape(shape))

    if len(all_polygons) == 0:
        return None

    if len(all_polygons) > 1:
        all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    else:
        all_polygons = all_polygons[0]

    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)

    return all_polygons


def largest_polygon(
    polygon: Polygon | MultiPolygon | None,
) -> Polygon | None:
    """Reduce a possibly-multi-part polygon to its single largest-area part.

    :func:`mask_to_polygons` returns a ``MultiPolygon`` whenever an object's
    mask has disconnected components -- a cell the segmentation split in two, a
    stray speck sharing the cell's label, or a self-intersecting outline that
    ``buffer(0)`` repaired into several pieces. Anything that needs *one*
    closed outline (a contour's coordinates, a drawn shape) has to pick a part,
    and the largest is the one that represents the object.

    Note that this discards the smaller parts, so a caller that persists or
    measures the result is losing whatever area they held; callers in a
    position to say so should report it. :attr:`~acia.base.Instance.area` is
    unaffected, being computed from the mask rather than the polygon.

    Args:
        polygon: A ``Polygon``, a ``MultiPolygon``, or ``None``.

    Returns:
        shapely.geometry.Polygon | None: The largest constituent polygon,
            ``polygon`` itself when it is already a single ``Polygon``, or
            ``None`` when ``polygon`` is ``None`` or holds no parts.
    """
    if polygon is None:
        return None
    if isinstance(polygon, MultiPolygon):
        parts = list(polygon.geoms)
        if not parts:
            return None
        return max(parts, key=lambda part: part.area)
    return polygon


def multi_mask_to_polygons(
    mask: np.ndarray,
) -> list[tuple[int, Polygon | MultiPolygon]]:
    unique_values = np.unique(mask)
    instance_ids = unique_values[unique_values > 0]

    polygons = []

    for instance_id in instance_ids:
        polygons.append((instance_id, mask_to_polygons(mask == instance_id)))

    return polygons


def polygon_to_mask(polygon, height: int, width: int):
    """Converts a polygon to a mask

    Args:
        polygon (_type_): shapely polygon or multipolygon
        height (int): height of the mask
        width (int): width of the mask

    Returns:
        (np.ndarray): boolean mask
    """
    return rasterio.features.rasterize(
        [polygon],
        out_shape=(height, width),
    ).astype(bool)


class ScaleBar:
    """Scalebar class"""

    #: width of the rendered bar in pixels (set by concrete subclasses)
    pixelWidth: int

    def draw(self, image: np.ndarray, xstart: int, ystart: int):
        raise NotImplementedError("Do not use the base class")
