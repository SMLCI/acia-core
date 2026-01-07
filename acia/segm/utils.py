"""Utils for segmentation data handling"""


import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from tqdm.auto import tqdm

from acia.base import Contour, Overlay
from acia.utils import pairwise_distances


def compute_indices(frame: int, size_t: int, size_z: int) -> tuple[int, int]:
    """Compute t and z values from a linearized frame number

    Args:
        frame (int): the linearized frame index
        size_t (int): the total size of the t dimension
        size_z (int): the total size of the z dimension

    Returns:
        (Tuple[int, int]): tuple of (t,z) indices
    """

    if size_t > 1 and size_z > 1:
        t = int(np.floor(frame / size_t))
        z = frame % size_t
    elif size_t > 1:
        t = frame
        z = 0
    elif size_z >= 1:
        t = 0
        z = frame
    elif size_t == 1 and size_z == 1:
        t = 0
        z = 0
    else:
        raise ValueError("This state should not be reachable!")

    return t, z


def length_and_area(contour: Contour) -> tuple[float, float]:
    """Compute length and area of a contour object (in pixel coordinates)

    Args:
        contour (Contour): contour object

    Returns:
        tuple[float, float]: length and area of the contour
    """

    polygon = Polygon(contour.coordinates)

    length = np.max(
        pairwise_distances(np.array(polygon.minimum_rotated_rectangle.exterior.coords))
    )
    return length, polygon.area


def merge_cells_to_colonies(overlay: Overlay, expand=10) -> Overlay:
    """Computing colony blobs from single-cell overlay

    Args:
        overlay (Overlay): Single-cell overlay containing the individual cell objects
        expand (int, optional): The number of pixels to expand single-cell objects in order to form blobs. Defaults to 10.

    Returns:
        Overlay: Overlay of colony blobs
    """

    merged_contours = []

    # iterate over frames and all the cell instances
    for frame_ind, frame_overlay in enumerate(
        tqdm(overlay.timeIterator(), desc="Merging cells to colonies...")
    ):

        # get all polygons
        cont_polys = [cont.polygon for cont in frame_overlay]

        # increase their size (like a dilation)
        oversized_polys = [poly.buffer(expand) for poly in cont_polys]

        # merge all polys
        intersection = unary_union(oversized_polys)

        # erose the merged polygon
        i = intersection.buffer(-expand)

        # make it a contour
        polygons = [i]
        if isinstance(i, MultiPolygon):
            polygons = list(i.geoms)

        contours = [
            Contour(
                np.array(list(zip(p.exterior.xy))).T.squeeze(), -1, frame_ind, frame_ind
            )
            for p in polygons
        ]

        contours = list(filter(lambda c: len(c.coordinates) >= 3, contours))

        # add merged contour to results
        merged_contours += contours

    # return new overlay with merged contours
    return Overlay(merged_contours)
