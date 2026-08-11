"""Filters for segmentating overlay objects"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import cv2
import numpy as np
import shapely
import shapely.affinity
import tqdm.auto as tqdm
from rtree import index
from shapely.geometry import Polygon
from shapely.validation import make_valid

from acia import Q_, ureg
from acia.base import Contour, ImageSequenceSource, Overlay

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


def bbox_to_rectangle(bbox: tuple[float, float, float, float]):
    minx, miny, maxx, maxy = bbox
    return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])


class NMSFilter:
    """Non-maximum supression filter based on contours"""

    @staticmethod
    def filter(overlay: Overlay, iou_thr=0.1, mode="iou") -> Overlay:
        prefiltered_contours = [
            cont for cont in overlay.contours if len(cont.coordinates) >= 3
        ]

        # sort contours by their score (lowest first)
        sorted_contours = sorted(
            prefiltered_contours, key=lambda c: c.score if c.score is not None else 0.0
        )
        # make (valid) shapely polygons
        polygons = [
            make_valid(shapely.geometry.polygon.Polygon(contour.coordinates))
            for contour in sorted_contours
        ]

        keep_list = []

        # build an rtree with bounding boxes
        idx = index.Index()
        for i, p_i in enumerate(polygons):
            minx, miny, maxx, maxy = p_i.bounds
            left = minx
            right = maxx
            top = maxy
            bottom = miny
            idx.insert(i, (left, bottom, right, top))

        for i, p_i in tqdm.tqdm(enumerate(polygons), total=len(polygons)):
            keep = True

            # zero area stuff is not considered
            if p_i.area <= 0:
                keep = False
                keep_list.append(keep)
                continue

            # get the intersection candidates by querying the rtree (overlapping bboxes)
            minx, miny, maxx, maxy = p_i.bounds
            left = minx
            right = maxx
            top = maxy
            bottom = miny
            candidate_idx_iter = idx.intersection((left, bottom, right, top))

            candidate_idx_list = list(
                filter(
                    partial(
                        lambda index, loop_index, sorted_contours: (
                            index > loop_index
                            and sorted_contours[loop_index].frame
                            == sorted_contours[index].frame
                        ),
                        loop_index=i,
                        sorted_contours=sorted_contours,
                    ),
                    candidate_idx_iter,
                )
            )

            # for those candidates we will compute the intersections in details
            for j in candidate_idx_list:
                p_j = polygons[j]

                # compute iou
                if mode == "i":
                    iou = p_i.intersection(p_j).area / p_i.area  # (p_i.union(p_j).area)
                elif mode == "iou":
                    iou = p_i.intersection(p_j).area / (p_i.union(p_j).area)
                # compare to threshold
                if iou >= iou_thr:
                    # if exceeding iou drop this cell detection
                    # print("iou: %.2f" % iou)
                    keep = False
                    break

            keep_list.append(keep)

        overlay = Overlay(
            [cont for i, cont in enumerate(sorted_contours) if keep_list[i]]
        )

        return overlay


class SizeFilter:
    """Filter by contour area"""

    @staticmethod
    def filter(overlay: Overlay, min_area, max_area) -> Overlay:
        """Filter an overlay based on contour sizes

        Args:
            overlay (Overlay): the overlay to filter
            min_area ([type]): minimum area of a contour
            max_area ([type]): maximum area of a contour

        Returns:
            Overlay: the filtered overlay
        """
        contour_shapes = [Polygon(cont.coordinates) for cont in overlay.contours]
        result_overlay = Overlay([])
        for cont, shape in zip(overlay.contours, contour_shapes, strict=False):
            area = shape.area

            if min_area < area < max_area:
                result_overlay.add_contour(cont)

        return result_overlay


class EllipsoidFilter:
    """Contour filter for ellipsoid shape"""

    @staticmethod
    def filter(
        overlay: Overlay, min_width_height_ratio, max_width_height_ratio
    ) -> Overlay:
        result_overlay = Overlay([])
        for cont in overlay.contours:
            if len(cont.coordinates) < 5:
                # need 5 points for ellipsoid fit
                continue
            ellipse = cv2.fitEllipse(cont.coordinates)

            center, (width, height), rotation = ellipse

            width_height_ratio = width / height

            # from shapely.figures import SIZE, GREEN, GRAY, set_limits

            # Let create a circle of radius 1 around center point:
            circ = shapely.geometry.Point(center).buffer(1)

            # Let create the ellipse along x and y:
            ell = shapely.affinity.scale(circ, width / 2, height / 2)

            # Let rotate the ellipse (clockwise, x axis pointing right):
            ellr = shapely.affinity.rotate(ell, rotation)

            # create shapely shape from contour coordinates
            shape = Polygon(cont.coordinates)
            # create minimal rotated rectangle
            min_rect = shape.minimum_rotated_rectangle

            rect_area_error = np.abs(shape.area - min_rect.area) / shape.area
            ellipse_area_error = np.abs(ellr.area - shape.area) / shape.area

            if (
                min_width_height_ratio <= width_height_ratio <= max_width_height_ratio
                and ellipse_area_error < rect_area_error
            ):
                # if an ellipse can better explain the cell detection than a rectangle
                result_overlay.add_contour(cont)

        return result_overlay


def _unit_is_dimensionless(unit) -> bool:
    """Whether ``unit`` (a unit string / pint unit / ``None``) has no dimension."""
    if unit is None:
        return True
    try:
        return bool(ureg.Quantity(str(unit)).dimensionless)
    except Exception:
        return False


def _column_magnitudes(
    properties: pd.DataFrame, name: str
) -> tuple[np.ndarray, object | None]:
    """Return ``(magnitudes, unit)`` for the ``name`` column of ``properties``.

    Handles both representations an extractor table can be in: plain floats with
    the unit map in ``df.attrs["units"]``, and ``pint``-dtype columns (the
    ``units="pint"`` form), whose unit lives on the dtype instead.
    """
    from acia.analysis.units import UNIT_ATTR

    if name not in properties.columns:
        raise KeyError(
            f"Filter {name!r} needs a {name!r} column in the properties table, "
            f"but it has {sorted(map(str, properties.columns))}. Add the matching "
            f"property extractor (e.g. the one whose `name` is {name!r}) to the "
            "ExtractorExecutor call that produced this table."
        )

    series = properties[name]

    # pint-dtype column: the unit is on the dtype, not in attrs
    try:
        import pint_pandas

        if isinstance(series.dtype, pint_pandas.PintType):
            return (
                np.asarray(series.pint.magnitude, dtype=float),
                series.pint.units,
            )
    except ImportError:  # pragma: no cover - pint_pandas is a hard dependency
        pass

    unit = properties.attrs.get(UNIT_ATTR, {}).get(name)
    return np.asarray(series.to_numpy(), dtype=float), unit


def _bound_magnitude(bound, unit) -> float:
    """Express ``bound`` as a plain number in ``unit``.

    Converting the two bounds once per run replaces the per-contour pint
    comparison the row-wise path used to do. The dimensionality check that the
    comparison provided is preserved -- it simply happens here, once, and so
    fails before any measurement rather than on the first contour.
    """
    if isinstance(bound, ureg.Quantity):
        if _unit_is_dimensionless(unit):
            # a dimensionless column only accepts a dimensionless bound; .to()
            # raises pint.DimensionalityError otherwise, as the old path did
            return float(bound.to("dimensionless").magnitude)
        return float(bound.to(unit).magnitude)
    return float(bound)


def _rotated_rect_coords(polygon) -> np.ndarray | None:
    """Min-rotated-rectangle vertices of ``polygon``; ``None`` if degenerate.

    A degenerate (``None``, empty, zero-area, or collinear) polygon has a
    ``minimum_rotated_rectangle`` that collapses to a ``LineString``/``Point``
    with no ``exterior``. Callers treat ``None`` as a 0 measurement so junk
    contours are dropped rather than crashing the whole filtering run.
    """
    if polygon is None or polygon.is_empty:
        return None
    mrr = polygon.minimum_rotated_rectangle
    exterior = getattr(mrr, "exterior", None)
    if exterior is None:
        return None
    return np.array(exterior.coords)


class CellFilter:
    """Pluggable, physical-unit cell filter for an :class:`~acia.base.Overlay`.

    A ``CellFilter`` keeps contours whose backing property (its calibrated
    :meth:`value`) falls within a ``(vmin, vmax)`` range. The range bounds are
    pint ``Quantity`` objects in *physical* units (e.g. ``Q_(2, "um**2")``) so
    the same filter stays valid across cameras with different ``pixel_size`` --
    the per-contour :meth:`value` is calibrated from the source ``pixel_size``,
    not measured in raw pixels.

    Mirrors :class:`~acia.analysis.PropertyExtractor`: adding a new filter is a
    single small subclass that sets ``name`` and overrides :meth:`value`; there
    is no central registry. Instances are simply passed in a list to
    :func:`apply_cell_filters`.

    Attributes:
        name: short identifier for the backing property (e.g. ``"area"``).
        vmin: inclusive lower bound (pint ``Quantity`` / number), or ``None``
            for an open lower bound.
        vmax: inclusive upper bound (pint ``Quantity`` / number), or ``None``
            for an open upper bound.
    """

    #: short identifier for the backing property; overridden by subclasses.
    name: str = "cell"

    def __init__(
        self,
        vmin: Q_ | float | None = None,
        vmax: Q_ | float | None = None,
    ) -> None:
        """Create a filter with an inclusive ``(vmin, vmax)`` range.

        Args:
            vmin: inclusive lower bound as a pint ``Quantity`` (physical unit),
                a plain number for dimensionless properties, or ``None`` for an
                open lower bound.
            vmax: inclusive upper bound, analogous to ``vmin``.
        """
        self.vmin = vmin
        self.vmax = vmax

    @property
    def range(self) -> tuple[Q_ | float | None, Q_ | float | None]:
        """The ``(vmin, vmax)`` range driving this filter."""
        return (self.vmin, self.vmax)

    def value(self, cont: Contour, *, images: ImageSequenceSource) -> Q_:
        """Return the calibrated physical value of ``cont`` for this filter.

        Subclasses derive a raw pixel measurement from ``cont`` and convert it
        to physical units using ``images.pixel_size``.

        Args:
            cont: the contour to measure.
            images: the calibrated image source (provides ``pixel_size``).

        Returns:
            The contour's property as a pint ``Quantity``.

        Raises:
            NotImplementedError: always, in the base class.
        """
        raise NotImplementedError()

    def accepts(self, cont: Contour, *, images: ImageSequenceSource) -> bool:
        """Whether ``cont`` falls within the ``(vmin, vmax)`` range.

        The comparison uses pint, so a dimensionality mismatch between the
        contour value and a bound (e.g. a µm² value vs a µm bound) raises a
        ``pint.DimensionalityError`` -- a deliberate guard against misconfigured
        filters. ``None`` bounds are open on that side.

        Args:
            cont: the contour to test.
            images: the calibrated image source (provides ``pixel_size``).

        Returns:
            ``True`` if the contour's value is within range.
        """
        v = self.value(cont, images=images)
        return (self.vmin is None or v >= self.vmin) and (
            self.vmax is None or v <= self.vmax
        )

    def mask(self, properties: pd.DataFrame) -> np.ndarray:
        """Boolean keep-mask over ``properties``, aligned with its row order.

        This is the path :func:`apply_cell_filters` uses. The values come from
        the column named after this filter -- the one its matching
        :class:`~acia.analysis.PropertyExtractor` already produced -- instead of
        being measured again from the contours, which is both the expensive part
        and a second, independent implementation of the same measurement.

        Args:
            properties: an extractor table (see
                :meth:`~acia.analysis.ExtractorExecutor.execute`) that contains a
                column named :attr:`name`.

        A row whose value is not finite is **always** dropped, including by a
        filter with no bounds at all. ``nan`` is what an extractor reports for a
        contour whose geometry it cannot measure -- a collinear outline, a mask
        with no pixels -- and a cell whose length is unknown cannot be asserted
        to lie within a length range. Dropping it here is what the row-wise
        filters effectively did before, and it keeps such a detection from
        surviving a one-sided bound.

        Returns:
            A boolean ``np.ndarray`` of ``len(properties)``, ``True`` where the
            row is finite and within the ``(vmin, vmax)`` range.

        Raises:
            KeyError: if ``properties`` has no column for this filter.
            pint.DimensionalityError: if a bound's dimension does not match the
                column's -- the same guard :meth:`accepts` provides, applied once
                per run instead of once per contour.
        """
        values, unit = _column_magnitudes(properties, self.name)

        keep = np.isfinite(values)
        if self.vmin is not None:
            keep &= values >= _bound_magnitude(self.vmin, unit)
        if self.vmax is not None:
            keep &= values <= _bound_magnitude(self.vmax, unit)
        return np.asarray(keep, dtype=bool)


class _ExtractorCalibratedFilter(CellFilter):
    """Base for filters that reuse an :class:`~acia.analysis.PropertyExtractor`.

    The shared extractor instance provides acia's canonical calibration
    (``convert`` / ``_dim`` / auto-unit from ``pixel_size``) and physical-unit
    bookkeeping, so values match exactly how acia reports the property.
    """

    def _extractor(self):
        """Return the (auto-unit) extractor instance backing this filter."""
        raise NotImplementedError()

    def _raw_value(self, cont: Contour) -> float:
        """Return the raw pixel measurement to be calibrated by the extractor."""
        raise NotImplementedError()

    def value(self, cont: Contour, *, images: ImageSequenceSource) -> Q_:
        extractor = self._extractor()
        # reuse acia's pixel_size -> physical-unit calibration
        extractor._calibrate(images)
        magnitude = extractor.convert(self._raw_value(cont))
        return magnitude * extractor.output_unit


class AreaFilter(_ExtractorCalibratedFilter):
    """Filter cells by physical area (``pixel_size**2`` -> µm²).

    Reuses :class:`~acia.analysis.AreaEx` for calibration, so the range bounds
    must be areas, e.g. ``AreaFilter(Q_(2, "um**2"), Q_(20, "um**2"))``.
    """

    name = "area"

    def _extractor(self):
        from acia.analysis import AreaEx

        return AreaEx()

    def _raw_value(self, cont: Contour) -> float:
        return cont.area


class LengthFilter(_ExtractorCalibratedFilter):
    """Filter cells by physical length (major axis, ``pixel_size`` -> µm).

    Length is the longer edge of the contour's minimum rotated bounding box,
    computed exactly like :class:`~acia.analysis.LengthEx`.
    """

    name = "length"

    def _extractor(self):
        from acia.analysis import LengthEx

        return LengthEx()

    def _raw_value(self, cont: Contour) -> float:
        from acia.utils import pairwise_distances

        coords = _rotated_rect_coords(cont.polygon)
        if (
            coords is None
        ):  # degenerate contour -> 0 length (dropped by a positive vmin)
            return 0.0
        return float(np.max(pairwise_distances(coords)))


class WidthFilter(_ExtractorCalibratedFilter):
    """Filter cells by physical width (minor axis, ``pixel_size`` -> µm).

    Width is the shorter edge of the contour's minimum rotated bounding box,
    computed exactly like :class:`~acia.analysis.WidthEx`.
    """

    name = "width"

    def _extractor(self):
        from acia.analysis import WidthEx

        return WidthEx()

    def _raw_value(self, cont: Contour) -> float:
        from acia.utils import pairwise_distances

        coords = _rotated_rect_coords(cont.polygon)
        if coords is None:  # degenerate contour -> 0 width (dropped by a positive vmin)
            return 0.0
        return float(np.min(pairwise_distances(coords)))


class CircularityFilter(CellFilter):
    """Filter cells by circularity (dimensionless, ``4*pi*area / perimeter**2``).

    Computed exactly like :class:`~acia.analysis.CircularityEx`. Bounds are
    dimensionless (plain floats or dimensionless ``Quantity``), e.g.
    ``CircularityFilter(vmin=0.8)``.
    """

    name = "circularity"

    def value(self, cont: Contour, *, images: ImageSequenceSource) -> Q_:
        polygon = cont.polygon
        if polygon is None or polygon.is_empty or polygon.length == 0:
            # degenerate contour -> circularity 0 (dropped by a positive vmin)
            return Q_(0.0)
        circularity = (4 * np.pi * polygon.area) / polygon.length**2
        # dimensionless quantity so comparisons stay pint-consistent
        return Q_(float(circularity))


class BoundaryClosenessFilter(CellFilter):
    """Drop cells whose bounding box lies near any image border.

    The :meth:`value` is the minimum distance (in physical units) from the
    contour's bounding box to any of the four image borders, using the source
    ``size_h`` / ``size_w`` and ``pixel_size``. The range is
    ``(min_distance, None)`` so cells closer than ``min_distance`` to a border
    are dropped.
    """

    name = "boundary_closeness"

    def __init__(self, min_distance: Q_) -> None:
        """Create the filter.

        Args:
            min_distance: minimum allowed distance from any image border as a
                pint length ``Quantity`` (e.g. ``Q_(1, "um")``).
        """
        super().__init__(vmin=min_distance, vmax=None)

    def value(self, cont: Contour, *, images: ImageSequenceSource) -> Q_:
        polygon = cont.polygon
        if polygon is None or polygon.is_empty:
            # degenerate contour -> distance 0 (treated as at-border, dropped)
            raw = 0.0
        else:
            minx, miny, maxx, maxy = polygon.bounds
            # distance (in px) to each of the four borders; x in [0, size_w], y in
            # [0, size_h]. The bbox is the closest part of the contour to a border.
            raw = float(min(minx, miny, images.size_w - maxx, images.size_h - maxy))

        # reuse acia's length calibration (pixel_size -> µm) via LengthEx
        from acia.analysis import LengthEx

        extractor = LengthEx()
        extractor._calibrate(images)
        return extractor.convert(raw) * extractor.output_unit


def apply_cell_filters(
    overlay: Overlay,
    filters: Sequence[CellFilter],
    *,
    properties: pd.DataFrame,
) -> Overlay:
    """Keep contours accepted by ALL filters, preserving the overlay time model.

    ``properties`` is the table an :class:`~acia.analysis.ExtractorExecutor`
    already produced for this overlay. Each filter reads its own column from it,
    so the contours are measured once (during extraction) rather than a second
    time here. Calibration comes from the table's units, which is why no image
    source is needed.

    Args:
        overlay: the overlay to filter.
        filters: the cell filters to apply; a contour is kept iff every filter
            accepts it (logical AND). An empty filter list keeps everything.
        properties: the extractor table describing ``overlay``, indexed by
            contour id. Every filter needs a column named after it.

    Returns:
        A new :class:`~acia.base.Overlay` with the kept contours and the same
        time model (timepoints / frame_interval). The result may be empty.

    Raises:
        ValueError: if ``properties`` does not describe every contour.
        KeyError: if a filter has no matching column in ``properties``.
    """
    keep = np.ones(len(properties), dtype=bool)
    for cell_filter in filters:
        keep &= cell_filter.mask(properties)

    kept_ids = set(properties.index[keep])

    # A contour the table does not describe cannot be judged. Dropping it
    # silently would quietly shrink the overlay, so say so instead.
    missing = [c.id for c in overlay.contours if c.id not in properties.index]
    if missing:
        raise ValueError(
            f"{len(missing)} contour(s) are missing from the properties "
            f"table (first: {missing[:3]}). It must describe the overlay "
            "being filtered -- extract before filtering, on the unfiltered "
            "overlay."
        )

    # iterate the overlay, not the table, so contour order is preserved
    kept = [cont for cont in overlay.contours if cont.id in kept_ids]

    # preserve the overlay's time model (private attrs, since Overlay exposes
    # only the resolved `timepoints` property, not `frame_interval`)
    return Overlay(
        kept,
        timepoints=getattr(overlay, "_timepoints", None),
        frame_interval=getattr(overlay, "_frame_interval", None),
    )
