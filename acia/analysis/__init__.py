"""Functionality for single-cell analysis"""

from __future__ import annotations

import logging
import multiprocessing
import os
import shutil
import warnings
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce
from itertools import starmap
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

import numpy as np
import pandas as pd
import papermill as pm
import shapely
from numpy import ma
from pint._typing import UnitLike
from tqdm.auto import tqdm

from acia import Q_, U_
from acia.analysis.growth_rate import AggMode as AggMode
from acia.analysis.growth_rate import GrowthRateResult as GrowthRateResult
from acia.analysis.growth_rate import estimate_growth_rate as estimate_growth_rate
from acia.analysis.stage import DEFAULT_KEY_PATTERN as DEFAULT_KEY_PATTERN
from acia.analysis.stage import MANIFEST_NAME as MANIFEST_NAME
from acia.analysis.stage import StageContext as StageContext
from acia.analysis.stage import population_id_of as population_id_of
from acia.analysis.stage import read_manifest as read_manifest
from acia.analysis.stage import stages_run as stages_run
from acia.analysis.units import UNIT_ATTR, units_in_header
from acia.analysis.units import attach_units as attach_units
from acia.analysis.units import from_header as from_header
from acia.analysis.units import read_units_csv as read_units_csv
from acia.analysis.units import strip_units as strip_units
from acia.analysis.units import write_units_csv as write_units_csv
from acia.base import BaseImage, ImageSequenceSource, Overlay

logger = logging.getLogger(__name__)

DEFAULT_UNIT_LENGTH = "micrometer"
DEFAULT_UNIT_AREA = "micrometer ** 2"


class PropertyExtractor:
    """Base class for single-cell property extractor"""

    def __init__(
        self,
        name: str,
        input_unit: UnitLike | None,
        output_unit: UnitLike | None = None,
    ):
        self.name = name

        # try to parse input quantity
        self.input_unit = Q_(input_unit)  # type: ignore[arg-type]
        if self.input_unit.dimensionless and isinstance(self.input_unit.magnitude, U_):
            # if we have no dimension and magnitude is unit -> we better go with a unit
            self.input_unit = U_(input_unit)
        if output_unit:
            # parse output unit
            self.output_unit = U_(output_unit)
        else:
            # no conversion if no output unit is specified
            self.output_unit = self.input_unit

        # test the conversion here
        self.output_unit.is_compatible_with(self.input_unit)

        # remember the configured units so auto-calibration can fall back to them
        self._default_input_unit = self.input_unit
        self._default_output_unit = self.output_unit

    #: spatial dimensionality for pixel-size auto-calibration (1=length, 2=area)
    _dim: int | None = None

    def _calibrate(self, images: ImageSequenceSource):
        """When in auto mode, derive the spatial ``input_unit`` from the source's
        ``pixel_size``; otherwise keep the explicitly configured units."""
        if not getattr(self, "_auto_unit", False) or self._dim is None:
            return

        from acia.timing import pixel_input_unit

        iu = pixel_input_unit(getattr(images, "pixel_size", None), self._dim)
        if iu is not None:
            self.input_unit = iu
            self.output_unit = iu.units
        else:
            # No source calibration. The configured default is used, which
            # labels the column µm (or µm²) while the values are really pixels
            # -- a physical-unit filter or plot downstream would then compare
            # against a scale that does not exist. Filtering used to refuse
            # uncalibrated sources outright; now that it reads this table
            # instead of the source, the ambiguity has to be reported here.
            warnings.warn(
                f"{type(self).__name__}: the image source has no pixel_size, so "
                f"{self.name!r} is measured in pixels but labelled "
                f"{self._default_output_unit!r}. Load the source with "
                "pixel_size=... for physically meaningful values.",
                UserWarning,
                stacklevel=3,
            )
            self.input_unit = self._default_input_unit
            self.output_unit = self._default_output_unit

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        """Extract the desired properties for a single contour

        Args:
            contour (Contour): contour for the qunatity
            overlay (Overlay): overlay containing all contours
            df (pd.DataFrame): DataFrame of properties so far

        Raises:
            NotImplementedError: Please implement this method
        """
        raise NotImplementedError()

    def convert(self, input: float | Q_) -> float:
        """Converts input to the specified output unit

        Args:
            input (float | Quantity): Input value

        Returns:
            float: the magnitude in the output unit
        """
        if isinstance(input, Q_):
            # 1. convert input to input unit
            # 2. scale with input unit
            # 3. convert to output unit
            return float(
                (input.to(self.input_unit).magnitude * self.input_unit)
                .to(self.output_unit)
                .magnitude
            )
        else:
            # 1. scale input with input unit/quantity
            # 2. convert to output unit
            return float((input * self.input_unit).to(self.output_unit).magnitude)

    def _affine(self) -> tuple[float, float] | None:
        """``(scale, offset)`` with ``convert(x) == scale * x + offset``.

        A unit conversion is affine, so the whole of :meth:`convert` -- a pint
        ``Quantity`` construction and two unit lookups, per value -- collapses to
        one multiply once the two coefficients are known.

        Returns ``None`` if the identity does not hold *exactly* on a spread of
        probe values, so an unusual unit falls back to the per-value path rather
        than silently returning near-enough numbers.
        """
        offset = self.convert(0.0)
        scale = self.convert(1.0) - offset

        for probe in (0.5, 3.0, 1234.5, 1e-6, 1e6, 188.0):
            if scale * probe + offset != self.convert(probe):
                return None
        return scale, offset

    def convert_array(self, values) -> np.ndarray:
        """Vectorised :meth:`convert` over an array of magnitudes.

        Produces bit-identical results to calling :meth:`convert` per value
        (verified in ``tests/test_property_equivalence.py``), because the
        conversion really is a single multiply and this performs the same one.
        """
        values = np.asarray(values, dtype=float)
        affine = self._affine()
        if affine is None:
            return np.array([self.convert(float(v)) for v in values], dtype=float)

        scale, offset = affine
        if offset == 0.0:
            return np.asarray(values * scale, dtype=float)
        return np.asarray(values * scale + offset, dtype=float)


class ExtractorExecutor:
    """Executor to extract a list of single-cell properties from segmentation and images"""

    def __init__(self) -> None:
        self.units: dict[str, Any] = {}

    def execute(
        self,
        overlay: Overlay,
        images: ImageSequenceSource,
        extractors: list[PropertyExtractor] | None = None,
        units: str = "none",
    ):
        """Extract single-cell properties into a DataFrame.

        Args:
            overlay: the contours to extract properties from.
            images: the image source (needed e.g. for fluorescence).
            extractors: the property extractors to run.
            units: representation of physical units in the returned table:

                * ``"none"`` (default) -- plain numeric columns; the unit map is
                  carried in ``df.attrs["units"]``. Not unit-safe.
                * ``"header"`` -- plain values with the unit as a column-index
                  level (export/readable form). Not unit-safe.
                * ``"pint"`` -- ``pint[...]`` columns; the only representation
                  with unit-safe arithmetic (propagation + dimensional checks).

                The forms are convertible afterwards via
                :func:`acia.analysis.attach_units` / ``strip_units`` /
                ``units_in_header``.
        """
        if extractors is None:
            extractors = []

        valid_units = {"none", "magnitude", "header", "pint"}
        if units not in valid_units:
            raise ValueError(
                f"Unknown units={units!r}. Expected one of {sorted(valid_units)}."
            )

        df = pd.DataFrame()

        # TODO: make the id the index
        df["id"] = [c.id for c in overlay]
        df = df.set_index("id")

        # extractor results are joined on `id`, and a join on a non-unique index
        # is a cartesian product: k detections sharing an id turn into k**2 rows
        # per extractor, silently inflating every downstream sum. Fail loudly
        # instead of returning a table that is wrong by orders of magnitude.
        if not df.index.is_unique:
            duplicated = df.index[df.index.duplicated()].unique()
            examples = ", ".join(repr(i) for i in duplicated[:5])
            raise ValueError(
                f"The overlay has {len(duplicated)} duplicate contour id(s) "
                f"(e.g. {examples}). Property extraction joins the extractor "
                "results on `id`, so duplicates multiply the rows. Give every "
                "contour in the overlay a unique id."
            )

        # the bar names the property currently being extracted; it used to also
        # print one line per extractor to stdout, which a notebook re-running
        # this over many sources drowns in
        progress = tqdm(extractors, unit="property", desc="extracting")
        for extractor in progress:
            progress.set_postfix_str(extractor.name)
            result_df, extractor_units = extractor.extract(overlay, images, df)

            # join on the shared `id` index -- order-independent and, unlike
            # merge(on="id"), dtype-tolerant for an empty overlay (0 contours)
            df = df.join(result_df)

            self.units.update(**extractor_units)

        # carry the unit map with the table (replaces relying on self.units)
        df.attrs[UNIT_ATTR] = {k: str(v) for k, v in self.units.items()}

        # convert once, after all merges, to avoid pint+merge edge cases
        if units == "pint":
            df = attach_units(df)
        elif units == "header":
            df = units_in_header(df)
        # "none"/"magnitude" -> leave as plain floats

        return df


def _polygons(overlay: Overlay) -> list:
    """Every contour's polygon, in overlay order."""
    return [cont.polygon for cont in overlay]


def _rect_edge_lengths(polygons: list) -> np.ndarray:
    """``(n, 4)`` edge lengths of each polygon's minimum rotated rectangle.

    Computes the rectangles for the whole overlay in one shapely call rather
    than one per contour per extractor, which had ``LengthEx`` and ``WidthEx``
    each deriving the same rectangle independently.

    A degenerate outline (collinear, zero-extent, or absent) has a rectangle
    that collapses to a ``LineString``/``Point`` with no ``exterior``, so it has
    no length or width to report and measures ``nan``. Reporting 0 instead would
    be a claim -- and a 0-length cell passes any open-below bound, letting junk
    through -- whereas ``nan`` states that the property is undefined, and the
    filters drop what they cannot measure.
    """
    if not polygons:
        return np.empty((0, 4), dtype=float)

    rectangles = shapely.oriented_envelope(np.array(polygons, dtype=object))

    # a rectangle ring is 5 coordinates (closed); anything else is degenerate
    edges = np.full((len(polygons), 4), np.nan, dtype=float)
    intact = shapely.get_num_coordinates(rectangles) == 5
    if intact.any():
        coords = shapely.get_coordinates(rectangles[intact]).reshape(-1, 5, 2)
        edges[intact] = np.linalg.norm(np.diff(coords, axis=1), axis=2)
    return edges


def _value_frame(overlay: Overlay, name: str, values: np.ndarray) -> pd.DataFrame:
    """``id``-indexed single-column frame, float-typed even when empty."""
    return pd.DataFrame(
        {"id": [cont.id for cont in overlay], name: np.asarray(values, dtype=float)}
    ).set_index("id")


def _id_indexed(records: list[dict], columns: list[str]) -> pd.DataFrame:
    """Build an ``id``-indexed DataFrame from per-contour records, empty-safe.

    ``pd.DataFrame([])`` has no columns, so ``.set_index("id")`` raises for an
    empty overlay; passing explicit ``columns`` keeps ``id`` and the value
    column(s) present (as an empty frame) instead.
    """
    return pd.DataFrame(records, columns=["id", *columns]).set_index("id")


class AreaEx(PropertyExtractor):
    """Extract area for every contour"""

    _dim = 2

    def __init__(
        self,
        input_unit: UnitLike | None = None,
        output_unit: UnitLike | None = None,
    ):
        self._auto_unit = input_unit is None
        PropertyExtractor.__init__(
            self,
            "area",
            input_unit=input_unit if input_unit is not None else DEFAULT_UNIT_AREA,
            output_unit=output_unit if output_unit is not None else DEFAULT_UNIT_AREA,
        )

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        self._calibrate(images)
        # `cont.area` stays per-contour: Instance counts mask pixels while
        # Contour measures its polygon, and only the unit conversion is common
        areas = [cont.area for cont in overlay]
        df = _value_frame(overlay, self.name, self.convert_array(areas))

        return df, {self.name: self.output_unit}


class PerimeterEx(PropertyExtractor):
    """Extract area for every contour"""

    _dim = 1

    def __init__(
        self,
        input_unit: UnitLike | None = None,
        output_unit: UnitLike | None = None,
    ):
        self._auto_unit = input_unit is None
        PropertyExtractor.__init__(
            self,
            "perimeter",
            input_unit=input_unit if input_unit is not None else DEFAULT_UNIT_LENGTH,
            output_unit=output_unit if output_unit is not None else DEFAULT_UNIT_LENGTH,
        )

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        self._calibrate(images)
        polygons = _polygons(overlay)
        if polygons:
            # shapely.length yields nan for an absent polygon, which is exactly
            # right: there is no outline, so there is no perimeter to report
            perimeters = shapely.length(np.array(polygons, dtype=object))
        else:
            perimeters = np.empty(0)
        df = _value_frame(overlay, self.name, self.convert_array(perimeters))

        return df, {self.name: self.output_unit}


class CircularityEx(PropertyExtractor):
    """Extract area for every contour"""

    def __init__(
        self,
        input_unit: UnitLike | None = "1",
        output_unit: UnitLike | None = "1",
    ):
        PropertyExtractor.__init__(
            self, "circularity", input_unit=input_unit, output_unit=output_unit
        )

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        areas = np.asarray(df["area"], dtype=float)
        perimeters = np.asarray(df["perimeter"], dtype=float)

        # a degenerate contour has no perimeter to divide by, so its circularity
        # is undefined rather than 0 or inf (nan also propagates in from an
        # undefined perimeter on its own)
        circularities = np.full_like(areas, np.nan)
        measurable = perimeters != 0
        circularities[measurable] = (4 * np.pi * areas[measurable]) / perimeters[
            measurable
        ] ** 2

        df = pd.DataFrame({self.name: circularities, "id": df.index}).set_index("id")

        return df, {self.name: self.output_unit}


class BoundaryClosenessEx(PropertyExtractor):
    """Distance from each cell's bounding box to the nearest image border.

    The backing property for
    :class:`~acia.segm.filter.BoundaryClosenessFilter`, which filters on how
    close a cell sits to the edge of the field of view (a cell that is partly
    outside it has unreliable size and shape). Extracting it as a column means
    the filter reads it like every other filter reads its own property, and it
    becomes plottable next to them in
    :func:`~acia.analysis.properties.plot_property_histograms`.

    The frame extent comes from the source (``size_w`` / ``size_h``), so a cell
    whose bounding box touches a border measures 0.
    """

    _dim = 1

    def __init__(
        self,
        input_unit: UnitLike | None = None,
        output_unit: UnitLike | None = None,
    ):
        self._auto_unit = input_unit is None
        PropertyExtractor.__init__(
            self,
            "boundary_closeness",
            input_unit=input_unit if input_unit is not None else DEFAULT_UNIT_LENGTH,
            output_unit=output_unit if output_unit is not None else DEFAULT_UNIT_LENGTH,
        )

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        self._calibrate(images)
        size_w, size_h = images.size_w, images.size_h

        raw = []
        for cont in overlay:
            polygon = cont.polygon
            if polygon is None or polygon.is_empty:
                # degenerate contour -> distance 0 (treated as at-border)
                raw.append(0.0)
            else:
                minx, miny, maxx, maxy = polygon.bounds
                # the bounding box is the closest part of the contour to a border
                raw.append(float(min(minx, miny, size_w - maxx, size_h - maxy)))

        df = _value_frame(overlay, self.name, self.convert_array(raw))

        return df, {self.name: self.output_unit}


class LengthEx(PropertyExtractor):
    """Extracts width of cells based on the shorter edge of a minimum rotated bbox approximation"""

    _dim = 1

    def __init__(
        self,
        input_unit: UnitLike | None = None,
        output_unit: UnitLike | None = None,
    ):
        self._auto_unit = input_unit is None
        PropertyExtractor.__init__(
            self,
            "length",
            input_unit=input_unit if input_unit is not None else DEFAULT_UNIT_LENGTH,
            output_unit=output_unit if output_unit is not None else DEFAULT_UNIT_LENGTH,
        )

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        self._calibrate(images)
        edges = _rect_edge_lengths(_polygons(overlay))
        # longer edge of minimum rotated bbox
        lengths = edges.max(axis=1) if len(edges) else np.empty(0)
        df = _value_frame(overlay, self.name, self.convert_array(lengths))

        return df, {self.name: self.output_unit}


class WidthEx(PropertyExtractor):
    """Extracts width of cells based on the shorter edge of a minimum rotated bbox approximation"""

    _dim = 1

    def __init__(
        self,
        input_unit: UnitLike | None = None,
        output_unit: UnitLike | None = None,
    ):
        self._auto_unit = input_unit is None
        PropertyExtractor.__init__(
            self,
            "width",
            input_unit=input_unit if input_unit is not None else DEFAULT_UNIT_LENGTH,
            output_unit=output_unit if output_unit is not None else DEFAULT_UNIT_LENGTH,
        )

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        """Extract width information for all contours"""
        self._calibrate(images)
        edges = _rect_edge_lengths(_polygons(overlay))
        # shorter edge of bbox approximation
        widths = edges.min(axis=1) if len(edges) else np.empty(0)
        df = _value_frame(overlay, self.name, self.convert_array(widths))

        return df, {self.name: self.output_unit}


class LengthWidthEx(PropertyExtractor):
    """Extracts length and width of cells based on the shorter edge of a minimum rotated bbox approximation"""

    _dim = 1

    def __init__(
        self,
        prefix="",
        input_unit: UnitLike | None = None,
        output_unit: UnitLike | None = None,
    ):
        self.prefix = prefix
        self._auto_unit = input_unit is None

        PropertyExtractor.__init__(
            self,
            f"{prefix}length-width",
            input_unit=input_unit if input_unit is not None else DEFAULT_UNIT_LENGTH,
            output_unit=output_unit if output_unit is not None else DEFAULT_UNIT_LENGTH,
        )

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        """Extract length and width information for all contours"""
        self._calibrate(images)
        edges = _rect_edge_lengths(_polygons(overlay))
        if len(edges):
            widths = self.convert_array(edges.min(axis=1))
            lengths = self.convert_array(edges.max(axis=1))
        else:
            widths = lengths = np.empty(0)

        df = pd.DataFrame(
            {
                f"{self.prefix}width": widths,
                f"{self.prefix}length": lengths,
                "id": [c.id for c in overlay],
            }
        ).set_index("id")

        return df, {
            f"{self.prefix}width": self.output_unit,
            f"{self.prefix}length": self.output_unit,
        }


class FrameEx(PropertyExtractor):
    """Extract the frame information for every contour"""

    def __init__(self):
        super().__init__("frame", 1)

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        frames = []
        for cont in overlay:
            frames.append(self.convert(cont.frame))

        df = pd.DataFrame({self.name: frames, "id": [c.id for c in overlay]}).set_index(
            "id"
        )

        return df, {self.name: self.output_unit}


class IdEx(PropertyExtractor):
    """Extract single-cell id for every contour"""

    def __init__(self):
        super().__init__("id", 1)

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        ids = []
        for cont in overlay:
            ids.append(self.convert(cont.id))

        df = pd.DataFrame({self.name: ids}).set_index("id")
        df["id"] = df.index

        return df, {self.name: self.output_unit}


class LabelEx(PropertyExtractor):
    """Extract single-cell label (from tracking) for every contour"""

    def __init__(self):
        super().__init__("label", 1)

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        labels = []
        for cont in overlay:
            labels.append(self.convert(cont.label))

        df = pd.DataFrame({self.name: labels, "id": [c.id for c in overlay]}).set_index(
            "id"
        )

        return df, {self.name: self.output_unit}


class TimeEx(PropertyExtractor):
    """Extract time information for every contour.

    If no ``input_unit`` is given, the per-frame timepoints are taken from the
    image source (or the overlay) calibration -- so a source loaded with a
    ``frame_interval`` (or sliced) yields correct, automatically-updated times.
    Passing ``input_unit`` keeps the legacy ``frame * interval`` behavior.
    """

    def __init__(
        self, input_unit: UnitLike | None = None, output_unit: UnitLike | None = "hour"
    ):
        self._auto_unit = input_unit is None
        super().__init__(
            "time", input_unit if input_unit is not None else "second", output_unit
        )

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        local_df = df.copy()

        if not self._auto_unit:
            # explicit input_unit -> legacy frame * interval
            local_df[self.name] = local_df["frame"].apply(self.convert)
            return local_df[[self.name]], {self.name: self.output_unit}

        # auto: pull per-frame timepoints from the source, then the overlay
        timepoints = getattr(images, "timepoints", None)
        if timepoints is None:
            timepoints = getattr(overlay, "timepoints", None)
        if timepoints is None:
            raise ValueError(
                "TimeEx(): no time information available. Pass input_unit=..., or "
                "set a frame_interval/timepoints on the image source or overlay."
            )

        out_unit = str(self.output_unit)
        local_df[self.name] = local_df["frame"].apply(
            lambda f: float(timepoints[int(f)].to(out_unit).magnitude)
        )
        return local_df[[self.name]], {self.name: self.output_unit}


class DynamicTimeEx(PropertyExtractor):
    """Extract time information for every contour when timepoints are not equi-distant"""

    def __init__(
        self,
        timepoints: list,
        relative=True,
        input_unit: UnitLike = "second",
        output_unit: UnitLike | None = "hour",
    ):
        super().__init__("time", input_unit, output_unit)

        if len(timepoints) == 0:
            raise ValueError("Need non-empty timepoint list")

        self.timepoints = np.array(timepoints)

        if relative:
            self.timepoints -= self.timepoints[0]

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        # get the number of frames
        df_num_frames = np.unique(df["frame"])
        num_frames = images.size_t

        if len(self.timepoints) != len(df_num_frames):
            logger.warning(
                "Number of specified timepoints does not match with number of frames in the Dataframe: %d vs. %d timepoints",
                len(df_num_frames),
                len(self.timepoints),
            )

        if len(self.timepoints) != num_frames:
            raise ValueError(
                f"Number of specified timepoints does not match with number of frames in the time-lapse: {num_frames=} vs. {len(self.timepoints)} timepoints"
            )

        data = []
        for id, row in df.iterrows():
            data.append(
                {
                    self.name:
                    # convert to timepoint units
                    self.convert(
                        # lookup frame timepoint
                        self.timepoints[int(row["frame"])]
                    ),
                    "id": id,
                }
            )

        df = _id_indexed(data, [self.name])

        return df, {self.name: self.output_unit}


class PositionEx(PropertyExtractor):
    """Extract cell center information from image RoI detections"""

    _dim = 1

    def __init__(
        self,
        input_unit: UnitLike | None = None,
        output_unit: UnitLike | None = DEFAULT_UNIT_LENGTH,
    ):
        self._auto_unit = input_unit is None
        super().__init__(
            "position",
            input_unit=input_unit if input_unit is not None else DEFAULT_UNIT_LENGTH,
            output_unit=output_unit,
        )

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        self._calibrate(images)
        # `Contour.center` is float32 (derived from the float32 `coordinates`)
        # and pint used to carry that magnitude through, so these two columns
        # were computed in single precision. Converting the array in float64
        # shifts them by ~1e-7 relative -- sub-picometre on a micrometre
        # coordinate, so the added accuracy is free of consequence here.
        centers = [cont.center for cont in overlay]
        positions_x = self.convert_array([c[0] for c in centers])
        positions_y = self.convert_array([c[1] for c in centers])
        ids = [cont.id for cont in overlay]

        return pd.DataFrame(
            {"position_x": positions_x, "position_y": positions_y, "id": ids}
        ).set_index("id"), {
            "position_x": self.output_unit,
            "position_y": self.output_unit,
        }


class FluorescenceEx(PropertyExtractor):
    """Extracting fluorescence properties from image sequence and RoI detections"""

    def __init__(
        self,
        channels,
        channel_names,
        summarize_operator=np.median,
        input_unit: UnitLike = "1",
        output_unit: UnitLike | None = "",
        parallel=6,
    ):
        super().__init__("Fluorescence", input_unit=input_unit, output_unit=output_unit)

        self.channels = channels
        self.channel_names = channel_names
        self.summarize_operator = summarize_operator
        self.parallel = parallel

        assert len(self.channels) == len(self.channel_names), (
            "Number of channels and number of channel names must comply"
        )

    @staticmethod
    def extract_fluorescence(
        overlay: Overlay,
        image: BaseImage,
        channels: list[int],
        channel_names: list[str],
        summarize_operator,
    ):
        """Extract fluorescence information based on an overlay(segmentation) and corresponding image.

        Args:
            overlay (Overlay): Ovleray providing the image segmentation information
            image (BaseImage): the image itself
            channels (List[int]): list of channels (image channels) we want to investigate
            channel_names (List[str]): list of names for the channel results
            summarize_operator (_type_): summarizing operator, e.g. np.media, to compress all fluorescence values to a single one

        Returns:
            pd.DataFrame: pandas data frame containing columns of channel_names and the rows represent the extracted fluorescence
        """

        data = []

        for cont in overlay:
            local_data = {"id": cont.id}
            for ch_id, channel in enumerate(channels):
                raw_image = image.get_channel(channel)

                height, width = raw_image.shape[:2]

                # draw cell mask
                roi_mask = cont.toMask(height=height, width=width)

                # create masked array
                masked_roi: ma.MaskedArray = ma.masked_array(raw_image, mask=~roi_mask)

                # compute fluorescence response
                value = summarize_operator(masked_roi.compressed())

                local_data[channel_names[ch_id]] = value
            data.append(local_data)

        return _id_indexed(data, list(channel_names))

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        assert overlay.numFrames() == len(images), (
            "Please make sure that the frames in your overlay fit to the frames in your image source"
        )

        def iterator(timeIterator):
            for i, overlay in enumerate(timeIterator):
                yield (
                    overlay,
                    images.get_frame(i),
                    self.channels,
                    self.channel_names,
                    self.summarize_operator,
                )

        result: list[pd.DataFrame] = []

        if self.parallel > 1:
            try:
                with Pool(self.parallel) as p:
                    result = p.starmap(
                        FluorescenceEx.extract_fluorescence,
                        iterator(overlay.timeIterator()),
                        chunksize=5,
                    )

            except Exception as e:
                logging.error(
                    "Parallel fluorescence extraction failed! Please run with 'parallel=1' to investigate the error!"
                )
                raise e

        else:
            result = list(
                starmap(
                    FluorescenceEx.extract_fluorescence,
                    iterator(overlay.timeIterator()),
                )
            )

        # concatenate all results
        combined_result = reduce(lambda a, b: pd.concat([a, b]), result)

        return combined_result, {
            self.channel_names[i]: self.output_unit for i in range(len(self.channels))
        }


def default_execution_naming(source) -> str:
    """Source-aware default folder name for one scaled execution.

    * ``int`` (e.g. an OMERO image id) -> ``"execution_<id>"``
    * ``str`` (a file path or fsspec URL) -> the file *stem*, i.e. the file name
      without directory and extension (``smb://host/share/pos1.tif`` -> ``pos1``)

    For other item types (e.g. a parameter ``dict``) the name cannot be inferred;
    pass an explicit ``execution_naming`` to :func:`scale` in that case.
    """
    if isinstance(source, bool):  # bool is an int subclass; treat as unsupported
        raise ValueError(f"Cannot derive an execution name from {source!r}.")
    if isinstance(source, int):
        return f"execution_{source}"
    if isinstance(source, str):
        # strip a possible query string, then take the file stem (works for URLs)
        return Path(source.split("?", 1)[0]).stem
    raise ValueError(
        f"Cannot derive a default execution name from {type(source).__name__}. "
        "Please pass an explicit `execution_naming` function to scale()."
    )


def _source_label(image_id, name: str) -> str:
    """Short, human-readable label for one scaled source, for progress output.

    Path/URL entries show the input **file name** (``/data/pos1_roi2.tiff`` ->
    ``pos1_roi2.tiff``); ids and parameter dicts fall back to the execution folder
    name, which is the only thing that identifies them.
    """
    if isinstance(image_id, str):
        return Path(image_id.split("?", 1)[0]).name or name
    return name


def _scale_execute_one(
    job,
    additional_parameters,
    exist_ok,
    exist_skip,
    kernel_name,
    storage_parameter_name,
    on_stage=None,
):
    """Run every analysis script for one image entry.

    Module-level (not a closure) so it is picklable for the ``spawn`` process pool.
    ``job`` is a dict with ``image_id``, ``output_parent`` (str), ``source_parameters``
    (dict) and ``scripts`` (list of ``(src, dst)`` path-string pairs). Returns the
    list of execution records; raises ``PapermillExecutionError`` if a notebook fails.

    ``on_stage`` (if given) is called with the name of each stage notebook just
    before it is executed, so the caller can report what is running. Only used on
    the sequential path -- the parallel one runs this in a child process, from
    which the parent's progress bar is not reachable.
    """
    output_parent = Path(job["output_parent"])
    os.makedirs(output_parent, exist_ok=exist_ok)

    executions = []
    for src, dst in job["scripts"]:
        dst = Path(dst)
        # the notebook already exists and we should skip it
        if dst.exists() and exist_skip:
            continue

        if on_stage is not None:
            on_stage(dst.name)

        shutil.copy(src, dst)

        # parameters to integrate into notebook -- inject the execution folder
        # under `storage_parameter_name` unless it is None (some notebooks derive
        # their output location differently and don't declare `storage_folder`,
        # which papermill would otherwise warn about as an unknown parameter).
        parameters = {}
        if storage_parameter_name is not None:
            parameters[storage_parameter_name] = str(dst.parent.absolute())
        parameters.update(job["source_parameters"])
        parameters.update(additional_parameters)

        # execute the notebook (its own kernel subprocess; cwd is this process's,
        # which is safe because each worker is a separate process)
        pm.execute_notebook(
            dst,
            dst,
            parameters=parameters,
            cwd=dst.parent,
            kernel_name=kernel_name,
        )

        executions.append(dict(parameters=parameters, storage_folder=dst.parent))
    return executions


def scale(
    output_path: Path,
    analysis_script: Path | list[Path],
    image_ids: list[int | str | dict],
    additional_parameters=None,
    exist_ok=False,
    execution_naming=None,
    exist_skip=False,
    kernel_name=None,
    parameter_name: str = "image_id",
    max_workers: int = 1,
    storage_parameter_name: str | None = "storage_folder",
):
    """Scale an analysis notebook to several image sources.

    Each entry in ``image_ids`` identifies one image source and triggers one
    notebook execution. An entry may be:

    * an ``int`` -- e.g. an OMERO image id (default folder ``execution_<id>``),
    * a ``str`` -- a local path or fsspec URL such as ``smb://host/share/x.tif``
      (default folder name is the file stem, e.g. ``x``),
    * a ``dict`` -- arbitrary parameters merged into the notebook; provide an
      explicit ``execution_naming`` for these.

    The identifier is injected into the notebook under ``parameter_name``
    (default ``"image_id"``), so existing notebooks keep working. The notebook is
    responsible for turning it into a concrete source (e.g.
    ``OmeroSequenceSource(image_id)`` or ``SambaSequenceSource.from_url(image_id)``).

    **Hint:** the analysis script should only use absolute paths as the file is copied and executed in another folder.

    Args:
        output_path (Path): the general output path to the storage
        analysis_script (Path): the template script
        image_ids (list[int | str | dict]): image sources to scale over (ids,
            paths/URLs, or parameter dicts).
        additional_parameters (dict): Parameters to be inserted into the jupyter script
        exist_ok (Bool): True when it is okay that the directory exists, False will throw an error when the directory exists.
        execution_naming (Callable): maps an entry to its output folder name. By
            default :func:`default_execution_naming` is used, which dispatches on
            the entry type (id -> ``execution_<id>``, path -> file stem).
        exist_skip (Bool): If true existing executions are skipped.
        kernel_name (str): specifies the notebook kernel to be used. None is the default kernel.
        parameter_name (str): name of the notebook parameter the identifier is
            injected as (ignored for ``dict`` entries, which are merged as-is).
        storage_parameter_name (str | None): the notebook parameter the per-run
            output/execution folder (absolute) is injected under. Defaults to
            ``"storage_folder"``. Set it to match the notebook's own output
            parameter, or to ``None`` to not inject it at all (avoids papermill's
            "Passed unknown parameter" warning for notebooks that derive their
            output location some other way).
        max_workers (int): how many notebooks to execute concurrently. ``1``
            (default) runs them sequentially, exactly as before. Values > 1 run
            that many notebooks in parallel using a **process** pool started with
            the ``"spawn"`` method (not threads: papermill sets the working
            directory with a process-global ``os.chdir`` that threads would race
            on; and not ``fork``: spawning fresh processes avoids duplicating a
            CUDA-initialised parent kernel -- the classic Jupyter crash). Each
            execution is still its own kernel subprocess, so a worker whose kernel
            dies only fails its own image. On a **single GPU** keep this small
            (2-3): every concurrent run loads its own model, so throughput is
            bounded by GPU memory, not CPU cores.
    """

    if execution_naming is None:
        execution_naming = default_execution_naming

    if isinstance(analysis_script, str):
        # if this is just a single string, then we make it a list of a single path
        analysis_script = [Path(analysis_script)]
    elif isinstance(analysis_script, Path):
        analysis_script = [analysis_script]
    elif isinstance(analysis_script, Iterable):
        analysis_script = list(map(Path, analysis_script))

    for script in analysis_script:
        if not script.exists():
            raise ValueError(f"Analysis script {script} does not exist!")

    if additional_parameters is None:
        additional_parameters = {}

    experiment_executions = []

    failed_ids: list[int | str | dict] = []

    # warn (don't fail) if two entries map to the same output folder, since later
    # executions would otherwise silently overwrite earlier ones
    seen_names: dict[str, object] = {}
    for entry in image_ids:
        if isinstance(entry, dict):
            continue  # naming for dicts is user-defined; skip the heuristic check
        name = execution_naming(entry)
        if name in seen_names:
            logger.warning(
                "Execution name %r maps to multiple sources (%r and %r); their "
                "outputs will collide. Pass a unique `execution_naming` to avoid this.",
                name,
                seen_names[name],
                entry,
            )
        seen_names[name] = entry

    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    # Resolve naming + per-entry parameters up front (in this process, so a lambda
    # `execution_naming` still works). Each job is fully picklable for the pool.
    jobs: list[dict[str, Any]] = []
    for image_id in image_ids:
        # per-entry notebook parameters: dict entries are merged as-is,
        # everything else is injected under `parameter_name`.
        if isinstance(image_id, dict):
            source_parameters = dict(image_id)
        else:
            source_parameters = {parameter_name: image_id}

        name = execution_naming(image_id)
        output_parent = output_path / name
        jobs.append(
            {
                "image_id": image_id,
                "label": _source_label(image_id, name),
                "output_parent": str(output_parent),
                "source_parameters": source_parameters,
                "scripts": [
                    (str(s), str(output_parent / s.name)) for s in analysis_script
                ],
            }
        )

    if max_workers == 1:
        # sequential (default) -- the bar reports the notebook currently running
        # and the source it runs on, e.g. "02_Track.ipynb | pos001_roi002.tiff".
        with tqdm(jobs, unit="source") as pbar:
            for job in pbar:
                pbar.set_description(job["label"])
                try:
                    experiment_executions.extend(
                        _scale_execute_one(
                            job,
                            additional_parameters,
                            exist_ok,
                            exist_skip,
                            kernel_name,
                            storage_parameter_name,
                            on_stage=lambda stage, label=job["label"]: (
                                pbar.set_description(f"{stage} | {label}")
                            ),
                        )
                    )
                except pm.PapermillExecutionError:
                    failed_ids.append(job["image_id"])
    else:
        # Run several notebooks in parallel. A *process* pool (spawn) is required,
        # not threads: papermill sets the working directory with a process-global
        # os.chdir, which threads would race on (mislocating each run's relative
        # outputs). spawn also avoids duplicating a CUDA-initialised parent kernel.
        # Failures are collected here in the parent as each future completes.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(
                    _scale_execute_one,
                    job,
                    additional_parameters,
                    exist_ok,
                    exist_skip,
                    kernel_name,
                    storage_parameter_name,
                ): job
                for job in jobs
            }
            # several sources run at once, so "currently running" is not a single
            # thing: report the one that just finished (and whether it failed)
            # rather than pretending there is one active source.
            with tqdm(total=len(futures), unit="source") as pbar:
                for future in as_completed(futures):
                    job = futures[future]
                    status = "done"
                    try:
                        experiment_executions.extend(future.result())
                    except pm.PapermillExecutionError:
                        failed_ids.append(job["image_id"])
                        status = "FAILED"
                    pbar.set_description(f"{status} {job['label']}")
                    pbar.update(1)

    if len(failed_ids) > 0:
        error_ratio = len(failed_ids) / len(image_ids) * 100

        logging.warning(
            "The scaling failed in %d/%d (%.3f%%) executions. Please report failes with the link to the script and the image id to your administrator in order to further improve the software.",
            len(failed_ids),
            len(image_ids),
            error_ratio,
        )
        if error_ratio > 10:
            # error rates of more than 10% are definitively acceptable
            logging.error("Such a high error rate is not acceptable!")

    return experiment_executions


def extract_growth(
    overlay: Overlay,
    images: ImageSequenceSource,
    *,
    time_unit: str = "hour",
    agg: AggMode = "sum",
) -> tuple[pd.DataFrame, GrowthRateResult, Figure]:
    """Single-cell table + log-linear growth-rate fit in one call.

    Convenience wrapper combining single-cell extraction and the growth-rate fit
    (the last two steps of a typical time-lapse pipeline). Builds a per-cell table
    with ``frame`` + physical ``time`` (in ``time_unit``) + physical ``area``
    columns via :class:`ExtractorExecutor`, then fits
    ``area ~ exp(growth_rate * time)`` aggregated per timepoint by ``agg`` with
    :func:`~acia.analysis.growth_rate.estimate_growth_rate`.

    Args:
        overlay: the (already filtered) contours to measure.
        images: the calibrated image source (provides ``pixel_size`` for area and
            ``timepoints`` for time).
        time_unit: output unit for the time column and growth rate.
        agg: per-timepoint aggregation of the value column (e.g. ``"sum"`` for
            total area, ``"count"`` for cell number).

    Returns:
        ``(table, result, figure)`` -- the single-cell ``DataFrame``, the
        :class:`~acia.analysis.growth_rate.GrowthRateResult`, and the fit
        ``matplotlib`` figure.
    """
    table = ExtractorExecutor().execute(
        overlay,
        images,
        [FrameEx(), TimeEx(output_unit=time_unit), AreaEx()],
        units="none",
    )
    result, figure = estimate_growth_rate(
        table, time_col="time", value_col="area", agg=agg
    )
    return table, result, figure
