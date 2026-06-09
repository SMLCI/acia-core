"""Functionality for single-cell analysis"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterable
from functools import reduce
from itertools import starmap
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import papermill as pm
from numpy import ma
from pint._typing import UnitLike
from tqdm.auto import tqdm

from acia import Q_, U_
from acia.analysis.units import UNIT_ATTR, units_in_header
from acia.analysis.units import attach_units as attach_units
from acia.analysis.units import from_header as from_header
from acia.analysis.units import strip_units as strip_units
from acia.base import BaseImage, ImageSequenceSource, Overlay
from acia.utils import pairwise_distances

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
            # no source calibration -> use the configured defaults
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

        for extractor in tqdm(extractors):
            print(f"Extracting: {extractor.name}...")
            result_df, extractor_units = extractor.extract(overlay, images, df)

            df = pd.merge(df, result_df, on="id")

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
        data = []
        for cont in overlay:
            data.append({"id": cont.id, self.name: self.convert(cont.area)})

        df = pd.DataFrame(data).set_index("id")

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
        data = []
        for cont in overlay:
            # extract the length of the polygon
            perimeter = cont.polygon.length

            data.append({"id": cont.id, self.name: self.convert(perimeter)})

        df = pd.DataFrame(data).set_index("id")

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
        circularities = (4 * np.pi * np.array(df["area"])) / np.array(
            df["perimeter"]
        ) ** 2

        df = pd.DataFrame({self.name: circularities, "id": df.index}).set_index("id")

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
        lengths = []
        for cont in overlay:
            lengths.append(
                self.convert(
                    # longer edge of minimum roated bbox
                    np.max(
                        pairwise_distances(
                            np.array(
                                cont.polygon.minimum_rotated_rectangle.exterior.coords
                            )
                        )
                    )
                )
            )

        df = pd.DataFrame(
            {self.name: lengths, "id": [c.id for c in overlay]}
        ).set_index("id")

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
        widths = []
        for cont in overlay:
            widths.append(
                self.convert(
                    # shorter edge of bbox approximation
                    np.min(
                        # measure edge lengths of bbox approximation
                        pairwise_distances(
                            np.array(
                                # bbox approaximation
                                cont.polygon.minimum_rotated_rectangle.exterior.coords
                            )
                        )
                    )
                )
            )

        df = pd.DataFrame({self.name: widths, "id": [c.id for c in overlay]}).set_index(
            "id"
        )

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
        widths = []
        lengths = []
        for cont in overlay:
            pd_box = pairwise_distances(
                np.array(
                    # bbox approaximation
                    cont.polygon.minimum_rotated_rectangle.exterior.coords
                )
            )

            widths.append(
                self.convert(
                    # shorter edge of bbox approximation
                    np.min(
                        # measure edge lengths of bbox approximation
                        pd_box
                    )
                )
            )

            lengths.append(
                self.convert(
                    # longer edge of minimum roated bbox
                    np.max(pd_box)
                )
            )

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

        df = pd.DataFrame(data).set_index("id")

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
        positions_x = []
        positions_y = []
        ids = []
        for cont in overlay:
            positions_x.append(self.convert(cont.center[0]))
            positions_y.append(self.convert(cont.center[1]))
            ids.append(cont.id)

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

        return pd.DataFrame(data).set_index("id")

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

    for image_id in tqdm(image_ids):
        try:
            # create the main output folder
            output_parent = output_path / execution_naming(image_id)
            os.makedirs(output_parent, exist_ok=exist_ok)

            # per-entry notebook parameters: dict entries are merged as-is,
            # everything else is injected under `parameter_name`.
            if isinstance(image_id, dict):
                source_parameters = dict(image_id)
            else:
                source_parameters = {parameter_name: image_id}

            for script in analysis_script:
                # path to the new notebook file
                # every execution should have its own folder to store local files
                output_file = output_parent / script.name

                if output_file.exists() and exist_skip:
                    # the notebook exists and we should skip it
                    continue

                shutil.copy(script, output_file)

                # parameters to integrate into notebook
                parameters = dict(
                    storage_folder=str(output_file.parent.absolute()),
                    **source_parameters,
                    **additional_parameters,
                )

                # execute the notebook
                pm.execute_notebook(
                    output_file,
                    output_file,
                    parameters=parameters,
                    cwd=output_file.parent,
                    kernel_name=kernel_name,
                )

                # save experiment in list
                experiment_executions.append(
                    dict(parameters=parameters, storage_folder=output_file.parent)
                )
        except pm.PapermillExecutionError:
            failed_ids.append(image_id)

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
