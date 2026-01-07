"""PyUAT based tracking"""

import gzip
import tempfile
import time
import warnings
from pathlib import Path

from uatrack.config import setup_assignment_generators
from uatrack.core import simpleTracking
from uatrack.utils import extract_single_cell_information, save_tracking

from acia.attribute import attribute_tracking
from acia.base import ImageSequenceSource, Overlay
from acia.segm.formats import overlay_from_masks
from acia.tracking.formats import parse_simple_tracking
from acia.tracking.processor.utils import overlay_to_masks

from . import TrackingProcessor

TRACKING_CONFIGURATIONS = ["NN", "FO", "FO+G", "FO+O", "FO+DD", "FO+G+O+DD"]


class PyUATTracker(TrackingProcessor):
    """Processor for PyUAT: https://arxiv.org/abs/2503.21914"""

    def __init__(
        self,
        tracking_configuration: str,
        subsampling_factor=1,
        num_particles=1,
        num_cores=1,
        max_num_hypotheses=1,
        cutOff=-1,
        max_num_solutions=1,
        mip_method="CBC",
    ):
        """
        Args:
            tracking_configuration (str): PyUAT tracking configuration
        """

        if tracking_configuration not in TRACKING_CONFIGURATIONS:
            raise ValueError(
                f"'{tracking_configuration}' is not a valid tracking configuration. Pleas one of {TRACKING_CONFIGURATIONS}!"
            )

        self.config = tracking_configuration

        self.subsampling_factor = subsampling_factor

        self.num_particles = num_particles
        self.num_cores = num_cores
        self.max_num_hypotheses = max_num_hypotheses
        self.cutOff = cutOff
        self.max_num_solutions = max_num_solutions

        self.mip_method = mip_method

        if mip_method != "GRB":
            warnings.warn(
                "You are not using Gurobi! Please install gurobi and specify 'GRB' as optimizer for a tremendous speedup!"
            )

    def __call__(self, images: ImageSequenceSource, segmentation: Overlay):
        print("Extract single-cell information...")
        df, all_detections = extract_single_cell_information(segmentation)

        print("Setup assignment generators...")
        assignment_generators = setup_assignment_generators(
            df, self.subsampling_factor, self.config
        )

        print("Perform tracking...")
        # start tracking
        start = time.time()
        res = simpleTracking(
            df,
            assignment_generators,
            self.num_particles,
            num_cores=self.num_cores,
            max_num_hypotheses=self.max_num_hypotheses,
            cutOff=self.cutOff,
            max_num_solutions=self.max_num_solutions,
            mip_method=self.mip_method,  # use "GRB" if you have gurobi installed
        )
        end = time.time()

        print("time for tracking", end - start)

        with tempfile.TemporaryDirectory() as td:
            output_file = Path(td) / "simpleTracking.json.gz"
            save_tracking(res[0], all_detections, output_file)

            # read the tracking result
            with gzip.open(output_file) as input_file:
                tracking_overlay, tracking_graph = parse_simple_tracking(
                    input_file.read()
                )

        # Convert from contour based overlay to a mask based overlay
        height, width = images.get_frame(0).raw.shape[:2]
        mask_stack = overlay_to_masks(tracking_overlay, height, width)

        tracking_ov_new = overlay_from_masks(mask_stack)

        attribute_tracking(tracking_ov_new, None, tracking_graph, self)

        # TODO: create tracklet graph
        return tracking_ov_new, None, tracking_graph
