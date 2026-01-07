"""Trackastra based tracking"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc

from acia.attribute import attribute_tracking
from acia.base import ImageSequenceSource, Overlay
from acia.segm.formats import read_ctc_segmentation_native
from acia.tracking import ctc_track_graph
from acia.tracking.formats import read_ctc_tracklet_graph

from . import TrackingProcessor
from .utils import overlay_to_masks


class TrackastraTracker(TrackingProcessor):
    """Processor for Trackastra: https://doi.org/10.48550/arXiv.2405.15700"""

    def __init__(self, mode="greedy"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load a pretrained model
        self.model = Trackastra.from_pretrained("general_2d", device=device)
        self.mode = mode

    def __call__(self, images: ImageSequenceSource, segmentation: Overlay):
        image = next(iter(images)).raw
        height, width = image.shape[:2]

        masks = overlay_to_masks(segmentation, height=height, width=width)
        imgs = np.stack([im.raw for im in images])

        if len(imgs.shape) == 4:
            # strip off last because it should not be used
            imgs = imgs[..., 0]

        if segmentation.numFrames() != len(masks):
            logging.warning("Number of segmented frames and masks is unequal!")

        # perform the actual tracking
        track_graph = self.model.track(imgs, masks, mode=self.mode)

        # Write to cell tracking challenge format
        with tempfile.TemporaryDirectory() as td:
            _, _ = graph_to_ctc(
                track_graph,
                masks,
                outdir=td,
            )

            input_path = Path(td)
            track_file = input_path / "man_track.txt"

            ov = read_ctc_segmentation_native(input_path)
            tracklet_graph = read_ctc_tracklet_graph(track_file)

        tracking_graph = ctc_track_graph(ov, tracklet_graph)

        attribute_tracking(ov, tracklet_graph, tracking_graph, self)

        return ov, tracklet_graph, tracking_graph
