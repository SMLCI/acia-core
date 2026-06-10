"""Omnipose segmentation implementation"""

from pathlib import Path

import numpy as np
import torch
from cellpose_omni import models
from tqdm.auto import tqdm

from acia.attribute import attribute_segmentation
from acia.base import ImageSequenceSource, Overlay
from acia.segm.formats import overlay_from_masks

from . import SegmentationProcessor


def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


class OmniposeSegmenter(SegmentationProcessor):
    """Omnipose segmentation implementation"""

    def __init__(
        self,
        use_GPU: bool | None = None,
        model="bact_phase_omni",
        autorelease: bool = True,
    ):
        super().__init__(autorelease=autorelease)
        if use_GPU is None:
            use_GPU = torch.cuda.is_available()
        self.use_GPU = use_GPU
        self.model_spec = model

    def _load_model(self):
        model = self.model_spec

        model_type = None
        model_path = None

        if Path(model).exists() and Path(model).is_file():
            model_path = model
        elif model in models.MODEL_NAMES:
            model_type = model
        else:
            raise ValueError(
                "Specified model is neither predefined nor a url to download"
            )

        if model_type:
            return models.CellposeModel(gpu=self.use_GPU, model_type=model_type)
        return models.CellposeModel(
            gpu=self.use_GPU, pretrained_model=model_path, nclasses=3, nchan=2
        )

    @staticmethod
    def __predict(
        images, model, omnipose_parameters: dict | None = None, batch_size=20
    ):
        if omnipose_parameters is None:
            omnipose_parameters = {}

        chans = [0, 0]  # this means segment based on first channel, no second channel

        # define parameters
        mask_threshold = -1
        verbose = 0  # turn on if you want to see more output
        transparency = True  # transparency in flow output
        rescale = (
            None  # give this a number if you need to upscale or downscale your images
        )
        omni = True  # we can turn off Omnipose mask reconstruction, not advised
        flow_threshold = 0.4  # default is .4, but only needed if there are spurious masks to clean up; slows down output
        resample = (
            True  # whether or not to run dynamics on rescaled grid or original grid
        )
        cluster = True  # use DBSCAN clustering

        all_masks = []

        pbar = tqdm(
            total=len(images),
            desc="Batched Omnipose prediction...",
        )

        for image_batch in batch(images, n=batch_size):
            # Make evaluation (flows and styles are not needed)
            masks, _, _ = model.eval(
                image_batch,
                channels=chans,
                rescale=rescale,
                mask_threshold=mask_threshold,
                transparency=transparency,
                flow_threshold=flow_threshold,
                omni=omni,
                cluster=cluster,
                resample=resample,
                verbose=verbose,
                model_loaded=True,
                show_progress=False,
                **omnipose_parameters,
            )

            all_masks.append(masks)

            pbar.update(len(image_batch))

        return np.concatenate(all_masks)

    def predict(self, images: ImageSequenceSource) -> Overlay:
        return self(images)

    def _segment(
        self, images: ImageSequenceSource, omnipose_parameters: dict | None = None
    ) -> Overlay:
        imgs = []
        for image in images:
            raw_image = image.raw

            # Reduce HxWxC=1 image to HxW shape
            if len(raw_image.shape) == 3:
                if raw_image.shape[2] != 1:
                    raise ValueError(
                        f"Omnipose Segmenter only accepts a single channel image. Currently it is HxWxC: {raw_image.shape}"
                    )

                # make it a grayscale image
                raw_image = raw_image[..., 0]

            imgs.append(raw_image)

        masks = self.__predict(
            imgs, self.model, omnipose_parameters=omnipose_parameters
        )

        ov = overlay_from_masks(masks)

        attribute_segmentation(ov, self)

        return ov
