"""Segmenter using StarDist: https://github.com/stardist/stardist"""

import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from tqdm.auto import tqdm

from acia.attribute import attribute_segmentation
from acia.base import ImageSequenceSource, Overlay
from acia.segm.formats import overlay_from_masks

from . import SegmentationProcessor


class StarDistSegmenter(SegmentationProcessor):
    """StarDistSegmenter using StarDist: https://github.com/stardist/stardist

    StarDist's shape prior is star-convex, which fits round cells/nuclei well
    but is a structurally poor match for elongated rod-shaped bacteria -- no
    bacteria-specific pretrained model exists (every registered 2D model is
    trained on roundish/nuclear targets). Include it in a comparison as a
    contrasting baseline, not as a primary bacterial segmenter.
    """

    def __init__(
        self,
        pretrained_model: str = "2D_versatile_fluo",
        autorelease: bool = True,
    ):
        """Initialize the StarDist segmenter.

        Args:
            pretrained_model: Name of the registered pretrained 2D StarDist
                model to load, e.g. ``"2D_versatile_fluo"``,
                ``"2D_versatile_he"`` (H&E/RGB brightfield), or
                ``"2D_paper_dsb2018"``.
            autorelease: Release the model after each call to free GPU memory.
        """
        super().__init__(autorelease=autorelease)
        self.pretrained_model = pretrained_model

    def _load_model(self):
        model = StarDist2D.from_pretrained(self.pretrained_model)
        print(f"StarDist model loaded: {self.pretrained_model}")
        return model

    def _segment(self, images: ImageSequenceSource) -> Overlay:
        masks = []
        for image in tqdm(images, desc="StarDist segmenting"):
            raw_image = image.raw

            # Reduce HxWxC=1 image to HxW; StarDist2D's versatile-fluo/dsb2018
            # models expect single-channel grayscale input.
            if len(raw_image.shape) == 3 and raw_image.shape[2] == 1:
                raw_image = raw_image[..., 0]

            labels, _details = self.model.predict_instances(normalize(raw_image))
            masks.append(labels)

        ov = overlay_from_masks(np.stack(masks))

        attribute_segmentation(ov, self)

        return ov
