"""Segmenter using micro-sam (μSAM): https://github.com/computational-cell-analytics/micro-sam"""

import numpy as np
from tqdm.auto import tqdm

from acia.attribute import attribute_segmentation
from acia.base import ImageSequenceSource, Overlay
from acia.segm.formats import overlay_from_masks

from . import SegmentationProcessor


class MicroSAMSegmenter(SegmentationProcessor):
    """MicroSAMSegmenter using micro-sam (μSAM), a SAM-based light-microscopy
    foundation model: https://github.com/computational-cell-analytics/micro-sam

    Uses the AIS (automatic instance segmentation) path -- a single forward
    pass through a finetuned decoder -- which is only available for the
    ``*_lm``/``*_em_organelles``/etc. finetuned checkpoints, not the generic
    natural-image SAM checkpoints. The default ``vit_b_lm`` checkpoint is a
    light-microscopy generalist; its training set includes DeepBacs (bacteria)
    alongside several eukaryotic-cell/nuclei datasets, so bacteria exposure
    exists but is not a specialization -- validate before trusting it on dense
    bacterial colonies.
    """

    def __init__(
        self,
        model_type: str = "vit_b_lm",
        device: str | None = None,
        autorelease: bool = True,
    ):
        """Initialize the micro-sam segmenter.

        Args:
            model_type: Which micro-sam checkpoint to load, e.g.
                ``"vit_b_lm"`` (light-microscopy generalist, default),
                ``"vit_l_lm"``, ``"vit_t_lm"``.
            device: Force a specific device (e.g. ``"cpu"`` or ``"cuda"``).
                ``None`` (default) lets micro-sam auto-detect.
            autorelease: Release the model after each call to free GPU memory.
        """
        super().__init__(autorelease=autorelease)
        self.model_type = model_type
        self.device = device

    def _load_model(self):
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter

        predictor, segmenter = get_predictor_and_segmenter(
            model_type=self.model_type, device=self.device
        )
        print(f"micro-sam model loaded: {self.model_type}")
        return (predictor, segmenter)

    def _segment(self, images: ImageSequenceSource) -> Overlay:
        from micro_sam.automatic_segmentation import automatic_instance_segmentation

        predictor, segmenter = self.model

        masks = []
        for image in tqdm(images, desc="micro-sam segmenting"):
            raw_image = image.raw

            # Reduce HxWxC=1 image to HxW; micro-sam's own preprocessing takes
            # care of converting grayscale to the 3-channel input its ViT
            # encoder expects.
            if len(raw_image.shape) == 3 and raw_image.shape[2] == 1:
                raw_image = raw_image[..., 0]

            label_mask = automatic_instance_segmentation(
                predictor=predictor,
                segmenter=segmenter,
                input_path=raw_image,
                ndim=2,
            )
            masks.append(label_mask)

        ov = overlay_from_masks(np.stack(masks))

        attribute_segmentation(ov, self)

        return ov
