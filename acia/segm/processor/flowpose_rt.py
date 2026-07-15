"""flowpose-rt segmentation implementation"""

import numpy as np
from tqdm.auto import tqdm

from acia.attribute import attribute_segmentation
from acia.base import ImageSequenceSource, Overlay
from acia.segm.formats import overlay_from_masks

from . import SegmentationProcessor


def _batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


class FlowposeRTSegmenter(SegmentationProcessor):
    """flowpose-rt segmentation implementation (omnipose-compatible, lighter deps)"""

    def __init__(
        self,
        model="bact_phase_omni",
        device="auto",
        precision="auto",
        compile=None,  # noqa: A002 - mirrors flowpose_rt.Segmenter's own kwarg name
        autorelease: bool = True,
        batch_size: int = 20,
    ):
        super().__init__(autorelease=autorelease)
        self.model_spec = model
        self.device = device
        self.precision = precision
        self.compile = compile
        self.batch_size = batch_size

    def _load_model(self):
        import flowpose_rt as ort

        return ort.Segmenter(
            model=self.model_spec,
            device=self.device,
            precision=self.precision,
            compile=self.compile,
        )

    def _segment(self, images: ImageSequenceSource) -> Overlay:
        imgs = []
        for image in images:
            raw_image = image.raw

            # Reduce HxWxC=1 image to HxW shape
            if len(raw_image.shape) == 3:
                if raw_image.shape[2] != 1:
                    raise ValueError(
                        f"FlowposeRTSegmenter only accepts a single channel image. Currently it is HxWxC: {raw_image.shape}"
                    )

                # make it a grayscale image
                raw_image = raw_image[..., 0]

            imgs.append(raw_image)

        # one vectorized net forward per batch (not per image) -- batch_size
        # trades progress granularity for throughput; a batch of 1 shows
        # per-image progress but loses flowpose-rt's batched-forward speedup
        all_masks = []
        pbar = tqdm(total=len(imgs), desc="Batched flowpose-rt prediction...")
        for image_batch in _batch(imgs, self.batch_size):
            stack = np.stack(image_batch)
            all_masks.append(self.model.segment(stack))
            pbar.update(len(image_batch))

        masks = np.concatenate(all_masks)

        ov = overlay_from_masks(masks)

        attribute_segmentation(ov, self)

        return ov
