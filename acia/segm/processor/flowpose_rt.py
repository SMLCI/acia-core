"""flowpose-rt segmentation implementation"""

import gc

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
    """flowpose-rt segmentation implementation (omnipose-compatible, lighter deps).

    ``batch_size`` trades memory for throughput: ``flowpose_rt.Segmenter.segment()``
    only tiles a genuinely single (H, W) image internally -- a stacked (N, H, W)
    batch (what a ``batch_size``-chunked call sends) skips that tiling and forwards
    the whole chunk through the network at full resolution, so a larger batch of
    frames bigger than flowpose-rt's ~224px tile size costs more memory per image
    than single-frame calls would. Lower ``batch_size`` if memory is a concern for
    large frames.
    """

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

    def _release_model(self) -> None:
        super()._release_model()
        try:
            import torch

            if torch.cuda.is_available():
                # torch.compile(mode="reduce-overhead") (flowpose_rt's default on
                # CUDA) caches CUDA-graph memory pools process-globally, not
                # scoped to the model instance -- dropping the model reference
                # and torch.cuda.empty_cache() (in the base class) don't reclaim
                # those pools, so reset the compile cache explicitly here.
                torch.compiler.reset()
        except Exception:
            pass

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
            # encourage prompt reclamation of this batch's activation memory
            # before the next batch starts, rather than only at end-of-call
            gc.collect()

        masks = np.concatenate(all_masks)

        ov = overlay_from_masks(masks)

        attribute_segmentation(ov, self)

        return ov
