"""Offline image processors to perform segmentation"""

import logging

import cv2
import numpy as np
import torch
import tqdm.auto as tqdm

from acia.base import Contour, ImageSequenceSource, Overlay, Processor

from .predict import contour_from_mask, prediction


class OfflineModel(Processor):
    """
    Model that runs on the local computer
    """

    def __init__(
        self, config_file, parameter_file, half=False, device="cuda", tiling=None
    ):
        """
        config_file: model configuration file
        parameter_file: model checkpoint file
        half: enables half-precision (16-bit) execution. A bit faster.
        device: chooses the device to execute (e.g. 'cpu' or 'cuda' or 'cuda:0')
        """
        # store file destinations
        self.config_file = config_file
        self.parameter_file = parameter_file
        # empty model instance
        self.model = None
        # half-precision execution
        self.half = half
        # determine the device
        self.device = device

        self.tiling = tiling

    def load_model(self, device=None, cfg_options=None, half=False):
        """
        Load model from definitions

        device: device type, e.g. 'cpu' or 'cuda'
        cfg_options: overwrite configuration options e.g. {'test_cfg.rpn.nms_thr': 0.7}
        """
        from mmcv.runner import wrap_fp16_model
        from mmdet.apis import init_detector

        # init model
        if self.model is None:
            self.model = init_detector(
                self.config_file,
                self.parameter_file,
                device=device,
                cfg_options=cfg_options,
            )
            if half:
                # make it 16-bit
                wrap_fp16_model(self.model)

            if "classes" in self.model.cfg:
                # update object classes from config
                self.model.CLASSES = self.model.cfg["classes"]

        return self.model

    def predict(self, source: ImageSequenceSource) -> Overlay:
        """
        Predicts the overlay for an image sequence

        source: image sequence source
        tiling: whether to enable tiling
        """
        self.load_model(
            half=self.half, device=self.device
        )  # , cfg_options={'test_cfg.rcnn.nms.iou_threshold': 0.3, 'test_cfg.rcnn.score_thr': 0.5})

        # TODO: super strange without [] it takes some other list as initialization. This leads to detected cells from other images...
        overlay = Overlay([])

        for frame_id, image in tqdm.tqdm(enumerate(source)):

            pred_result = prediction(image, self.model, tiling=self.tiling)

            if len(pred_result) == 0:
                # no predictions
                continue

            all_masks = np.stack([det["mask"] for det in pred_result])
            all_contours = [contour_from_mask(mask, 0.5) for mask in all_masks]
            # drop non-sense contours
            all_contours = list(
                filter(lambda comb: len(comb[1]) >= 5, zip(pred_result, all_contours, strict=False))
            )

            contours = [
                Contour(cont, pred["score"], frame_id, id=-1, label=pred["label"])
                for pred, cont in all_contours
            ]
            overlay.add_contours(contours)

        return overlay


class PoseModel(Processor):
    """
    Model that runs on the local computer
    """

    def __init__(
        self,
        model_name="bact_omni",
        omni=True,
        use_gpu=torch.cuda.is_available(),
        diameter=None,
        flow_threshold=None,
    ):
        """
        config_file: model configuration file
        parameter_file: model checkpoint file
        half: enables half-precision (16-bit) execution. A bit faster.
        device: chooses the device to execute (e.g. 'cpu' or 'cuda' or 'cuda:0')
        """
        self.omni = omni
        self.use_gpu = use_gpu
        self.model_name = model_name
        self.model = None

        self.diameter = diameter
        self.flow_threshold = flow_threshold

    def load_model(self):
        """
        Load model from definitions
        """
        from cellpose import models
        logging.info("Loading model %s", self.model_name)
        self.model = models.Cellpose(
            gpu=self.use_gpu, model_type=self.model_name, omni=self.omni
        )

    def predict(self, source: ImageSequenceSource) -> Overlay:
        """
        Predicts the overlay for an image sequence

        source: image sequence source
        tiling: whether to enable tiling
        """
        self.load_model()
        channels = [[0, 0]]

        # TODO: super strange without [] it takes some other list as initialization. This leads to detected cells from other images...
        overlay = Overlay([])

        for frame_id, image in tqdm.tqdm(enumerate(source)):
            try:
                masks, _, _, _ = self.model.eval(
                    [image],
                    channels=channels,
                    rescale=None,
                    diameter=70,
                    flow_threshold=0.9,
                    mask_threshold=0.0,
                    resample=True,
                    diam_threshold=100,
                )
            # TODO: more precise exception for Omnipose failure
            # pylint: disable=W0703
            except Exception:
                print("Error in OmniPose prediction")
                masks = [
                    [],
                ]

            int_mask = masks[0]

            num_cells = np.max(int_mask)
            score_threshold = 0.5
            all_contours = []

            for index in range(1, num_cells + 1):
                bool_mask = int_mask == index

                contours, _ = cv2.findContours(
                    np.where(bool_mask > score_threshold, 1, 0).astype(np.uint8),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                for contour in contours:
                    contour = np.squeeze(contour)
                    if len(contour) > 3:
                        all_contours.append(contour)

            for contour in all_contours:
                overlay.add_contour(Contour(contour, -1, frame_id, -1))

        return overlay
