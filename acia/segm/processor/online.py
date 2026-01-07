"""Processors to apply remote models for segmentation"""

import json
import logging
import os
from io import BytesIO
from itertools import chain, islice
from urllib.parse import urlparse

import numpy as np
import requests
import tqdm.auto as tqdm
from PIL import Image
from retry import retry

from acia.base import Contour, ImageSequenceSource, Overlay, Processor


class OnlineModel(Processor):
    """
    The model is not running locally on the computer but in a remote location
    """

    def __init__(self, url: str, username=None, password=None, timeout=30):
        """
        url: remote model executer (can also contain a port definition)
        username: username
        password: password
        """
        self.url = url
        self.username = username
        self.password = password
        self.timeout = timeout

        # try to parse port from url
        self.port = urlparse(url).port

        if not self.port:
            # autodetermine port from protocol
            scheme = urlparse(url).scheme
            if scheme == "http":
                self.port = 80
            elif scheme == "https":
                self.port = 443
            else:
                logging.warning(
                    'Could not determine port! Did you specify "http://" or "https://" at the beginning of your url?'
                )

    def predict(self, source: ImageSequenceSource, params=None):
        if params is None:
            params = {}

        contours = []

        # iterate over images from image source
        for frame, image in enumerate(tqdm.tqdm(source)):

            # convert image into a binary png stream
            byte_io = BytesIO()
            Image.fromarray(image).save(byte_io, "png")
            byte_io.seek(0)

            # pack this into form data
            multipart_form_data = {
                "data": ("data.png", byte_io, "image/png"),
            }

            # send a request to the server
            response = requests.post(
                self.url, files=multipart_form_data, params=params, timeout=self.timeout
            )

            # raise an error if the response is not as expected
            if response.status_code != 200:
                raise ValueError(
                    f"HTTP request for prediction not successful. Status code: {response.status_code}"
                )

            body = response.json()

            content = body

            for detection in content:
                # label = detection['label']
                contour_lists = detection["contours"][0]
                contour = list(zip(contour_lists["x"], contour_lists["y"], strict=False))
                score = detection["score"]

                contours.append(Contour(contour, score, frame, -1))

        return Overlay(contours)

    @staticmethod
    def parseContours(response_body) -> list[Contour]:
        pass


class ModelDescriptor:
    """Describing all parameters to run a segmentation model from a git repository"""

    def __init__(self, repo: str, entry_point: str, version: str, parameters=None):
        if parameters is None:
            parameters = {}

        self.repo = repo
        self.entry_point = (entry_point,)
        self.version = version
        self.parameters = parameters


def batch(iterable, size):
    """Make iterable over packages of a certain size

    Args:
        iterable (_type_): source iterable
        size (_type_): size of the batches

    Yields:
        _type_: iterable over individual batches
    """

    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        try:
            logging.debug("Next batch...")
            # get first element to known when iterator is at end (will throw StopIteration)
            first_element = next(batchiter)
            if size == 1:
                # single sized batches shall also be lists
                yield [first_element]
            else:
                # chain longer batches
                yield chain([first_element], batchiter)
        except StopIteration:
            # we have reached the end of the iterator
            # --> leave loop
            return


class FlexibleOnlineModel(Processor):
    """
    The model is not running locally on the computer but in a remote location
    """

    def __init__(
        self, executorUrl: str, modelDesc: ModelDescriptor, timeout=600, batch_size=1
    ):
        self.url = executorUrl
        self.timeout = timeout
        self.modelDesc = modelDesc
        self.batch_size = batch_size

        # try to parse port from url
        self.port = urlparse(self.url).port

        if not self.port:
            # autodetermine port from protocol
            scheme = urlparse(self.url).scheme
            if scheme == "http":
                self.port = 80
            elif scheme == "https":
                self.port = 443
            else:
                logging.warning(
                    'Could not determine port! Did you specify "http://" or "https://" at the beginning of your url?'
                )

        # get username from environment for statistics
        # 1. try to get jupyter user, then try to readout other usernames
        self.username = (
            os.environ.get("JUPYTERHUB_USER", None)
            or os.environ.get("USER", None)
            or os.environ.get("USERNAME", None)
        )

    def predict(self, source: ImageSequenceSource, params=None):
        if params is None:
            params = {}

        contours = []

        # Segmentation method parameters
        additional_parameters = dict(self.modelDesc.parameters)
        additional_parameters.update(**params)

        # http request parameters
        params = dict(
            repo=self.modelDesc.repo,
            entry_point=self.modelDesc.entry_point,
            version=self.modelDesc.version,
            username=self.username,
            parameters=json.dumps(additional_parameters),
        )

        # collect all the frame indices that we use for prediction
        all_frames = []

        # Do batch prediction
        for local_batch in batch(enumerate(tqdm.tqdm(source)), self.batch_size):
            local_batch = np.array(list(local_batch), dtype=object)

            frames = local_batch[:, 0]
            all_frames += list(frames)
            images = map(lambda img: img.raw, local_batch[:, 1])

            contours += self.predict_batch(frames, images, params)

        for i, cont in enumerate(contours):
            cont.id = i

        # create new overlay based on all contours and frame numbers
        return Overlay(contours, frames=all_frames)

    def predict_single(self, frame_id, image, params):
        contours = []

        # convert image into a binary png stream
        byte_io = BytesIO()
        Image.fromarray(image).save(byte_io, "png")
        byte_io.seek(0)

        # pack this into form data
        multipart_form_data = {
            "file": ("data.png", byte_io, "image/png"),
        }

        # send a request to the server
        response = requests.post(
            self.url, files=multipart_form_data, params=params, timeout=self.timeout
        )

        # raise an error if the response is not as expected
        if response.status_code != 200:
            raise ValueError(
                f"HTTP request for prediction not successful. Status code: {response.status_code}"
            )

        body = response.json()

        if body["format_version"] != "0.2":
            logging.warning(
                "Using segmentation approach with unsupported version number!"
            )

        content = body["segmentation_data"][0]

        for detection in content:
            # label = detection['label']
            contour = np.array(detection["contour_coordinates"], dtype=np.float32)
            score = -1.0

            contours.append(Contour(contour, score, frame_id, -1))

        return contours

    @retry(requests.exceptions.ReadTimeout, tries=3)
    def predict_batch(self, frame_ids: list[int], images: list, params) -> Overlay:
        """Predict segmentation for a batch of frames

        Args:
            frame_ids (List[int]): ids of the frames
            images (List): numpy array for every image
            params ([type]): dictionary of additional parameters

        Raises:
            ValueError: [description]

        Returns:
            [Overlay]: An overlay containing the segmentation information for the images
        """
        contours = []

        binary_images = []

        for image in images:
            # convert image into a binary png stream
            byte_io = BytesIO()
            Image.fromarray(image).save(byte_io, "png")
            byte_io.seek(0)

            binary_images.append(byte_io)

        multipart_form_data = [
            ("files", ("data.png", bin_image, "image/png"))
            for bin_image in binary_images
        ]

        # send a request to the server
        response = requests.post(
            self.url, files=multipart_form_data, params=params, timeout=self.timeout
        )

        # raise an error if the response is not as expected
        if response.status_code != 200:
            raise ValueError(
                f"HTTP request for prediction not successful. Status code: {response.status_code}"
            )

        body = response.json()

        if body["format_version"] != "0.2":
            logging.warning(
                "Using segmentation approach with unsupported version number!"
            )

        for i, frame_id in enumerate(frame_ids):
            content = body["segmentation_data"][i]
            for detection in content:
                # label = detection['label']
                contour = np.array(detection["contour_coordinates"], dtype=np.float32)
                score = -1.0

                contours.append(Contour(contour, score, frame_id, -1))

        logging.debug("Finished batch prediction")

        return contours

    @staticmethod
    def parseContours(response_body) -> list[Contour]:
        pass
