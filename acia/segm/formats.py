"""Functions for different segmentation formats"""

import json
from pathlib import Path

import numpy as np
from tifffile import imread

from acia.base import Contour, Instance, Overlay
from acia.utils import multi_mask_to_polygons


def parse_simple_segmentation(file_content: str) -> Overlay:
    """Parse simple segmentation format from string (json)

    Args:
        file_content (str): simple segmentation file content as string

    Returns:
        Overlay: the overlay representation of the segmentation
    """

    file_data = json.loads(file_content)

    contours = []

    for frame in file_data:
        frame_id = frame["frame"]
        for det in frame["detections"]:
            contours.append(
                Contour(det["contour"], -1.0, frame_id, det["id"], det["label"])
            )

    return Overlay(contours)


def gen_simple_segmentation(overlay: Overlay) -> str:
    """Create a simple segmentation string from an Overlay

    Args:
        overlay (Overlay): the overlay to store

    Returns:
        str: string containing the stringified simple segmentation json format
    """

    frame_packages = []

    # loop over frames
    for frame_overlay in overlay.timeIterator():
        if len(frame_overlay) == 0:
            continue

        det_objects = []
        frame_id = -1

        # loop over all contours in a frame
        for cont in frame_overlay:
            # transform coordinates to list (otherwise not json serializable)
            coordinates = cont.coordinates
            if isinstance(coordinates, np.ndarray):
                coordinates = coordinates.tolist()

            # create detection object
            det_objects.append(
                dict(
                    label=cont.label,
                    contour=coordinates,
                    id=cont.id,
                )
            )

            frame_id = cont.frame

        # create the frame package
        frame_package = dict(
            frame=frame_id,
            detections=det_objects,
        )
        frame_packages.append(frame_package)

    # serialize into json format
    return json.dumps(frame_packages, separators=(",", ":"))


def load_ctc_segmentation(segmentation_path: Path) -> Overlay:
    segmentation_path = Path(segmentation_path)

    segm_mask_files = sorted(segmentation_path.glob("*.tif"))

    overlay = Overlay([], frames=list(range(len(segm_mask_files))))

    c_id = 0

    for frame_id, segm_file in enumerate(segm_mask_files):
        polygons = multi_mask_to_polygons(imread(segm_file))

        for _, poly in polygons:
            points = np.array(poly.exterior.coords.xy)
            overlay.add_contour(Contour(points, -1, frame_id, c_id, "cell"))
            c_id += 1

    return overlay


def read_ctc_segmentation_native(segmentation_path: Path) -> Overlay:
    """Fast loading of CTC segmentation masks into an Overlay

    Args:
        segmentation_path (Path): Path to the folder containing all the *.tif masks

    Returns:
        Overlay: Overlay containing all masks
    """

    # List all the segmentation masks
    segmentation_path = Path(segmentation_path)
    segm_mask_files = sorted(segmentation_path.glob("*.tif"))

    segm_masks = [imread(segm_file) for segm_file in segm_mask_files]

    return overlay_from_masks(segm_masks)


def overlay_from_masks(segm_masks: np.ndarray) -> Overlay:
    """Create a multi-frame overlay from an array of masks

    Args:
        segm_masks (np.ndarray): mask array [T x H x W]

    Returns:
        Overlay: returns the multi-frame overly with cell instances
    """
    overlay = Overlay([], frames=list(range(len(segm_masks))))

    # unique id for instances
    uid = 1

    # Iterate all the mask files
    for frame_id, mask in enumerate(segm_masks):
        # Find all cell labels (except 0)
        labels = np.unique(mask)[1:]

        # for every label create an instance and add it to the contour
        for label in labels:
            instance = Instance(mask=mask, frame=frame_id, label=label, id=uid)
            overlay.add_contour(instance)
            uid += 1

    return overlay
