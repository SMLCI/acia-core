""" Helper file to generate contours from masks"""

import logging

import cv2
import mmcv
import numpy as np
import rtree
import torch
from mmdet.apis import inference_detector
from shapely.geometry import LineString, Polygon

logger = logging.getLogger(__name__)


def contour_from_mask(mask, score_threshold):
    """
    Estimate largest contour from pixel-wise mask
    """
    contours, _ = cv2.findContours(
        np.where(mask > score_threshold, 1, 0).astype(np.uint8),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # select largest contour
    selected_contour = []
    for cont in contours:
        if len(cont) > len(selected_contour):
            selected_contour = cont

    return np.squeeze(selected_contour)


def prepare_contours(
    segm_result, labels, offset_x=0, offset_y=0, seg_score_threshold=0.3
):
    offset = np.array([offset_x, offset_y])

    all_contours = []
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)

        for seg in segms:
            seg = seg.astype(np.float32)

            # Creating kernel
            # kernel = np.ones((3, 3), np.uint8)
            # find contours with cv2
            contours, _ = cv2.findContours(
                np.where(seg > seg_score_threshold, 1, 0).astype(np.uint8),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            selected_contour = []
            for cont in contours:
                if len(cont) > len(selected_contour):
                    selected_contour = cont

            cont_data = [
                (np.ceil(cont).astype(np.int32).squeeze(axis=1) + offset[None, :])
                for cont in [selected_contour]
                if len(cont) > 0
            ]  # [[str(x), str(y)] for cont in contours for x,y in cont]

            logging.info("Num contours: %d", len(contours))

            all_contours.append(cont_data)

    return all_contours


def postprocess(output_data, model, offset_x=0, offset_y=0, contours=False):
    segm_result = None

    if model.with_mask:
        bbox_result, segm_result = output_data
    else:
        bbox_result = output_data

    output_contours = (segm_result is not None) and contours

    bboxes = np.vstack(bbox_result)

    # print("Num bounding boxes:")

    # identify the labels of the detected boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    # prepare contours if necessary
    if output_contours:
        contours = prepare_contours(segm_result, labels, offset_x, offset_y)
        logging.info("Num contours: %d", len(contours))
    else:
        contours = [[]] * len(bboxes)

    result = []

    for box, label, contour, segm in zip(
        bboxes, labels, contours, mmcv.concat_list(segm_result), strict=False
    ):
        if output_contours and len(contour) == 0:
            # skip this detection
            continue

        box_coords = list(map(float, box[:4]))
        score = float(box[4])
        label_name = model.CLASSES[label]

        # print("Class '%s' detected at %s with score %.2f" % (label_name, box_coords, score))

        result_dict = {
            "label": label_name,
            "bbox": box_coords,
            "score": score,
            "mask": segm,
        }

        # sort descending
        if output_contours:
            result_dict["contours"] = sorted(
                ({"x": cont[:, 0], "y": cont[:, 1]} for cont in contour),
                key=lambda cont: -len(cont["x"]),
            )

        result.append(result_dict)  # 'contours': contour

    return result


def tile_touch_filter(detection, image_tile_poly: LineString, threshold=10):
    """
    returns False iff detections it too close (<threshold) to the boundaries and likely to be restricted by them

    detection: the detection dict
    image_tile_poly: Line string of the image tile boundaries
    threshold: minimum allowed distance to the boundaries
    """

    contour = detection["contours"][0]
    detection_poly = Polygon(zip(contour["x"], contour["y"], strict=False))

    distance = detection_poly.distance(image_tile_poly)

    # print(distance)

    return distance >= threshold


def inference(image, model, offset_x=0, offset_y=0):
    # inference on tile
    raw_tile_results = inference_detector(model, image)

    # postprocess
    tile_results = postprocess(
        raw_tile_results, model, offset_x=offset_x, offset_y=offset_y
    )

    return tile_results


# TODO: make this function shorter
# pylint: disable=R0915
def tiled_inference(
    image,
    model,
    x_shift=256 - 128,
    y_shift=256 - 128,
    tile_width=256,
    tile_height=256,
    pd=25,
):
    """
    Execute inference in a tiled fashion

    x_shift: shift on the x-axis for every image slot
    y_shift: shift on the y-axis for every image slot
    tile_width: width of the image tile
    tile_height: height of the image tile

    TODO: When tiles align with image borders, we should not do tile touch filtering
    """

    # get the image dimensions
    height, width = image.shape[:2]

    x_start = 0
    y_start = 0

    all_detections = []

    # padding the image (rgb)
    padding_size = pd
    padded_image = np.zeros(
        (height + 2 * padding_size, width + 2 * padding_size, 3), dtype=np.uint8
    )
    padded_image[
        padding_size : padding_size + height, padding_size : padding_size + width
    ] = image
    orig_image = image
    image = padded_image

    height, width = image.shape[:2]

    # iterate over top coordinate of tile
    ys = list(range(max(1, 1 + int(np.ceil((height - tile_height) / y_shift)))))
    for iY in ys:
        y = y_start + iY * y_shift
        # iterate over left coordinate of tile
        xs = list(range(max(1, 1 + int(np.ceil((width - tile_width) / x_shift)))))
        for iX in xs:
            x = x_start + iX * x_shift

            # print(x,y)

            # compute the lower right coordinates of the tile
            y_end = min(height, y + tile_height)
            x_end = min(width, x + tile_width)

            # print(x_end, y_end)
            # print(y_end-y, x_end -x)
            # get the image tile
            image_tile = image[y:y_end, x:x_end]
            # print(image_tile.shape)
            # zero padding to constant tile size (otherwise we get devision errors)
            const_tile_format = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            const_tile_format[: y_end - y, : x_end - x] = image_tile

            # print(const_tile_format.shape)

            tile_results = inference(const_tile_format, model, x, y)

            tile_results = list(
                filter(lambda det: np.sum(det["mask"]) >= 3, tile_results)
            )

            if len(tile_results) > 0:
                filter_mask = mask_nms(
                    np.stack([det["mask"] for det in tile_results]),
                    np.stack([det["bbox"] for det in tile_results]),
                    np.stack([det["score"] for det in tile_results]),
                )

                tile_results = list(np.array(tile_results)[filter_mask])

            # print(len(tile_results))

            # polygon for the image tile
            # image_tile_poly = LineString([(x, y), (x+tile_width, y), (x+tile_width, y+tile_height), (x, y+tile_height), (x,y)])

            # filter the detections
            #   -> no detections close to the border of the tile schould be considered
            filter_mask = np.ones(len(tile_results), dtype=bool)

            for i, det in enumerate(tile_results):
                det_mask = det["mask"]

                row, col = np.nonzero(det_mask)

                miny = np.min(row)
                maxy = np.max(row)
                minx = np.min(col)
                maxx = np.max(col)

                min_distance = np.min(
                    [miny, minx, tile_height - maxy, tile_width - maxx]
                )

                if min_distance < padding_size:
                    filter_mask[i] = False

            tile_results = list(np.array(tile_results)[filter_mask])
            # tile_results += filter(partial(tile_touch_filter, image_tile_poly=image_tile_poly), tile_results)

            new_masks = np.zeros((len(tile_results), *orig_image.shape[:2]), dtype=bool)

            # expand masks to full image
            for i, det in enumerate(tile_results):
                new_mask = new_masks[i]

                y_offset = 0
                y_endset = 0
                if iY == 0:
                    y_offset = pd
                if y_end > orig_image.shape[0] + pd:
                    y_endset = y_end - (orig_image.shape[0] + pd)
                x_offset = 0
                x_endset = 0
                if iX == 0:
                    x_offset = pd
                if x_end > orig_image.shape[1] + pd:
                    x_endset = x_end - (orig_image.shape[1] + pd)

                mask_height = (y_end - y) - y_endset
                mask_width = (x_end - x) - x_endset

                new_mask[
                    max(0, y - pd) : y - pd + mask_height,
                    max(0, x - pd) : x - pd + mask_width,
                ] = det["mask"][
                    y_offset:mask_height, x_offset:mask_width
                ]  # [:y_end - y,:x_end - x]
                det["mask"] = new_mask
                det["bbox"] += np.array([x, y, x, y]) - pd

            all_detections += tile_results

    return all_detections


def non_max_supression(all_detections: list[Polygon], iou=0.3):
    """
    Performing something like non-maximum supression on a list of detections

    TODO: make sure that this corresponds with some paper for nms

    all_detections: all detections found in an image
    iou: intersection over union: if a poly intersects more than that with another poly and it's score is lower it gets discarded.

    returns the filtered list of detections
    """
    # descending sort
    all_detections = sorted(all_detections, key=lambda det: det["score"])
    polygons = []
    for det in all_detections:
        contour = det["contours"][0]
        xs = contour["x"]
        ys = contour["y"]

        poly = Polygon(zip(xs, ys, strict=False))

        if not poly.is_valid:
            logging.warning("Invalid polygon!")

        polygons.append(poly)

    idx = rtree.index.Index()
    for pos, poly in enumerate(polygons):
        idx.insert(pos, poly.bounds)

    set_remove_indices = set()

    # Loop through each Shapely polygon
    for i, poly in enumerate(polygons):
        score = all_detections[i]["score"]
        area = poly.area

        # Merge cells that have overlapping bounding boxes
        for pos in idx.intersection(poly.bounds):
            if pos == i:
                continue

            poly_other = polygons[pos]
            score_other = all_detections[pos]["score"]

            # distance = poly.distance(poly_other)
            # intersect = poly.intersects(poly_other)

            # print(poly.is_valid)
            # print(poly_other.is_valid)

            poly_other = poly_other.buffer(0)
            # print(poly_other.is_valid)

            # compute intersection
            intersect_poly = poly.intersection(poly_other)
            intersect_area = intersect_poly.area

            if score < score_other and intersect_area / area > iou:
                # do not take poly
                set_remove_indices.add(i)
                break

    return list(
        map(
            lambda idet: idet[1],
            filter(
                lambda idet: idet[0] not in set_remove_indices,
                enumerate(all_detections),
            ),
        )
    )


def np_vec_no_jit_iou(boxes1, boxes2):
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou


def torch_vec_no_jit_iou(boxes1, boxes2):
    x11, y11, x12, y12 = torch.chunk(boxes1, 4, dim=1)
    x21, y21, x22, y22 = torch.chunk(boxes2, 4, dim=1)

    xA = torch.maximum(x11, torch.transpose(x21, 0, 1))
    yA = torch.maximum(y11, torch.transpose(y21, 0, 1))
    xB = torch.minimum(x12, torch.transpose(x22, 0, 1))
    yB = torch.minimum(y12, torch.transpose(y22, 0, 1))

    interArea = torch.maximum(
        (xB - xA + 1), torch.tensor(0, device=boxes1.device)
    ) * torch.maximum((yB - yA + 1), torch.tensor(0, device=boxes1.device))
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + torch.transpose(boxBArea, 0, 1) - interArea)

    return iou


def torch_mask_nms(
    masks,
    bboxes,
    scores,
    bbox_iou_threshold=0.1,
    mask_iou_threshold=0.4,
    score_threshold=0.1,
):
    """
    iou: if intersection between two cells is larger, only take the better scored one
    """

    device = "cuda:0"

    masks = torch.tensor(masks, device=device)
    bboxes = torch.tensor(bboxes, device=device)

    # areas = torch.sum(torch.tensor(masks), axis=(1,2)).numpy()

    bbox_iou = torch_vec_no_jit_iou(bboxes, bboxes)

    print(masks.shape)

    # print(masks.nbytes)

    filter_mask = scores >= score_threshold

    scores = torch.tensor(scores, device=device)

    drops = []  # torch.zeros_like(scores, dtype=torch.bool)

    # intersection = masks[None] & np.r
    for i, (mask, score) in enumerate(zip(masks, scores, strict=False)):
        if not filter_mask[i]:
            continue

        candidate_mask = bbox_iou[i, :] > bbox_iou_threshold

        # area = areas[i]#np.sum(mask)
        intersection = mask[None, :] & masks[candidate_mask]
        joint = mask[None, :] | masks[candidate_mask]

        intersection_areas = torch.sum(intersection, dim=(1, 2))
        joint_areas = torch.sum(joint, dim=(1, 2))

        relative_intersections = intersection_areas / joint_areas

        over_threshold = torch.where(relative_intersections > mask_iou_threshold)

        higher_scored = scores[candidate_mask][over_threshold] > score

        drops.append(~(torch.sum(higher_scored) >= 1).cpu())

        # if drop:
        #    filter_mask[i] = False

        # print(relative_intersections)

    filter_mask = np.array(drops, dtype=bool)

    return np.arange(len(masks))[filter_mask]


def mask_nms(
    masks,
    bboxes,
    scores,
    bbox_iou_threshold=0.1,
    mask_iou_threshold=0.4,
    score_threshold=0.1,
):
    """
    iou: if intersection between two cells is larger, only take the better scored one
    """
    # masks = torch.tensor(masks)
    # bboxes = torch.tensor(bboxes)

    # areas = torch.sum(torch.tensor(masks), axis=(1, 2)).numpy()

    bbox_iou = torch_vec_no_jit_iou(torch.tensor(bboxes), torch.tensor(bboxes)).numpy()

    print(masks.shape)

    print(masks.nbytes)

    filter_mask = scores >= score_threshold
    # intersection = masks[None] & np.r
    for i, (mask, score) in enumerate(zip(masks, scores, strict=False)):
        if not filter_mask[i]:
            continue

        candidate_mask = bbox_iou[i, :] > bbox_iou_threshold

        # area = areas[i]#np.sum(mask)
        intersection = mask[None, :] & masks[candidate_mask]
        joint = mask[None, :] | masks[candidate_mask]

        intersection_areas = np.sum(intersection, axis=(1, 2))
        joint_areas = np.sum(joint, axis=(1, 2))

        relative_intersections = intersection_areas / joint_areas

        over_threshold = np.where(relative_intersections > mask_iou_threshold)

        higher_scored = scores[candidate_mask][over_threshold] > score

        drop = np.sum(higher_scored) >= 1

        if drop:
            filter_mask[i] = False

        # print(relative_intersections)

    return np.arange(len(masks))[filter_mask]


def prediction(image, model, min_score=0.0, tiling=None):
    # apply tiled inference
    if tiling:
        all_detections = tiled_inference(image, model, **tiling)
        # filter by score
        all_detections = list(
            filter(lambda det: det["score"] > min_score, all_detections)
        )
        # perform non-max supressions (due to tiling this is needed)
        if len(all_detections) > 0:
            filter_mask = torch_mask_nms(
                np.stack([det["mask"] for det in all_detections]),
                np.stack([det["bbox"] for det in all_detections]),
                np.stack([det["score"] for det in all_detections]),
                score_threshold=min_score,
                mask_iou_threshold=0.6,
            )

            all_detections = list(np.array(all_detections)[filter_mask])
    else:
        all_detections = inference(image, model)
        all_detections = list(
            filter(lambda det: det["score"] > min_score, all_detections)
        )

    return all_detections
