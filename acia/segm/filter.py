"""Filters for segmentating overlay objects"""

from functools import partial

import cv2
import numpy as np
import shapely
import shapely.affinity
import tqdm.auto as tqdm
from rtree import index
from shapely.geometry import Polygon
from shapely.validation import make_valid

from acia.base import Overlay


def bbox_to_rectangle(bbox: tuple[float, float, float, float]):
    minx, miny, maxx, maxy = bbox
    return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])


class NMSFilter:
    """Non-maximum supression filter based on contours"""

    @staticmethod
    def filter(overlay: Overlay, iou_thr=0.1, mode="iou") -> Overlay:
        prefiltered_contours = [
            cont for cont in overlay.contours if len(cont.coordinates) >= 3
        ]

        # sort contours by their score (lowest first)
        sorted_contours = sorted(
            prefiltered_contours, key=lambda c: c.score if c.score is not None else 0.0
        )
        # make (valid) shapely polygons
        polygons = [
            make_valid(shapely.geometry.polygon.Polygon(contour.coordinates))
            for contour in sorted_contours
        ]

        keep_list = []

        # build an rtree with bounding boxes
        idx = index.Index()
        for i, p_i in enumerate(polygons):
            minx, miny, maxx, maxy = p_i.bounds
            left = minx
            right = maxx
            top = maxy
            bottom = miny
            idx.insert(i, (left, bottom, right, top))

        for i, p_i in tqdm.tqdm(enumerate(polygons), total=len(polygons)):
            keep = True

            # zero area stuff is not considered
            if p_i.area <= 0:
                keep = False
                keep_list.append(keep)
                continue

            # get the intersection candidates by querying the rtree (overlapping bboxes)
            minx, miny, maxx, maxy = p_i.bounds
            left = minx
            right = maxx
            top = maxy
            bottom = miny
            candidate_idx_iter = idx.intersection((left, bottom, right, top))

            candidate_idx_list = list(
                filter(
                    partial(
                        lambda index, loop_index, sorted_contours: index > loop_index
                        and sorted_contours[loop_index].frame
                        == sorted_contours[index].frame,
                        loop_index=i,
                        sorted_contours=sorted_contours,
                    ),
                    candidate_idx_iter,
                )
            )

            # for those candidates we will compute the intersections in details
            for j in candidate_idx_list:
                p_j = polygons[j]

                # compute iou
                if mode == "i":
                    iou = p_i.intersection(p_j).area / p_i.area  # (p_i.union(p_j).area)
                elif mode == "iou":
                    iou = p_i.intersection(p_j).area / (p_i.union(p_j).area)
                # compare to threshold
                if iou >= iou_thr:
                    # if exceeding iou drop this cell detection
                    # print("iou: %.2f" % iou)
                    keep = False
                    break

            keep_list.append(keep)

        overlay = Overlay(
            [cont for i, cont in enumerate(sorted_contours) if keep_list[i]]
        )

        return overlay


class SizeFilter:
    """Filter by contour area"""

    @staticmethod
    def filter(overlay: Overlay, min_area, max_area) -> Overlay:
        """Filter an overlay based on contour sizes

        Args:
            overlay (Overlay): the overlay to filter
            min_area ([type]): minimum area of a contour
            max_area ([type]): maximum area of a contour

        Returns:
            Overlay: the filtered overlay
        """
        contour_shapes = [Polygon(cont.coordinates) for cont in overlay.contours]
        result_overlay = Overlay([])
        for cont, shape in zip(overlay.contours, contour_shapes, strict=False):
            area = shape.area

            if min_area < area < max_area:
                result_overlay.add_contour(cont)

        return result_overlay


class EllipsoidFilter:
    """Contour filter for ellipsoid shape"""

    @staticmethod
    def filter(
        overlay: Overlay, min_width_height_ratio, max_width_height_ratio
    ) -> Overlay:
        result_overlay = Overlay([])
        for cont in overlay.contours:
            if len(cont.coordinates) < 5:
                # need 5 points for ellipsoid fit
                continue
            ellipse = cv2.fitEllipse(cont.coordinates)

            center, (width, height), rotation = ellipse

            width_height_ratio = width / height

            # from shapely.figures import SIZE, GREEN, GRAY, set_limits

            # Let create a circle of radius 1 around center point:
            circ = shapely.geometry.Point(center).buffer(1)

            # Let create the ellipse along x and y:
            ell = shapely.affinity.scale(circ, width / 2, height / 2)

            # Let rotate the ellipse (clockwise, x axis pointing right):
            ellr = shapely.affinity.rotate(ell, rotation)

            # create shapely shape from contour coordinates
            shape = Polygon(cont.coordinates)
            # create minimal rotated rectangle
            min_rect = shape.minimum_rotated_rectangle

            rect_area_error = np.abs(shape.area - min_rect.area) / shape.area
            ellipse_area_error = np.abs(ellr.area - shape.area) / shape.area

            if (
                min_width_height_ratio <= width_height_ratio <= max_width_height_ratio
                and ellipse_area_error < rect_area_error
            ):
                # if an ellipse can better explain the cell detection than a rectangle
                result_overlay.add_contour(cont)

        return result_overlay
