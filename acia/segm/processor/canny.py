"""Canny edge detection based segmentation processor"""

import cv2
import numpy as np
from tqdm.auto import tqdm

from acia.base import Contour, ImageSequenceSource, Overlay

from . import SegmentationProcessor


class CannySegmentationProcessor(SegmentationProcessor):
    """Segmentation processor using Canny edge detection.

    This processor uses OpenCV's Canny edge detection algorithm to identify
    cell boundaries in images, then extracts contours from the edge map.

    Args:
        canny_low (int): Lower threshold for Canny edge detection. Default: 50
        canny_high (int): Upper threshold for Canny edge detection. Default: 150
        min_area (float): Minimum contour area in pixels. Contours smaller than
            this are filtered out. Default: 50
        max_area (float): Maximum contour area in pixels. Contours larger than
            this are filtered out. Default: 100000
        blur_kernel (int): Size of Gaussian blur kernel (must be odd). Default: 5
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        min_area: float = 50,
        max_area: float = 100000,
        blur_kernel: int = 5,
    ):
        """Initialize Canny edge detection processor with configurable parameters."""
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area = min_area
        self.max_area = max_area
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1

    def __call__(self, images: ImageSequenceSource) -> Overlay:
        """Process image sequence and return segmentation overlay.

        Args:
            images (ImageSequenceSource): Source of images to segment

        Returns:
            Overlay: Overlay containing detected contours with correct frame indices
        """
        overlay = Overlay([])

        for frame_id, image in enumerate(
            tqdm(images, desc="Performing Canny edge detection segmentation...")
        ):
            # Extract raw image
            img = image.raw

            # Convert to grayscale if needed
            if len(img.shape) == 3:
                # Handle multi-channel image
                if img.shape[2] == 1:
                    # Single channel, just squeeze
                    gray = img[..., 0]
                else:
                    # Multiple channels, convert from BGR to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                # Already grayscale
                gray = img

            # Ensure uint8 format for Canny
            if gray.dtype != np.uint8:
                gray = cv2.convertScaleAbs(gray)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 1.5)

            # Apply Canny edge detection
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

            # Find contours from edge map
            contours, _ = cv2.findContours(
                edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            # Process and filter contours
            for contour in contours:
                # Squeeze contour array to remove singleton dimension
                contour = np.squeeze(contour)

                # Skip degenerate contours (need at least 3 points for a valid polygon)
                if contour.ndim < 2 or len(contour) < 3:
                    continue

                # Calculate contour area
                area = cv2.contourArea(contour)

                # Filter by area thresholds
                if area < self.min_area or area > self.max_area:
                    continue

                # Calculate perimeter for validation
                perimeter = cv2.arcLength(contour, True)
                if perimeter <= 0:
                    continue

                # Optional: Calculate solidity (area / convex hull area)
                # This filters for well-shaped contours
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = float(area) / hull_area
                    if solidity < 0.5:
                        # Skip poorly shaped contours
                        continue

                # Create contour object with area as the score
                # Ensure contour is float32 and in (x, y) format
                contour_float = contour.astype(np.float32)
                contour_obj = Contour(
                    coordinates=contour_float,
                    score=area,
                    frame=frame_id,
                    id=-1,
                    label=None,
                )
                overlay.add_contour(contour_obj)

        return overlay
