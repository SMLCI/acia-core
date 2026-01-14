#!/usr/bin/env python
"""Verification script for Canny segmentation processor."""

import numpy as np
from acia.segm.processor.canny import CannySegmentationProcessor
from acia.segm.local import THWCSequenceSource

# Create test image with a white square
img = np.zeros((512, 512, 1), dtype=np.uint8)
img[100:200, 100:200] = 255

# Stack into 4D array (T, H, W, C)
imgs = np.stack([img], axis=0)

# Create sequence source
seq = THWCSequenceSource(imgs)

# Create processor with area filtering
proc = CannySegmentationProcessor(min_area=50, max_area=20000)

# Process
ov = proc(seq)

# Report
print(f"OK: filtered to {len(ov)} contours")
