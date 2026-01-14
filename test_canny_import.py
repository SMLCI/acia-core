import numpy as np
from acia.segm.processor.canny import CannySegmentationProcessor
from acia.segm.local import THWCSequenceSource

img = np.random.randint(0, 256, (512, 512, 1), dtype=np.uint8)
seq = THWCSequenceSource([img])
proc = CannySegmentationProcessor()
ov = proc(seq)
print(f'OK: {len(ov)} contours')
