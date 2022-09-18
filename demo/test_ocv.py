import time
import cv2
import numpy as np

lastt = time.time()

for i in range(120):
    # Capture frame-by-frame
    frame = np.zeros((224,224,3)).astype(np.uint8)

    while lastt + 1. / 6 > time.time():
        time.sleep(0.005)
    lastt = time.time()

    # Display the resulting frame
    cv2.imshow('Frame', (np.ones((224,224,3))*224).astype(np.uint8))
    cv2.waitKey()