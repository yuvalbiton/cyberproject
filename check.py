import cv2
import numpy as np

img = np.zeros((431,450), np.uint8)
center = (269,161)
axes = (123,85)
img = cv2.ellipse(img, center, axes,45,0,360,(255,255,255),1)
cv2.imshow('test', img)
cv2.waitKey(0)
