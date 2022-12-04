import numpy as np
import cv2

moving = np.load('moving.npy')
mov_mask = np.load('moving_mask.npy')

img = moving[101]
img = img.astype(np.float32)
mask = mov_mask[101]
mask = mask.astype(np.uint8)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

output = cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
cv2.imshow('img', output)
cv2.waitKey(10000)
cv2.destroyAllWindows()