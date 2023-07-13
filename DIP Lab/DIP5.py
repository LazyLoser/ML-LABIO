import cv2
import numpy as np
img = cv2.imread(wallpaper.jpg)

enhanced = cv2.equalizeHist(img)
ret,threshold = cv2.threshold(enhanced,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("original",img)
cv2.imshow("enhanced", enhanced)
cv2.imshow("segmented", threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()
