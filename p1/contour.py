import numpy as np
from matplotlib import pyplot as plt
import cv2

im = cv2.imread('im.png',1)
plt.subplot(222),plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.title('org'), plt.xticks([]), plt.yticks([])

imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
plt.subplot(221),plt.imshow(thresh)
plt.title('Threshold'), plt.xticks([]), plt.yticks([])

image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
plt.subplot(223),plt.imshow(image)
plt.title('Image'), plt.xticks([]), plt.yticks([])

print("Found %d objects." % len(contours))
for (i, c) in enumerate(contours):
    print("\tSize of contour %d: %d" % (i, len(c)))
img = cv2.drawContours(im, contours, -1, (255,0,0), 10)
cv2.imshow('final',img)
plt.subplot(224),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Final'), plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()