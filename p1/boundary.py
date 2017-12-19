import numpy as np
from matplotlib import pyplot as plt
import cv2

org = cv2.imread('images.jpg',1)
plt.subplot(231),plt.imshow(cv2.cvtColor(org, cv2.COLOR_BGR2RGB))
plt.title('org'), plt.xticks([]), plt.yticks([])

imgray = cv2.cvtColor(org,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
plt.subplot(232),plt.imshow(thresh)
plt.title('Threshold'), plt.xticks([]), plt.yticks([])

image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print("Found %d objects." % len(contours))
for (i, c) in enumerate(contours):
    print("\tSize of contour %d: %d" % (i, len(c)))
final = cv2.drawContours(org, contours, -1, (255,0,0), 10)
plt.subplot(233),plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
plt.title('Final'), plt.xticks([]), plt.yticks([])

cnt=contours[1]
x,y,w,h = cv2.boundingRect(cnt)
boundR = cv2.rectangle(org,(x,y),(x+w,y+h),(0,255,0),2)
plt.subplot(234),plt.imshow(cv2.cvtColor(boundR, cv2.COLOR_BGR2RGB))
plt.title('BoundR'), plt.xticks([]), plt.yticks([])

cnt=contours[2]
(x1,y1),radius = cv2.minEnclosingCircle(cnt)
center = (int(x1),int(y1))
radius = int(radius)
boundC = cv2.circle(org,center,radius,(0,255,0),2)
plt.subplot(235),plt.imshow(cv2.cvtColor(boundC, cv2.COLOR_BGR2RGB))
plt.title('BoundC'), plt.xticks([]), plt.yticks([])

cnt=contours[3]
ellipse = cv2.fitEllipse(cnt)
boundE = cv2.ellipse(org,ellipse,(0,255,0),2)
plt.subplot(236),plt.imshow(cv2.cvtColor(boundE, cv2.COLOR_BGR2RGB))
plt.title('BoundE'), plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()