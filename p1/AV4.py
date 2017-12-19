import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('logo.png',1)
img_og = img
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_3_channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

numpy_horizontal1 = np.hstack((img, gray_3_channel))
numpy_horizontal_concat0 = np.concatenate((img, gray_3_channel), axis=0)

ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
thresh_3_channel = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
img_og = cv2.drawContours(img_og, contours, -1, (255,0,0), 10)

numpy_horizontal2 = np.hstack((thresh_3_channel, img_og))

numpy_vertical = np.vstack((numpy_horizontal1,numpy_horizontal2))


numpy_horizontal_concat1 = np.concatenate((thresh_3_channel, img_og), axis=0)
numpy_vertical_concat = np.concatenate((numpy_horizontal_concat0, numpy_horizontal_concat1), axis=1)

cv2.imshow('Numpy Vertical Concat', numpy_vertical_concat)
cv2.waitKey()
