import cv2
import numpy as np

img = cv2.imread('im.png')


px = img[100,100]
print px

# accessing only blue pixel
blue = img[100,100,0]
print blue

red = img[100,100,1]
print red

green = img[100,100,2]
print green
