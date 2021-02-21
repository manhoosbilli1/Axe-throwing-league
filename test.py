import matplotlib.pylab as plt
import cv2 as cv 
import numpy as np 


image = cv.imread('Photos/road.jpg')
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 75,150)


print(image.shape)
cv.imshow('img', edges)
cv.waitKey(0)
cv.destroyAllWindows()