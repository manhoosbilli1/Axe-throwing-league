import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt

img = cv.imread('Photos\clean target.png')
#save a copy of original image
org= img
#Resizing
print('original dimensions: ' , img.shape)
#cropping the image by providing start and end pixel coordinates
img = img[150:500, 200:500]
print('Resized dimensions: ', img.shape)

#Prepping the image for further processing. (smoothing de-noising?)
img = cv.bilateralFilter(img, 20, 75,75)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(img, (9,9),0)
#maybe should use adaptive thresholding with otsu for better results. 
_,result = cv.threshold(img, 120,255,cv.THRESH_BINARY_INV)
# result = cv.Canny(img, 220,250)
kernel = np.ones((8,8),np.uint8)

# window = np.zeros([])
contours, _ = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) 
cv.drawContours(img,contours,-1,(0,255,0),3)

#detect circles


#detect position of dart/axe



#calculate position 


#assign score


#display score
result[100:200,100:200] = 0
cv.imshow("resized" , img)
cv.imshow('result', result)
cv.waitKey(0)