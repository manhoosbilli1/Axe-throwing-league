import matplotlib.pylab as plt
import cv2 as cv 
import numpy as np 

circle_org = cv.imread('Photos/target.jpg')
circle = circle_org.copy()
circle = cv.cvtColor(circle, cv.COLOR_BGR2GRAY)
circle = cv.GaussianBlur(circle, (7, 7), 1)
canny = cv.Canny(circle, 175, 250)
contours, hieracrhy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

#Detecting the position of board and drawing a box around the circle  to get our ROI
max_cont = -1
max_idx = 0
for i in range(len(contours)):
    length = cv.arcLength(contours[i], True)
    if(length > max_cont):
        max_idx = i
        max_cont = length
x, y, w, h = cv.boundingRect(contours[max_idx])
img = cv.rectangle(canny, (x, y), (x+w, y+h), (255, 255, 255), 1)
#Get region of interest 
if(img.ndim == 2):
    IM_ROI = img[y:y+h,x:x+w]
else:
    IM_ROI = img[y:y+h,x:x+w,:]
#houghcircle 
# IM_ROI = cv.cvtColor(IM_ROI,cv.COLOR_BGR2GRAY)
print(IM_ROI.shape)
IM_ROI = cv.Canny(IM_ROI,175,250)
circles = cv.HoughCircles(IM_ROI, cv.HOUGH_GRADIENT, 1.2,100)

bg = np.ones((1000,1000,3), dtype=np.uint8)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    for (x,y,r) in circles:
        cv.circle(IM_ROI, (x,y), r, (0,0,255), 4)
        cv.rectangle(IM_ROI, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)


cv.imshow('org',circle_org)
cv.imshow('result', IM_ROI)
cv.waitKey(0)