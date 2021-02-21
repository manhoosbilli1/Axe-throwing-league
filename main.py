import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt


#Making a simple circle with numpy. 
circle = np.zeros((400, 400, 3), np.uint8)
img = circle.copy()
cv.circle(circle, (200, 200), 75, (255, 0, 0), -1)

circle = cv.GaussianBlur(circle, (5, 5), 1)
canny = cv.Canny(circle, 175, 250)
contours, hieracrhy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

#draw a bounding box around the circle  
max_cont = -1
max_idx = 0

for i in range(len(contours)):
    length = cv.arcLength(contours[i], True)
    if(length > max_cont):
        max_idx = i
        max_cont = length
x, y, w, h = cv.boundingRect(contours[max_idx])
img_copy = canny.copy()
img = cv.rectangle(img_copy, (x, y), (x+w, y+h), (255, 255, 255), 1)

cv.drawContours(img, contours, -1, (255, 25, 255), 1)
cv.imshow('contours', img)
cv.imshow('Original', circle)
cv.waitKey(0)


#finding region of interest
#since camera will be stationary and target is standard ROI will be constant

#detecting the board 



#detecting circles 



#detect axe 


#compute difference


#assign score 





# # #display score
# cv.imshow("result" , gray)
# cv.imshow("img" , img)




#As explained in a paper i read. 
#Simple canny with gaussian blur works best for good results. what about binarization and denoising? 
#use harris corner detector with hough transform to detect corner of the board