import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt


#Making a simple circle with numpy. 

# circle = np.zeros((800, 800, 3), np.uint8)
# cv.circle(circle, (300, 300), 90, (255, 0, 0), -1)

circle_org = cv.imread('Photos/target2.jpg')
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



#Prepare second image in order to be compared



cv.imshow('contours', img)
cv.imshow('Original', circle)
cv.imshow('ROI', IM_ROI)
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