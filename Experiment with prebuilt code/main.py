import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.nanfunctions import _divide_by_count

def detectDartboard(IM):
    #DETECTING THE DART BOARD/ROI
    IM_blur = cv.blur(IM,(5,5))
    #convert to hsv to find regions of color
    base_frame_hsv = cv.cvtColor(IM_blur, cv.COLOR_BGR2HSV)
    #extract green
    green_thres_low = int(90/255. * 180)
    green_thres_high = int(95/255. * 180)
    green_min = np.array([green_thres_low,100,100],np.uint8)
    green_max = np.array([green_thres_high,255,255],np.uint8)
    frame_threshed_green = cv.inRange(base_frame_hsv,green_min,green_max)
    #Extract red
    red_thres_low = int(0 /255. * 180)   #why tho?
    red_thres_high = int(20 /255. * 180)
    red_min = np.array([red_thres_low, 100, 100],np.uint8)
    red_max = np.array([red_thres_high, 255, 255],np.uint8)
    frame_threshed_red = cv.inRange(base_frame_hsv, red_min, red_max)
    #Combine both results into original image
    combined = frame_threshed_red + frame_threshed_green
    #close to get rid of annoying dots inside object
    kernel = np.ones((100,100),np.uint8)   #why 100X100?
    closing = cv.morphologyEx(combined, cv.MORPH_CLOSE, kernel)
    #find contours
    ret, thresh = cv.threshold(combined, 127,255,0)
    contours,hierarchy = cv.findContours(closing.copy(),cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    max_cont = -1
    max_idx = 0
    for i in range(len(contours)):
        length = cv.arcLength(contours[i], True)
        if length > max_cont:
            max_idx = 1
            max_cont = length

    x,y,w,h = cv.boundingRect(contours[max_idx])
    x = x-200
    x = y-200
    w = w+int(2*200)
    h = h+int(2*200)
    return x,y,w,h,closing,frame_threshed_green,frame_threshed_red

def debugRectangle(IM,x,y,w,h):
    IM_copy = IM.copy()
    cv.rectangle(IM_copy,(x,y),(x+w,y+h),(255,255,255),1)
    return IM_copy

def scaleROI(IM):
    if(IM.ndim == 3):
        IM_normal = np.zeros((1000,1000,IM.shape[2]),"uint8")
    else:
        IM_normal = np.zeros((1000,1000),"uint8")
    scale = 1
    if IM.shape[0] > IM.shape[1]:
        #higher than width
        scale = IM_normal.shape[0] / IM.shape[0]
    else:
        #widther than high
        scale = IM_normal.shape[1] / IM.shape[1]
    new_y =  int(IM.shape[0] * scale)
    new_x =  int(IM.shape[1] * scale)
    offset_y = int((IM_normal.shape[0] - new_y)/2) 
    offset_x = int((IM_normal.shape[1] - new_x)/2)
    IM_resized = cv.resize(IM, (new_x,new_y),cv.INTER_AREA)
    if(IM.ndim == 3):
        IM_normal[offset_y:offset_y+new_y,offset_x:offset_x+new_x,:] = IM_resized
    else:
        IM_normal[offset_y:offset_y+new_y,offset_x:offset_x+new_x] = IM_resized
    return IM_normal


def getROI(IM,x,y,w,h):
    if(IM.ndim == 2):
        IM_ROI = IM[y:y+h,x:x+w]
    else:
        IM_ROI = IM[y:y+h,x:x+w,:]
    return IM_ROI

def imShow(frame):
    if(frame.ndim == 3):
        plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    else:
        plt.imshow(frame,cmap='Greys_r')
    plt.show()

class GUI:

    ENV = {
        'GUI_RESOLUTION_SCALE' : 0.5,
        'SHOW_GUI' : ["FEED","DIFFERENCE", "ARROW","DARTBOARD", "APEX", "ORIENTATION"]#['ORIENTATION','FEED',"ARROW1","ARROW2" ,"DIFFERENCE", "ROTATED", "DARTBOARD"]
    }

def show(frame, window='feed'):
    if window in GUI.ENV['SHOW_GUI']:
        cv.imshow(window, cv.resize(frame, (int(frame.shape[1] * GUI.ENV['GUI_RESOLUTION_SCALE']), int(frame.shape[0] * GUI.ENV['GUI_RESOLUTION_SCALE']))))


def readImage(path, dimension=None):
    IM = cv.imread(path)
    if dimension is not None:
        IM = cv.resize(IM, dimension)
    if IM.ndim == 3: 
        base_frame_gray = cv.cvtColor(IM, cv.COLOR_BGR2GRAY)
    #print(IM.shape)
    #print(IM.dtype)
    #show(IM)
    return IM, base_frame_gray    


#INITIAL SETUP
IM, BASE_FRAME_GRAY = readImage("dart.jpg", (1080,1920))
x,y,w,h,BOARD,GREEN,RED = detectDartboard(IM)
plt.imshow(debugRectangle(IM,x,y,w,h))
IM_ROI = scaleROI(getROI(IM,x,y,w,h))
IM_ROI_grey = scaleROI(getROI(BASE_FRAME_GRAY,x,y,w,h)) 
IM_ROI_green = scaleROI(getROI(GREEN,x,y,w,h))  
IM_ROI_red = scaleROI(getROI(RED,x,y,w,h))  
IM_ROI_board = scaleROI(getROI(BOARD,x,y,w,h))  
imShow(IM_ROI)


#roi is not correct. need to correct it. 