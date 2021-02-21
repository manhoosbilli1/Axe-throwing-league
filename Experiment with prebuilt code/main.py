import cv2
import matplotlib.pyplot as plt
import numpy as np


class Dartboard_Detector:

    ENV = {
        'DARTBOARD_SHAPE' : (1000,1000),

        'DETECTION_BLUR' : (5,5),
        'DETECTION_GREEN_LOW' : 90,
        'DETECTION_GREEN_HIGH' : 95,
        'DETECTION_RED_LOW' : 0,
        'DETECTION_RED_HIGH' : 20,
        'DETECTION_STRUCTURING_ELEMENT' : (100,100),
        'DETECTION_BINARY_THRESHOLD_MIN' : 127,
        'DETECTION_BINARY_THRESHOLD_MAX' : 255,
        'DETECTION_OFFSET' : 200,

        'ORIENTATION_BLUR' : (5,5),
        'ORIENTATION_COLOR_LOW' : 45,
        'ORIENTATION_COLOR_HIGH': 60,
        'ORIENTATION_KERNEL' : (100,100),
        'ORIENTATION_ELEMENT_SIZE_MIN' : 350,
        'ORIENTATION_ELEMENT_SIZE_MAX' : 600,

        'ORIENTATION_TEMPLATES' : ['shape_top.png','shape_bottom.png','shape_left.png','shape_right.png']

        }

    def scaleROI(self,IM):
        if(IM.ndim == 3):
            IM_normal = np.zeros((self.ENV['DARTBOARD_SHAPE'][0],self.ENV['DARTBOARD_SHAPE'][1],IM.shape[2]),"uint8")
        else:
            IM_normal = np.zeros((self.ENV['DARTBOARD_SHAPE'][0],self.ENV['DARTBOARD_SHAPE'][1]),"uint8")
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
        IM_resized = cv2.resize(IM, (new_x,new_y),cv2.INTER_AREA)
        if(IM.ndim == 3):
            IM_normal[offset_y:offset_y+new_y,offset_x:offset_x+new_x,:] = IM_resized
        else:
            IM_normal[offset_y:offset_y+new_y,offset_x:offset_x+new_x] = IM_resized
        return IM_normal

    def detectDartboard(self,IM):
        IM_blur = cv2.blur(IM,Dartboard_Detector.ENV['DETECTION_BLUR'])
        #convert to HSV
        base_frame_hsv = cv2.cvtColor(IM_blur, cv2.COLOR_BGR2HSV)
        # Extract Green
        green_thres_low = int(Dartboard_Detector.ENV['DETECTION_GREEN_LOW'] /255. * 180)
        green_thres_high = int(Dartboard_Detector.ENV['DETECTION_GREEN_HIGH'] /255. * 180)
        green_min = np.array([green_thres_low, 100, 100],np.uint8)
        green_max = np.array([green_thres_high, 255, 255],np.uint8)
        frame_threshed_green = cv2.inRange(base_frame_hsv, green_min, green_max)
        #Extract Red
        red_thres_low = int(Dartboard_Detector.ENV['DETECTION_RED_LOW'] /255. * 180)
        red_thres_high = int(Dartboard_Detector.ENV['DETECTION_RED_HIGH'] /255. * 180)
        red_min = np.array([red_thres_low, 100, 100],np.uint8)
        red_max = np.array([red_thres_high, 255, 255],np.uint8)
        frame_threshed_red = cv2.inRange(base_frame_hsv, red_min, red_max)
        #Combine
        combined = frame_threshed_red + frame_threshed_green
        #Close
        kernel = np.ones(Dartboard_Detector.ENV['DETECTION_STRUCTURING_ELEMENT'],np.uint8)
        closing = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        GUI.show(closing, "Dart_Detector")
        #find contours
        ret,thresh = cv2.threshold(combined,Dartboard_Detector.ENV['DETECTION_BINARY_THRESHOLD_MIN'],Dartboard_Detector.ENV['DETECTION_BINARY_THRESHOLD_MAX'],0)
        contours, hierarchy = cv2.findContours(closing.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        #drawing a bounding box around the board by using contours
        max_cont = -1   
        max_idx = 0
        for i in range(len(contours)):
            length = cv2.arcLength(contours[i], True)
            if  length > max_cont:
                max_idx = i
                max_cont = length
        x,y,w,h = cv2.boundingRect(contours[max_idx])
        x = x-Dartboard_Detector.ENV['DETECTION_OFFSET']
        y = y-Dartboard_Detector.ENV['DETECTION_OFFSET']
        w = w+int(2*Dartboard_Detector.ENV['DETECTION_OFFSET'])
        h = h+int(2*Dartboard_Detector.ENV['DETECTION_OFFSET'])
        return x,y,w,h,closing,frame_threshed_green,frame_threshed_red


    def getOrientation(self,IM_ROI,IM_ROI_board):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,Dartboard_Detector.ENV['ORIENTATION_KERNEL'])
        #Segment zones
        IM_ROI_blur = cv2.blur(IM_ROI,Dartboard_Detector.ENV['ORIENTATION_BLUR'])
        #convert to HSV
        IM_ROI_HSV = cv2.cvtColor(IM_ROI_blur, cv2.COLOR_BGR2HSV)
        purple_thres_low = int(Dartboard_Detector.ENV['ORIENTATION_COLOR_LOW'] /255. * 180)
        purple_thres_high = int(Dartboard_Detector.ENV['ORIENTATION_COLOR_HIGH'] /255. * 180)
        purple_min = np.array([purple_thres_low, 100, 100],np.uint8)
        purple_max = np.array([purple_thres_high, 255, 255],np.uint8)
        frame_thres_color = cv2.inRange(IM_ROI_HSV, purple_min, purple_max)
        #Mask
        frame_thres_color = cv2.subtract(frame_thres_color,IM_ROI_board)
        frame_thres_color_closed = cv2.morphologyEx(frame_thres_color, cv2.MORPH_CLOSE, kernel)
        
        #Compute contours
        im2, contours, hierarchy = cv2.findContours(frame_thres_color_closed.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contour_lengths = []
        contours_structure = []
        for i in range(len(contours)):
            length = cv2.arcLength(contours[i],True)
            contour_lengths.append(length)
            if length > Dartboard_Detector.ENV['ORIENTATION_ELEMENT_SIZE_MIN'] and length < Dartboard_Detector.ENV['ORIENTATION_ELEMENT_SIZE_MAX']:
                contours_structure.append(contours[i])
        #debug histogramm
        #print(len(point_contours))
        #plt.hist(contour_lengths, bins=20, range=(50,1000), normed=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, hold=None, data=None)
        #plt.show()
        return frame_thres_color,frame_thres_color_closed,contours_structure


    def getOrientationCorr(self,IM_ROI,base_dir):
        kernel_l = cv2.imread(base_dir + self.ENV['ORIENTATION_TEMPLATES'][2])
        kernel_r = cv2.imread(base_dir + self.ENV['ORIENTATION_TEMPLATES'][3])
        kernel_t = cv2.imread(base_dir + self.ENV['ORIENTATION_TEMPLATES'][0])
        kernel_b = cv2.imread(base_dir + self.ENV['ORIENTATION_TEMPLATES'][1])
        h = kernel_l.shape[0]
        w = kernel_l.shape[1]
       
        #right
        res = cv2.matchTemplate(IM_ROI,kernel_r,cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        right_top_left = max_loc
        right = (right_top_left[0] + w, right_top_left[1] + h//2)
        #GUI.imShow(kernel_r)
        

        #left
        res = cv2.matchTemplate(IM_ROI,kernel_l,cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        left_top_left = max_loc
        left = (left_top_left[0], left_top_left[1] + h//2)
        #GUI.imShow(kernel_l)
        
        h = kernel_t.shape[0]
        w = kernel_t.shape[1]
        #top
        res = cv2.matchTemplate(IM_ROI,kernel_t,cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_top_left = max_loc
        top = (top_top_left[0] + w//2, top_top_left[1])
        #GUI.imShow(kernel_t)
        #GUI.imShow(res)
        #print(max_loc)

        #bottom
        res = cv2.matchTemplate(IM_ROI,kernel_b,cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        bottom_top_left = max_loc
        bottom = (bottom_top_left[0] + w//2, bottom_top_left[1] + h)
        #GUI.imShow(kernel_b)
        
        return top_top_left,bottom_top_left,left_top_left,right_top_left,top,bottom,left,right

class Image_Tools:

    @staticmethod
    def getMaxContourIdx(contours):
        max_contour_length = 0
        max_contour = None
        for j in range(len(contours)):
            length = cv2.arcLength(contours[j],True)
            if length > max_contour_length:
                max_contour_length = length
                max_contour = j
        return max_contour

    @staticmethod
    def readFrame(time_in_millis, size):
        cap.set(cv2.CAP_PROP_POS_MSEC, time_in_millis)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, size) 
            return frame

    @staticmethod
    def readAndSafeFrames(path):
        cap = cv2.VideoCapture(path)
        frames = [] 
        frames.append(readFrame(100)) 
        frames.append(readFrame(2000)) 
        frames.append(readFrame(3000)) 
        frames.append(readFrame(5000)) 
        cap.release() 
        for i in range(len(frames)): 
            cv2.imwrite("Vid" + str(i) + ".png", frames[i])    
    

    @staticmethod
    def normalizeHist(arr):
        minval = arr[:,:].min()
        maxval = arr[:,:].max()
        print(minval)
        print(maxval)
        if minval != maxval:
            arr -= minval
            arr *= int((255.0 / (maxval - minval)))
        return arr

    @staticmethod
    def rotateImage(image, angle):
        image_center = tuple(np.array(image.shape)/2)
        rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
        return result

    @staticmethod
    def sift(GrayIM):
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(GrayIM,None)
        out = np.array(GrayIM.shape)
        out = cv2.drawKeypoints(GrayIM,kp,out)
        return out, kp

    @staticmethod
    def debugRectangle(IM,x,y,w,h):
        IM_copy = IM.copy()
        cv2.rectangle(IM_copy,(x,y),(x+w,y+h),(255,255,255),1)
        return IM_copy

    @staticmethod
    def debugContours(IM,contours):
        SAMPLE = np.zeros(IM.shape,"uint8")
        cv2.drawContours(SAMPLE, contours, -1, (255,255,255), 10)
        return SAMPLE

    @staticmethod
    def getROI(IM,x,y,w,h):
        if(IM.ndim == 2):
            IM_ROI = IM[y:y+h,x:x+w]
        else:
            IM_ROI = IM[y:y+h,x:x+w,:]
        return IM_ROI


    @staticmethod
    def readImage(path, dimension=None):
        IM = cv2.imread(path)
        if dimension is not None:
            IM = cv2.resize(IM, dimension)
        if IM.ndim == 3: 
            base_frame_gray = cv2.cvtColor(IM, cv2.COLOR_BGR2GRAY)
        #print(IM.shape)
        #print(IM.dtype)
        #show(IM)
        return IM, base_frame_gray

    @staticmethod
    def prepareImage(IM, dimension):
        IM = cv2.resize(IM, dimension) 
        base_frame_gray = cv2.cvtColor(IM, cv2.COLOR_BGR2GRAY)
        #print(IM.shape)
        #print(IM.dtype)
        #show(IM)
        return IM, base_frame_gray

    @staticmethod
    def getIntersection(src_points):
        if src_points.shape != (4,2):
            return None, None
        #interesect lines
        x1 = src_points[0,0] 
        y1 = src_points[0,1]
        x2 = src_points[3,0]
        y2 = src_points[3,1]
        x3 = src_points[1,0] 
        y3 = src_points[1,1]
        x4 = src_points[2,0]
        y4 = src_points[2,1]
        py = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        px =  ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        #show(IM)
        #swap output
        return int(px),int(py)

    @staticmethod    
    def debugIntersection(IM,src_points):
        IM = IM.copy()
        x1 = src_points[0,0] 
        y1 = src_points[0,1]
        x2 = src_points[3,0]
        y2 = src_points[3,1]
        x3 = src_points[1,0] 
        y3 = src_points[1,1]
        x4 = src_points[2,0]
        y4 = src_points[2,1]
        cv2.line(IM,(x1,y1),(x2,y2),255,5)
        cv2.line(IM,(x3,y3),(x4,y4),255,5)
        return IM

class GUI:

    ENV = {
        'GUI_RESOLUTION_SCALE' : 0.5,
        'SHOW_GUI' : ["FEED","DIFFERENCE", "ARROW","DARTBOARD", "APEX", "ORIENTATION"]#['ORIENTATION','FEED',"ARROW1","ARROW2" ,"DIFFERENCE", "ROTATED", "DARTBOARD"]
    }

    @staticmethod
    def show(frame, window='feed'):
        if window in GUI.ENV['SHOW_GUI']:
            cv2.imshow(window, cv2.resize(frame, (int(frame.shape[1] * GUI.ENV['GUI_RESOLUTION_SCALE']), int(frame.shape[0] * GUI.ENV['GUI_RESOLUTION_SCALE']))))

    @staticmethod
    def imShow(frame):
        if(frame.ndim == 3):
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(frame,cmap='Greys_r')
        plt.show()

class Difference_Detector:

    ENV = {
        'BLUR' : (5,5),
        'BINARY_THRESHOLD_MIN' : 75,
        'BINARY_THRESHOLD_MAX' : 255,
        'CLAHE_CLIP_LIMIT' : 5,
        'CLAHE_TILE_SIZE' : (10,10),

        'ARROW_BLUR' : (5,5),
        'ARROW_BINARY_THRESHOLD_MIN' : 50,
        'ARROW_BINARY_THRESHOLD_MAX' : 255,
        'ARROW_CLAHE_CLIP_LIMIT' : 20,
        'ARROW_CLAHE_TILE_SIZE' : (10,10)
    }
    #def __init__(self):

    def computeDifference(self,grey1,grey2):
        # blur
        blur = Difference_Detector.ENV['BLUR']
        grey2 = cv2.blur(grey2,blur)
        grey1 = cv2.blur(grey1,blur)
        #normalize
        grey1 = cv2.equalizeHist(grey1)
        grey2 = cv2.equalizeHist(grey2)
        clahe = cv2.createCLAHE(Difference_Detector.ENV['CLAHE_CLIP_LIMIT'], Difference_Detector.ENV['CLAHE_TILE_SIZE'])
        #clahe
        grey1 = clahe.apply(grey1)
        grey2 = clahe.apply(grey2)
        #diff
        diff = cv2.subtract(grey2,grey1) + cv2.subtract(grey1,grey2)
        ret2,dif_thred = cv2.threshold(diff,Difference_Detector.ENV['BINARY_THRESHOLD_MIN'],Difference_Detector.ENV['BINARY_THRESHOLD_MAX'],cv2.THRESH_BINARY)
        return dif_thred,grey1,grey2,diff


    def computeDifferenceHighRes(self,grey1,grey2):
        # blur
        blur = Difference_Detector.ENV['BLUR']
        grey2 = cv2.blur(grey2,blur)
        grey1 = cv2.blur(grey1,blur)
        #normalize
        grey1 = cv2.equalizeHist(grey1)
        grey2 = cv2.equalizeHist(grey2)
        clahe = cv2.createCLAHE(Difference_Detector.ENV['ARROW_CLAHE_CLIP_LIMIT'], Difference_Detector.ENV['ARROW_CLAHE_TILE_SIZE'])
        #clahe
        grey1 = clahe.apply(grey1)
        grey2 = clahe.apply(grey2)
        #diff
        diff = cv2.subtract(grey2,grey1) + cv2.subtract(grey1,grey2)
        ret2,dif_thred = cv2.threshold(diff,Difference_Detector.ENV['ARROW_BINARY_THRESHOLD_MIN'],Difference_Detector.ENV['ARROW_BINARY_THRESHOLD_MAX'],cv2.THRESH_BINARY)
        return dif_thred

class Dartboard:
    ENV = {
        'BULL_CENTER_DETECTION_SIZE' : 50,

        'DART_BOARD_TARGET_DIMENSION' : (1000,1000),
        'DART_BOARD_TARGET_ROTATION' : 45, 
        'DART_BOARD_TARGET_OFFSET' : 500, #in mm-1 500=5cm

        'TEXT_SIZE' : 3,
        'TEXT_THICKNESS' : 10
        }

    
    def _correctCenterOfBull(self,IM_ROI_grey,IM_ROI_red,px,py,src_points):
        src_points = src_points.copy()
        offset = Dartboard.ENV["BULL_CENTER_DETECTION_SIZE"]
        ROI_center = IM_ROI_grey[px-offset:px+offset,py-offset:py+offset] 
        IM_ROI_red_center = IM_ROI_red[px-offset:px+offset,py-offset:py+offset]
        
        ROI_bull = ROI_center * IM_ROI_red_center
        
        im2, contours, hierarchy = cv2.findContours(ROI_bull.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None , None ,None 

        idx = Image_Tools.getMaxContourIdx(contours)
        bull_contour = contours[idx]
        
        M = cv2.moments(bull_contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        #scale back
        cx = (px - offset) + cx
        cy = (py - offset) + cy
        correction_offset_x = cx - px 
        correction_offset_y = cy - py 
        src_points[0,0] += correction_offset_x
        src_points[3,0] += correction_offset_x
        src_points[1,1] += correction_offset_y
        src_points[2,1] += correction_offset_y
        return src_points, ROI_bull, ROI_center

    def computePerspectiveTransformation(self,contours_structure,BASE_IM_GRAY,BASE_IM_red):
        target_dim = Dartboard.ENV['DART_BOARD_TARGET_DIMENSION']
        pts1 = np.zeros((4,2),'int')
        for i in range(len(contours_structure)):
            cnt = contours_structure[i]
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #print(str(i) + " x: " + str(cx) + " y: " +str(cy))
            pts1[i,0] = cx
            pts1[i,1] = cy
        #print(pts1)
        sorted_points_x = pts1[pts1[:,0].argsort()]
        #print(sorted_points_x)
        left_column = sorted_points_x[0:2,:]
        right_column = sorted_points_x[2:,:]
        sorted_points_y_left = left_column[left_column[:,1].argsort()]
        sorted_points_y_right = right_column[right_column[:,1].argsort()]

        pts_src = np.asarray(np.concatenate((sorted_points_y_left, sorted_points_y_right), axis=0),"float32")
        #print(pts_src)
        pts_dest = np.float32([[0,0],[0,target_dim[1]],[target_dim[1],0],[target_dim[1],target_dim[1]]])
        M = cv2.getPerspectiveTransform(pts_src,pts_dest)

        px,py = Image_Tools.getIntersection(pts_src)
        pts_src_corrected, ROI_bull, ROI_center = self._correctCenterOfBull(BASE_IM_GRAY,BASE_IM_red,px,py,pts_src)
        return M, pts_src_corrected


    def computePerspectiveTransformationPts(self,PTS,BASE_IM_GRAY,BASE_IM_red):
        #PTS: top,bottom,left,right
        target_dim = Dartboard.ENV['DART_BOARD_TARGET_DIMENSION']
        pts_src = np.asarray(([PTS[0][0],PTS[0][1]],[PTS[3][0],PTS[3][1]],[PTS[2][0],PTS[2][1]],[PTS[1][0],PTS[1][1]]),"float32")
 
        diag = np.sqrt(((target_dim[0]/2) * (target_dim[0]/2)) + ((target_dim[1]/2) * (target_dim[1]/2)))
        r = target_dim[0]/2
        offset_diag = diag - r
        offset = np.sqrt((offset_diag*offset_diag)/2)

        #top, right,left,bottom
        pts_dest = np.float32([[0+offset,0+offset],[0+offset,target_dim[1]-offset],[target_dim[1]-offset,0+offset],[target_dim[1]-offset,target_dim[1]]-offset])
        M = cv2.getPerspectiveTransform(pts_src,pts_dest)

        px,py = Image_Tools.getIntersection(pts_src)
        pts_src_corrected, ROI_bull, ROI_center = self._correctCenterOfBull(BASE_IM_GRAY,BASE_IM_red,px,py,pts_src)
        return M, pts_src_corrected



    def computePerspectiveTransformationCorrection(self,pts_src):
        target_dim = Dartboard.ENV['DART_BOARD_TARGET_DIMENSION']
        pts_dest = np.float32([[0,0],[0,target_dim[1]],[target_dim[1],0],[target_dim[1],target_dim[1]]])
        M = cv2.getPerspectiveTransform(pts_src,pts_dest)
        return M


    def warpWithRotation(self,IM_ROI_grey,M):
        target_dim = Dartboard.ENV['DART_BOARD_TARGET_DIMENSION']
        IM_ROI_NORMAL = cv2.warpPerspective(IM_ROI_grey,M,(target_dim[1],target_dim[1]))
        IM_ROI_ROTATED = Image_Tools.rotateImage(IM_ROI_NORMAL,-1*Dartboard.ENV['DART_BOARD_TARGET_ROTATION'])
        return IM_ROI_ROTATED,IM_ROI_NORMAL


    def drawDartboard(self):
        IM = np.zeros(Dartboard.ENV['DART_BOARD_TARGET_DIMENSION'],"uint8")
        offset = Dartboard.ENV['DART_BOARD_TARGET_OFFSET']
        center = (IM.shape[0] // 2,IM.shape[1] // 2)
        size_dartboard = 3400 + offset
        scale = IM.shape[0] / size_dartboard
        rad_board = int(3400 / 2 * scale)
        rad_bull = int(127 / 2 * scale)
        rad_ring = int(318 / 2 * scale)
        rad_double = int((3400 - 160) / 2 * scale)
        rad_triple = int((2140 - 160) / 2 * scale)
        width_rings = int(80 * scale)
        line_thickness = int(12 * scale)
        angle = 360 // 20 
        angle_offset = 9

        #rings
        cv2.circle(IM, center, rad_bull, (255,255,255),line_thickness)
        cv2.circle(IM, center, rad_ring, (255,255,255),line_thickness)
        cv2.circle(IM, center, rad_double, (255,255,255),line_thickness)
        cv2.circle(IM, center, rad_double + width_rings, (255,255,255),line_thickness)
        cv2.circle(IM, center, rad_triple, (255,255,255),line_thickness)
        cv2.circle(IM, center, rad_triple + width_rings, (255,255,255),line_thickness)
        
        #lines  
        line_shape = np.zeros(IM.shape,"uint8")
        line_shape[:,(line_shape.shape[1]//2)-line_thickness:(line_shape.shape[1]//2)+line_thickness] = 255
        IM_temp = np.zeros(IM.shape,"uint8")
        for i in range(0,360,angle):
            line_shape_rot = Image_Tools.rotateImage(line_shape,i + angle_offset)   
            IM_temp = IM_temp + line_shape_rot
        
        #restore bull
        IM_mask = np.zeros(IM.shape,"uint8")
        cv2.circle(IM_mask, center, rad_board, (255,255,255),-1)
        cv2.circle(IM_mask, center, rad_ring, (0,0,0),-1)
        IM = IM + (IM_temp * IM_mask)
        
        #create Mask
        IM_mask = np.zeros(IM.shape,"uint8")
        cv2.circle(IM_mask, center, rad_board, (255,255,255),-1)
        
        #make color
        IM_color = np.repeat(IM[:, :, np.newaxis], 3, axis=2)    
        return IM_color, IM_mask

    def drawFinalDartboard(self,IM_dartboard,IM_dot,IM_dartboard_rotated):
        return IM_dartboard[:,:,0] + IM_dot,IM_dartboard_rotated + IM_dartboard[:,:,0]

    def drawScore(self,IM,score):
        IM = IM.copy()
        cv2.putText(IM,str(score), (0+10,IM.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, Dartboard.ENV['TEXT_SIZE'], 255,Dartboard.ENV['TEXT_THICKNESS'])
        return IM

    def calcScore(self,arrow_angle,length):
        offset = Dartboard.ENV['DART_BOARD_TARGET_OFFSET']
        IM = np.zeros(Dartboard.ENV['DART_BOARD_TARGET_DIMENSION'],"uint8")
        center = (IM.shape[0] // 2,IM.shape[1] // 2)
        size_dartboard = 3400 + offset
        scale = IM.shape[0] / size_dartboard
        rad_board = int(3400 / 2 * scale)
        rad_bull = int(127 / 2 * scale)
        rad_ring = int(318 / 2 * scale)
        rad_double = int((3400 - 160) / 2 * scale)
        rad_triple = int((2140 - 160) / 2 * scale)
        width_rings = int(80 * scale)
        line_thickness = int(12 * scale)
        angle = 360 // 20 
        angle_offset = 9

        NUMBERS = [20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5]

        #is out
        if length > rad_board:
            return -1
        
        #is double
        if length > rad_double and lenght < rad_double + width_rings:
            NUMBERS[:] = [x*2 for x in NUMBERS]
            
        #is triple
        if length > rad_triple and lenght < rad_triple + width_rings:
            NUMBERS[:] = [x*3 for x in NUMBERS]
        
        #is bull
        if length > rad_bull and length < rad_ring:
            return 25
        
        #is bull
        if length < rad_bull:
            return 50 
        
        #calc numbers
        for i in range(len(NUMBERS)):
            if arrow_angle < (i * angle) + angle_offset:
                return NUMBERS[i]
        
        #was 20
        return NUMBERS[0]

class Arrow_Detector:

    ENV = {
        'DETECTION_KERNEL_SIZE' : (100,100),
        'DETECTION_RADIAL_STEP' : 10,
        'DETECTION_KERNEL_THICKNESS' : 1,
        'DETECTION_APEX_OFFSET' : 20, #20
        'DETECTION_APEX_LINE_THICKNESS' : 20, #20
        'DETECTION_APEX_LINE_THICKNESS_PEAK' : 10, #20

        'APEX_CLIPPING_OFFSET' : 50, 
        'APEX_MARK_SIZE' : 10
    }

    def detectArrowState(self,IM_arrow):
        lu = IM_arrow[0:IM_arrow.shape[0]//2,0:IM_arrow.shape[1]//2]
        ru = IM_arrow[0:IM_arrow.shape[0]//2,IM_arrow.shape[1]//2:IM_arrow.shape[1]]
        lb = IM_arrow[IM_arrow.shape[0]//2:IM_arrow.shape[0],0:IM_arrow.shape[1]//2]
        rb = IM_arrow[IM_arrow.shape[0]//2:IM_arrow.shape[0],IM_arrow.shape[1]//2:IM_arrow.shape[1]]
        verbs = [('l','u'),('r','u'),('l','b'),('r','b')]
        stack = [lu,ru,lb,rb]
        max = -1
        maxIdx = 0
        for i in range(len(stack)):
            if np.sum(stack[i]) > max:
                max = np.sum(stack[i])
                maxIdx = i
        #print(verbs[maxIdx])
        return verbs[maxIdx]

    def computeArrowOrientation(self,IM,arange,kernel):
                    max_contour_length = 0
                    max_angle = 0
                    max_contour = 0
                    max_img = 0
                    for i in arange:
                        kernel_rot = Image_Tools.rotateImage(kernel,i)
                        closed = cv2.morphologyEx(IM, cv2.MORPH_CLOSE, kernel_rot)
                        contours, hierarchy = cv2.findContours(closed.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                        for j in range(len(contours)):
                            length = cv2.arcLength(contours[j],True)
                            if length > max_contour_length:
                                max_contour_length = length
                                max_angle = i
                                max_contour = contours[j]
                                max_img = closed
                    return max_contour_length,max_angle,max_contour,max_img

    def _detectArrowLine(self,IM_closed,max_contour,xx,yy,ww,hh):
        # Improve with fitting line
        line_image = np.zeros(IM_closed.shape,"uint8")
        line_image_peak = np.zeros(IM_closed.shape,"uint8")
        
        # then apply fitline() function
        [vx,vy,x,y] = cv2.fitLine(max_contour,cv2.DIST_L2,0,0.01,0.01)

        # Now find two extreme points on the line to draw line
        righty = int((-x*vy/vx) + y)
        lefty = int(((line_image.shape[1]-x)*vy/vx)+y)

        #Finally draw the line
        cv2.line(line_image,(line_image.shape[1]-1,lefty),(0,righty),255,Arrow_Detector.ENV['DETECTION_APEX_LINE_THICKNESS'])
        cv2.line(line_image_peak,(line_image.shape[1]-1,lefty),(0,righty),255,Arrow_Detector.ENV['DETECTION_APEX_LINE_THICKNESS_PEAK'])
        
        #compute orientation
        (h,v) = self.detectArrowState(Image_Tools.getROI(IM_closed,xx,yy,ww,hh))
        if h == 'l':
            if v == 'u':
                arrow_x1 = xx+ww
                arrow_y1 = yy+hh
            else:
                arrow_x1 = xx+ww
                arrow_y1 = yy
        else:
            if v == 'u':
                arrow_x1 = xx
                arrow_y1 = yy+hh
            else:
                arrow_x1 = xx
                arrow_y1 = yy  
        return arrow_x1,arrow_y1,line_image_peak,h,v

    def _detectApex(self,IM_ROI2_grey,line_image_peak,arrow_x1,arrow_y1,h,v):
        # Isolate the apex
        offset = Arrow_Detector.ENV['DETECTION_APEX_OFFSET']
        IM_ROI_APEX = IM_ROI2_grey[arrow_y1-offset:arrow_y1+offset,arrow_x1-offset:arrow_x1+offset]
        IM_ROI_LINE = line_image_peak[arrow_y1-offset:arrow_y1+offset,arrow_x1-offset:arrow_x1+offset] 
        IM_ROI_APEX_edges = cv2.Canny(IM_ROI_APEX,50,100)
        IM_ROI_APEX_masekd = cv2.multiply(IM_ROI_LINE,IM_ROI_APEX_edges)
        
        GUI.imShow(IM_ROI_APEX)
        GUI.imShow(IM_ROI_APEX_edges)

        contours_line, hierarchy_line = cv2.findContours(IM_ROI_APEX_masekd.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours_line) == 0:
            return None, None, None, None,None,None,None,None,None,None
        
        max_contour_idx = Image_Tools.getMaxContourIdx(contours_line)
        xxx,yyy,www,hhh = cv2.boundingRect(contours_line[max_contour_idx])

        #GUI.imShow(Image_Tools.debugRectangle(IM_ROI_APEX_masekd,xxx,yyy,www,hhh))
        #GUI.imShow(IM_ROI_APEX_masekd)

        IM_ROI_APEX_clipped = np.zeros(IM_ROI_APEX_masekd.shape, "uint8")
        IM_ROI_APEX_clipped[yyy:yyy+hhh,xxx:xxx+www] = IM_ROI_APEX_masekd[yyy:yyy+hhh,xxx:xxx+www] 

        IM_ROI_APEX_masekd = IM_ROI_APEX_clipped
        #GUI.imShow(IM_ROI_APEX_clipped)

        # respect orientation
        y,x = np.where(IM_ROI_APEX_masekd > 1)
        np.sort(y)
        #print(h)
        #print(v)
        if h == 'l':
            if v == 'u':
                arrow_y2 = y[y.shape[0]-1]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2,:] > 1)[0]
                np.sort(x)
                arrow_x2 = x[x.shape[0]-1]
            else:
                arrow_y2 = y[0]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2,:] > 1)[0]
                arrow_x2 = x[x.shape[0]-1]
        else:
            if v == 'u':
                arrow_y2 = y[y.shape[0]-1]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2,:] > 1)[0]
                np.sort(x)
                arrow_x2 = x[0]
                #arrow_y2 = yyy
                #arrow_x2 = xxx
            else:
                arrow_y2 = y[0]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2,:] > 1)[0]
                np.sort(x)
                arrow_x2 = x[0]   
        
        # transform to original space
        arrow_y1 = (arrow_y1 - offset) + arrow_y2
        arrow_x1 = (arrow_x1 - offset) + arrow_x2

        return arrow_x1,arrow_y1,IM_ROI_APEX



    def detectArrow(self,diff_image,IM_ROI2_grey):
        kernel_size = Arrow_Detector.ENV['DETECTION_KERNEL_SIZE']
        kernel = np.zeros(kernel_size,np.uint8)
        kernel_thickness = Arrow_Detector.ENV['DETECTION_KERNEL_THICKNESS']
        kernel[:,(kernel.shape[1]//2)-kernel_thickness:(kernel.shape[1]//2)+kernel_thickness] = 1
        max_contour_length,max_angle,max_contour,max_img = self.computeArrowOrientation(diff_image,range(0,180,Arrow_Detector.ENV['DETECTION_RADIAL_STEP']),kernel)
        
        if len(max_contour) == 0:
            return None, None, None, None,None,None,None,None,None,None
                
        xx,yy,ww,hh = cv2.boundingRect(max_contour)

        #Detect line of arrow
        arrow_x1,arrow_y1,line_image_peak,h,v = self._detectArrowLine(max_img,max_contour,xx,yy,ww,hh) 
        
        #Detect apex of arrow
        arrow_x1,arrow_y1,IM_apex = self._detectApex(IM_ROI2_grey,line_image_peak,arrow_x1,arrow_y1,h,v)

        return max_img,arrow_x1,arrow_y1,xx,yy,ww,hh,line_image_peak,IM_apex

    def debugApex(self,IM,arrow_x,arrow_y,color):
        IM = IM.copy()
        clipping_offset = Arrow_Detector.ENV['APEX_CLIPPING_OFFSET']
        cv2.rectangle(IM,(arrow_x,arrow_y),(arrow_x,arrow_y),color,2)
        IM_arrow_roi = IM[arrow_y-clipping_offset:arrow_y+clipping_offset,arrow_x-clipping_offset:arrow_x+clipping_offset]
        #show(IM_arrow_roi)
        return IM_arrow_roi

    def markApex(self,IM_ROI,arrow_x,arrow_y):
        IM_ROI = (IM_ROI.copy() - 2)
        cv2.rectangle(IM_ROI,(arrow_x,arrow_y),(arrow_x,arrow_y),(255,255,255),Arrow_Detector.ENV["APEX_MARK_SIZE"])
        return IM_ROI

    def getMetricOfArrow(self,IM_ROI_ROTATED):
        ret2,thred = cv2.threshold(IM_ROI_ROTATED,254,255,cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(thred.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None,None,None,None,None,None,None

        IM_dot = thred.copy()
        cnt = contours[0]
        MO = cv2.moments(cnt)
        cx = int(MO['m10']/MO['m00'])
        cy = int(MO['m01']/MO['m00'])
        cv2.line(thred,(thred.shape[0]//2,thred.shape[1]//2),(cx,cy),(255,255,255),2)
        IM_line = thred
        dx = cx - (thred.shape[0]//2)  
        dy = cy - (thred.shape[1]//2)
        length = np.sqrt((dx*dx) + (dy*dy))
        nx = 0
        ny = -1
        angle = np.arccos ( ((nx * dx) + (ny*dy)) / length)
        cross = (dx * ny) - (nx * dy)
        if cross > 0:
            angle = (np.pi * 2) - angle
        
        angle = np.rad2deg(angle)
        return cx,cy,angle,length,cross,IM_dot,IM_line

#INITIAL SETUP
IM,BASE_FRAME_GRAY = Image_Tools.readImage('dart1.jpg',(1080,1920))
IM2,BASE_FRAME_GRAY2 = Image_Tools.readImage('dart2.jpg',(1080,1920))

dartboard_detector = Dartboard_Detector()
x,y,w,h,BOARD,GREEN,RED = dartboard_detector.detectDartboard(IM)
GUI.imShow(Image_Tools.debugRectangle(IM,x,y,w,h))
IM_ROI = dartboard_detector.scaleROI(Image_Tools.getROI(IM,x,y,w,h))
IM_ROI_grey = dartboard_detector.scaleROI(Image_Tools.getROI(BASE_FRAME_GRAY,x,y,w,h)) 
IM_ROI_green = dartboard_detector.scaleROI(Image_Tools.getROI(GREEN,x,y,w,h))  
IM_ROI_red = dartboard_detector.scaleROI(Image_Tools.getROI(RED,x,y,w,h))  
IM_ROI_board = dartboard_detector.scaleROI(Image_Tools.getROI(BOARD,x,y,w,h))  
GUI.imShow(IM_ROI)



#for 2nd image which is to be compared. 
GUI.imShow(Image_Tools.debugRectangle(IM2,x,y,w,h))
IM_ROI2 =  dartboard_detector.scaleROI(Image_Tools.getROI(IM2,x,y,w,h))
IM_ROI2_grey =  dartboard_detector.scaleROI(Image_Tools.getROI(BASE_FRAME_GRAY2,x,y,w,h))
GUI.imShow(IM_ROI2)

#compute difference
difference_detector = Difference_Detector()
IM_ROI_difference,IM_ROI_GRAY_NORM,IM_ROI_GRAY2_NORM,IM_ROI_GRAY_NORM_DIFF = difference_detector.computeDifference(IM_ROI_grey,IM_ROI2_grey)
GUI.imShow(IM_ROI_GRAY_NORM_DIFF)
GUI.imShow(IM_ROI_difference)

#arrow detector
arrow_detector = Arrow_Detector()
IM_arrow_closed,arrow_x1,arrow_y1,xxx,yyy,www,hhh, line_image,apex_image = arrow_detector.detectArrow(IM_ROI_difference,IM_ROI2_grey)
GUI.imShow(apex_image)
GUI.imShow(IM_arrow_closed)
#GUI.imShow(Image_Tools.debugRectangle(IM_arrow_closed,xxx,yyy,www,hhh))
#print(arrow_x1)
#print(arrow_y1)


#detect Apex
arrow_detector = Arrow_Detector()
#average arrows
IM_arrow_roi1 = arrow_detector.debugApex(IM_ROI2,arrow_x1,arrow_y1,(0,255,0))
GUI.imShow(IM_arrow_roi1)

#detect scoring zones
dartboard_detector = Dartboard_Detector()
GUI.imShow(IM_ROI)
IM_ROI_thres_color,IM_ROI_thres_color_closed,contours_structure = dartboard_detector.getOrientation(IM_ROI,IM_ROI_board)
GUI.imShow(Image_Tools.debugContours(IM_ROI_thres_color,contours_structure))

#-
dartboard_detector = Dartboard_Detector()
GUI.imShow(IM_ROI)
a, shape_top = Image_Tools.readImage('./images/shape_top.png')
shape_top_hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
GUI.imShow(shape_top_hsv[:,:,0])
IM_ROI_hsv = cv2.cvtColor(IM_ROI, cv2.COLOR_BGR2HSV)
GUI.imShow(IM_ROI_hsv[:,:,0])

res = cv2.matchTemplate(IM_ROI_hsv,shape_top_hsv,cv2.TM_CCORR_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_top_left = max_loc
w = shape_top.shape[1]
h = shape_top.shape[0]

IM_ROI_copy = IM_ROI.copy()
bottom_right = (top_top_left[0] + w, top_top_left[1] + h)
cv2.rectangle(IM_ROI_copy,top_top_left, bottom_right, 255, 2)
GUI.imShow(IM_ROI_copy)


#-
dartboard_detector = Dartboard_Detector()

top_top_left,bottom_top_left,left_top_left,right_top_left,top,bottom,left,right = dartboard_detector.getOrientationCorr(IM_ROI,"./darty/")
IM_ROI_copy = IM_ROI.copy()
print(top_top_left)
w = shape_t.shape[1]
h = shape_t.shape[0]
bottom_right = (top_top_left[0] + w, top_top_left[1] + h)
cv2.rectangle(IM_ROI_copy,top_top_left, bottom_right, 255, 2)
GUI.imShow(IM_ROI_copy)

w = shape_b.shape[1]
h = shape_b.shape[0]
IM_ROI_copy = IM_ROI.copy()
bottom_right = (bottom_top_left[0] + w, bottom_top_left[1] + h)
cv2.rectangle(IM_ROI_copy,bottom_top_left, bottom_right, 255, 2)
GUI.imShow(IM_ROI_copy)

w = shape_l.shape[1]
h = shape_l.shape[0]
IM_ROI_copy = IM_ROI.copy()
bottom_right = (left_top_left[0] + w, left_top_left[1] + h)
cv2.rectangle(IM_ROI_copy,left_top_left, bottom_right, 255, 2)
GUI.imShow(IM_ROI_copy)

w = shape_r.shape[1]
h = shape_r.shape[0]
IM_ROI_copy = IM_ROI.copy()
bottom_right = (right_top_left[0] + w, right_top_left[1] + h)
cv2.rectangle(IM_ROI_copy,right_top_left, bottom_right, 255, 2)
GUI.imShow(IM_ROI_copy)

dartboard = Dartboard()
M, src_points = dartboard.computePerspectiveTransformation(contours_structure,IM_ROI_grey,IM_ROI_red)


dartboard = Dartboard()
M, src_points = dartboard.computePerspectiveTransformationPts((top,bottom,left,right),IM_ROI_grey,IM_ROI_red)


px,py = Image_Tools.getIntersection(src_points)
IM_ROI_bull = Image_Tools.debugIntersection(IM_ROI,src_points)
GUI.imShow(IM_ROI_bull)


dartboard = Dartboard()
IM_RED_NORMAL,IM_ROI_RED_NORMAL = dartboard.warpWithRotation(IM_ROI_red,M)
GUI.imShow(IM_RED_NORMAL)
IM_RED_NORMAL,IM_ROI_RED_NORMAL = dartboard.warpWithRotation(IM_ROI_green,M)
GUI.imShow(IM_RED_NORMAL)



arrow_detector = Arrow_Detector()
IM_ROI2_grey_with_apex = arrow_detector.markApex(IM_ROI2_grey,arrow_x1,arrow_y1)



dartboard = Dartboard()
IM_ROI_ROTATED,IM_ROI_NORMAL = dartboard.warpWithRotation(IM_ROI2_grey_with_apex,M_corrected)
GUI.imShow(IM_ROI_ROTATED)


arrow_detector = Arrow_Detector()
GUI.imShow(IM_ROI_ROTATED)
cx,cy,angle,length,cross,IM_dot,IM_line = arrow_detector.getMetricOfArrow(IM_ROI_ROTATED)
GUI.imShow(IM_dot)
print(angle)
print(length)

dartboard = Dartboard()
IM_dartboard, IM_mask = dartboard.drawDartboard()
GUI.imShow(IM_dartboard)


#follow dart.ipynb for further explanation
