import cv2
import numpy as np

class LaneDetection:
    def __init__(self) -> None:
        pass

    def check_limits(self, value, min_value, max_value):
        value = max(value, min_value)
        value = min(value, max_value)
        return value

    def generate_coords(self, image, line_params):
        slope, intercept = line_params
        y1 = 3*image.shape[0]/4
        y2 = image.shape[0]/2
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)

        x1 = self.check_limits(x1, 0, image.shape[1])
        x2 = self.check_limits(x2, 0, image.shape[1])
        
        return np.array([x1,y1,x2,y2],dtype=np.int32)
        
    def lane_detection(self, image, line_optimized = True, show_intermediate_steps = False):
        # Edge detection
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        cannyImage = cv2.Canny(blur, 40, 150)

        # Region of interest selection
        height,width, _  = image.shape

        spacing = height / 4
        roi = np.array([(0, height - spacing),(width, height - spacing),(10*width/16, height/2),(5*width/16, height/2)],dtype=np.int32)
        roi2 = np.array([(0, height),(width, height),(width, height - spacing),(0, height - spacing)],dtype=np.int32)

        mask = np.zeros_like(cannyImage)
        cv2.fillPoly(mask, [roi, roi2], 255)

        canny_roi_image = cv2.bitwise_and(mask,cannyImage)

        if show_intermediate_steps:
            cv2.imshow("Canny ROI",canny_roi_image)
            cv2.waitKey(1)

        # Line detection
        lines = cv2.HoughLinesP(canny_roi_image, 2, np.pi / 180, 100,np.array([]), minLineLength=40, maxLineGap=5)

        if show_intermediate_steps:
            lines_only = np.zeros_like(image)
            if lines is not None:
                for line in lines:
                    x1,y1,x2,y2 = line.reshape(4)
                    cv2.line(lines_only, (x1,y1),(x2,y2), (255, 0, 0), 10)
            cv2.imshow("Lines Detected",lines_only)
            cv2.waitKey(1)

        # Line optimization
        if line_optimized and lines is not None:
            left_fit = []
            right_fit = []
            slope_th = 0.5
            for line in lines:
                x1,y1,x2,y2 = line.reshape(4)
                slope, intercept = np.polyfit((x1,x2),(y1,y2),1)

                if abs(slope) > slope_th: # Filter Lines
                    if slope < 0 :
                        left_fit.append((slope,intercept))
                    else:
                        right_fit.append((slope,intercept))
            
            lines = []

            if len(left_fit) > 0:
                left_fit_avg = np.average(left_fit, axis=0)
                left_line = self.generate_coords(image,left_fit_avg)
                lines.append(left_line)
            
            if len(right_fit) > 0:
                right_fit_avg = np.average(right_fit, axis=0)
                right_line = self.generate_coords(image,right_fit_avg)
                lines.append(right_line)

        # Show lines
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line.reshape(4)
                cv2.line(line_image, (x1,y1),(x2,y2), (255, 0, 0), 10)
        lane_image = cv2.addWeighted(image, 1.0, line_image, 2.0, 0.0)


        return lane_image, lines