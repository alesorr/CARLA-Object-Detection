#==================== OBJECT DETECTION FUNCTION AND UTILS ==================

import cv2
import numpy as np

PATHDICT = {
    'alex': "yolov3/",
    'daniele' : "C:/Users/danis/Desktop/AutonomousVehicleDriving/Carla/python_client_folder/CARLA-obstacle-detection/yolov3/",
    'christian' : "",
    'francesco' : ""
}

PATH = PATHDICT['alex']

class Yolo:

    def __init__(self):      
        self.confThreshold = 0.55
        self.nmsThreshold = 0.40
        self.inpWidth = 416
        self.inpHeight = 416
        # load names of classes and turn that into a list
        self.classesFile = PATH + "coco.names"
        self.classes = None
        # model configuration
        self.modelConf = PATH + 'yolov3.cfg'
        self.modelWeights = PATH + 'yolov3.weights'
        self.net = cv2.dnn.readNetFromDarknet(self.modelConf, self.modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        with open(self.classesFile,'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        # process inputs
        self.winName = 'DL OD with OpenCV'
        cv2.namedWindow(self.winName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.winName, 300,300)
        

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIDs = []
        confidences = []
        boxes = []

        # iterate 
        for out in outs:
            for detection in out:
                
                scores = detection [5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.confThreshold:
                    centerX = int(detection[0] * frameWidth)
                    centerY = int(detection[1] * frameHeight)

                    width = int(detection[2]* frameWidth)
                    height = int(detection[3]*frameHeight )

                    classIDs.append(classID)
                    confidences.append(float(confidence))
                    boxes.append([centerX, centerY, width, height])

        # create a list of indices
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)

        # iterate through indices to draw bounding boxes around detected objects
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            
            self.drawPred(classIDs[i], confidences[i], left, top, left + width, top + height, frame, frameWidth)

            return classIDs[i][0], box

    def get_optimal_font_scale(self, text, width):
        for scale in reversed(range(0, 60, 1)):
            textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
            new_width = textSize[0][0]
            if (new_width <= width):
                return scale/10

    # function for drawing bounding boxes around detected objects
    def drawPred(self, classId, conf, left, top, right, bottom, frame, frameWidth):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.classes:
            assert (classId < len(self.classes))
            if classId == 1 or classId == 3 or classId == 5 or classId == 7:
                classId = 2
            label = '%s:%s' % (self.classes[classId], label)
        
        cv2.putText(frame, label, (left,top), cv2.FONT_HERSHEY_SIMPLEX, self.get_optimal_font_scale(label, frameWidth), (255, 255, 255), 3)

    # function for getting the names outputted in the output layers of the net
    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
    
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
