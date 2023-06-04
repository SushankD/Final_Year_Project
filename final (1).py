import sys
import cv2
import numpy 
import time
from random import randint
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow import keras



class Laser(object):

    def __init__(self, cam_width=320, cam_height=240, hue_min=100, hue_max=160,
                 sat_min=100, sat_max=255, val_min=200, val_max=256,
                 display_thresholds=False):

        self.cam_width = cam_width
        self.cam_height = cam_height
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self.display_thresholds = display_thresholds

        self.capture = None  # camera capture device
        self.channels = {
            'hue': None,
            'saturation': None,
            'value': None,
            'laser': None,
        }

        self.previous_position = None
        self.trail = numpy.zeros((self.cam_height, self.cam_width, 3),
                                 numpy.uint8)


        ''' def __init__(self, master):
            self.master = master
            self.master.title("Camera App")
            self.video_source = 0
            self.cap = None
            self.canvas = None
            self.img = None

            # Create GUI elements
            self.start_button = tk.Button(self.master, text="Start", command=self.start_video)
            self.start_button.pack(padx=10, pady=10)'''

    def setup_camera_capture(self, device_num=0):
        """Perform camera setup for the device number (default device = 0).
        Returns a reference to the camera Capture object.
        """
        try:
            device = int(device_num)
            sys.stdout.write("Using Camera Device: {0}\n".format(device))
        except (IndexError, ValueError):
            # assume we want the 1st device
            device = 0
            sys.stderr.write("Invalid Device. Using default device 0\n")

        # Try to start capturing frames
        self.capture = cv2.VideoCapture(device)
        if not self.capture.isOpened():
            sys.stderr.write("Faled to Open Capture device. Quitting.\n")
            sys.exit(1)

        return self.capture
    
    def none(args):
        pass

    def getShapeName(edges, width, height):
        name = ''
        aspectRatio = float(height) / width

        if edges == 3:
            name = 'Triangle'
        elif edges == 4:
            if 0.9 < aspectRatio < 1.1:
                name = 'Square'
            else: 
                name = 'Rectangle'
        elif edges > 30:
            name = 'Unknown'
        else:
            if 0.9 < aspectRatio < 1.1:
                name = 'Circle'
            else:
                name = 'Polygon'
        return name

    def calcContoursArea(contours, minArea):
        areas = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > minArea:
                areas.append(area)
        list.sort(areas, reverse=True)
        return areas


    def defineContours(img, imgContour):
        contours, hierarchy =  cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        minArea = cv2.getTrackbarPos('MinArea', 'Parameters')
        areas = Laser.calcContoursArea(contours, minArea)
        for c in contours:
            contourArea = cv2.contourArea(c)
            if contourArea > minArea:
                areaPos = areas.index(contourArea)+1

                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.03 * peri, True)

                x, y, w, h = cv2.boundingRect(approx)
                name = Laser.getShapeName(len(approx), w, h)
                
                cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(imgContour, name, (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), thickness=2)
                cv2.putText(imgContour, 'Area #{0}'.format(areaPos), (x+w//2-60, y+h//2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), thickness=2)

                #cv2.drawContours(imgContour, c, -1, (255, 255, 255), 2)
                #cv2.drawContours(imgContour, [approx], -1, (255, 255, 255), 3)
        return imgContour

    def handle_quit(self, delay=10):
        """Quit the program if the user presses "Esc" or "q"."""
        key = cv2.waitKey(delay)
        c = chr(key & 255)
        if c in ['c', 'C']:
            self.trail = numpy.zeros((self.cam_height, self.cam_width, 3),
                                     numpy.uint8)
        if c in ['q', 'Q', chr(27)]:
            sys.exit(0)

    def threshold_image(self, channel):
        if channel == "hue":
            minimum = self.hue_min
            maximum = self.hue_max
        elif channel == "saturation":
            minimum = self.sat_min
            maximum = self.sat_max
        elif channel == "value":
            minimum = self.val_min
            maximum = self.val_max

        (t, tmp) = cv2.threshold(
            self.channels[channel],  # src
            maximum,  # threshold value
            0,  # we dont care because of the selected type
            cv2.THRESH_TOZERO_INV  # t type
        )

        (t, self.channels[channel]) = cv2.threshold(
            tmp,  # src
            minimum,  # threshold value
            255,  # maxvalue
            cv2.THRESH_BINARY  # type
        )

        if channel == 'hue':
            # only works for filtering red color because the range for the hue
            # is split
            self.channels['hue'] = cv2.bitwise_not(self.channels['hue'])


    def detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # split the video frame into color channels
        h, s, v = cv2.split(hsv_img)
        self.channels['hue'] = h
        self.channels['saturation'] = s
        self.channels['value'] = v

        # Threshold ranges of HSV components; storing the results in place
        self.threshold_image("hue")
        self.threshold_image("saturation")
        self.threshold_image("value")

        # Perform an AND on HSV components to identify the laser!
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['value'],
            self.channels['saturation']
        )
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['hue'],
            self.channels['laser']
        )
            # Find contours of laser lines
        contours, _ = cv2.findContours(
            self.channels['laser'],
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        '''num_lines = len(contours)

        if num_lines == 1:
            #print(f"{num_lines} laser line(s) detected")
            print("No Obstacle")
        elif num_lines>1:
            print("Obstacle detected")
        else:
            print("laser_line not detected")
        '''
        




        # Merge the HSV components back together.
        hsv_image = cv2.merge([
            self.channels['hue'],
            self.channels['saturation'],
            self.channels['value'],
        ])

        return hsv_image
        
        

    def display(self, img ,frame):
        """Display the combined image and (optionally) all other image channels
        NOTE: default color space in OpenCV is BGR.
        """
        #cv2.imshow('RGB_VideoFrame', frame)
        #cv2.imshow('LaserPointer', self.channels['laser'])
        '''if self.display_thresholds:
            cv2.imshow('Thresholded_HSV_Image', img)
            cv2.imshow('Hue', self.channels['hue'])
            cv2.imshow('Saturation', self.channels['saturation'])
            cv2.imshow('Value', self.channels['value'])'''


    

    def run(self):
        self.setup_camera_capture()

        dnn = cv2.dnn.readNet('E:\OpenCV-Object-Detection-main\yolov4-tiny.weights', 'E:\OpenCV-Object-Detection-main\yolov4-tiny.cfg')
        model = cv2.dnn_DetectionModel(dnn)
        model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

        with open('E:\OpenCV-Object-Detection-main\classes.txt') as f:
            classes = f.read().strip().splitlines()

        color_map = {}

        #cap = cv2.VideoCapture(0)
        cv2.namedWindow('Parameters')
        cv2.resizeWindow('Parameters', 640, 240)
        cv2.createTrackbar('Treshhold1', 'Parameters', 45, 255,Laser.none)
        cv2.createTrackbar('Treshhold2', 'Parameters', 60, 255,Laser.none)
        cv2.createTrackbar('MinArea', 'Parameters', 8000, 10000,Laser.none)

        # Define the folder paths and labels
        labels = ['No Obstacle Detected', 'Obstacle Detected', 'Laser Line Not Detected']

        # Load the trained model
        model1 = keras.models.load_model('laser_line_detection_model_new')



        while True:
            # 1. capture the current image
            success, frame = self.capture.read()
            if not success:  # no image captured... end the processing
                sys.stderr.write("Could not read camera frame. Quitting\n")
                sys.exit(1)

            
            ret, fram = self.capture.read()

            class_ids, confidences, boxes = model.detect(fram)
            for id, confidence, box in zip(class_ids, confidences, boxes):
                x, y, w, h = box
                obj_class = classes[id]

                if obj_class not in color_map:
                    color = (randint(0, 10), randint(0, 10), randint(0, 10))
                    color_map[obj_class] = color
                else:
                    color = color_map[obj_class]

                cv2.putText(fram, f'{obj_class.title()} {format(confidence, ".2f")}', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                cv2.rectangle(fram, (x, y), (x + w, y + h), color, 2)

            #cv2.imshow('model',fram)


            imgsa,imgs = self.capture.read()
            imgContour = imgs.copy()

            imgGray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (9, 9), 0)

            threshold1 = cv2.getTrackbarPos('Treshhold1', 'Parameters')
            threshold2 = cv2.getTrackbarPos('Treshhold2', 'Parameters')

            imgBrdrs = cv2.Canny(imgBlur, 45, 60)
            kernel = numpy.ones((5, 5))
            imgClosed = cv2.dilate(imgBrdrs, kernel, iterations=1)

            Laser.defineContours(imgClosed, imgContour)

            #cv2.imshow('Result', imgContour)

            #ret, fram = self.capture.read()
            # Preprocess the frame and make a prediction on it
            imz = cv2.resize(frame, (224, 224)) / 255.0
            imz = numpy.expand_dims(imz, axis=0)
            predictions = model1.predict(imz)
                
                # Get the predicted label and add it on top of the frame
            predicted_class_index = numpy.argmax(predictions[0])
            predicted_label = labels[predicted_class_index]
            self.channels['laser'] = cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the frame with the predicted label
            cv2.imshow('Laser Line Detection', fram)       
    


            

            hsv_image = self.detect(frame)
            self.display(hsv_image, frame,)
            self.handle_quit()

obsdec = Laser()

obsdec.run()      
        

