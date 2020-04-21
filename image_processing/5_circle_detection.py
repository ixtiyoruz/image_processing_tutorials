# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:00:15 2020

@author: essys
"""
import cv2
import numpy as np

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 200;

# Filter by Area.
params.filterByArea = True
params.minArea = 500
params.maxArea = 20000
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.9

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1

params.filterByColor  = True
params.blobColor =0

detector = cv2.SimpleBlobDetector_create(params)
    

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
     # Detect blobs 
    keypoints = detector.detect(gray) 

    # Draw blobs on our image as red circles 
    blank = np.zeros((1, 1))  
    img = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255), 
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 

#    number_of_blobs = len(keypoints) 
    text = "Number of Circular Blobs: " + str(len(keypoints)) 
    cv2.putText(img, text, (20, 550), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2) 
    cv2.imshow("Original", img)
    key = cv2.waitKey(3)
    if(key == 27):
        cv2.destroyAllWindows()
        cap.release();
        break
