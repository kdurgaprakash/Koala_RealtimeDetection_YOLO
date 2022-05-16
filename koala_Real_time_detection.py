# Import all the dependancies required for object detection and localization

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading a network model stored in Darknet model files and loading configuration file and weights trained from google colab
# Input correct path for configuration file and weights for the below code
net = cv2.dnn.readNetFromDarknet("path of yolov3_custom.cfg","path of trained weights")

# Creating a list for classes

classes = ['koala']

# Load webcam for capturing live video or input video

cap = cv2.VideoCapture(0)

# For uploading a video uncomment below code
# cap = cv2.VideoCapture("Path of recorded Koala video")

while 1:
    # capture frames of video
    _, frame = cap.read()
    frame = cv2.resize(frame,(1280,720))
    height,width,channels = frame.shape

    # create 4 dimensional blob from image by resizing the image to 416*416
    blob = cv2.dnn.blobFromImage(frame, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

    # setting the new input value for the network
    net.setInput(blob)

    # Returns the indexes of layers with unconnected outputs
    output_layers_name = net.getUnconnectedOutLayersNames()

    # Runs a forward pass in the network to compute the net ouput
    output_layers = net.forward(output_layers_name)

    # Create lists for storing bounding box information including image centre, size and confidence score
    boxes =[]
    confidences = []
    class_ids = []

    # Store output values from the output layer network in different lists
    for output in output_layers:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]

            if confidence > 0.7:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                w = int(detection[2] * width)
                h = int(detection[3]* height)

                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    # Perform non maximum suppression to remove extra bounding boxes by setting a minimum threshold of confidence scores
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.5,.4)

    # Define random colors and font for bounding box and confidence score text
    colors = np.random.uniform(0,255,size =(len(boxes),3))
    font = cv2.FONT_HERSHEY_PLAIN

    # Create bounding boxes 
    if  len(indexes)>0:
        for i in indexes.flatten():
            #Get bounding box info for drawing and showing confidence score
            x,y,w,h = boxes[i]
            color = colors[i]
            confidence = str(round(confidences[i],2))
            label = str(classes[class_ids[i]])

            #Draw rectangles for detected koala bear
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            #Show confidence score of detected koala  bear
            cv2.putText(frame,label + " " + confidence, (x,y+400),font,2,color,2)
    
    # continue showing video until key "k" is pressed
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('k'):
        break
    
cap.release()
cv2.destroyAllWindows()
