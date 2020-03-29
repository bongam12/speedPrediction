import numpy as np
import argparse
import cv2 

path_mp4 = "train.mp4"
train_txt= "train.txt"
final_result_txt= "train_pred.txt"

def get_speed_data(filename):
    speed=[]
    with open(filename) as f:
        for line in f:
            val = line.rstrip('\n')
            val = float(val)
            speed.append(val)
    return speed

train_speed = get_speed_data(train_txt)
pred_speed = get_speed_data(final_result_txt)

# construct the argument parse 
parser = argparse.ArgumentParser(
    description='Script to run Comma.ai video for speed prediction and  object detection')
parser.add_argument("--video", help="path to video file")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt")
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel")
parser.add_argument("--thr", default=0.15, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()

classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }


net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
count = 0 

font                   = cv2.FONT_HERSHEY_SIMPLEX
position               = (50,50)
position1              = (220,50)

position2              = (50,70)
position3              = (220,70)

fontScale              = 0.7
fontColor              = (0,0,0)
lineType               = 2

cap = cv2.VideoCapture(args.video)
while True:
    
    ret, frame = cap.read()
    frame_resized = cv2.resize(frame,(300,300)) 
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detections = net.forward()
    cols = frame_resized.shape[1] 
    rows = frame_resized.shape[0]

    cv2.rectangle(frame,(40,30),(350,75),(255,0,0),4)

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] 
        if confidence > args.thr: 
            class_id = int(detections[0, 0, i, 1]) 

            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows) #obj location
            
        
            heightFactor = frame.shape[0]/300.0  
            widthFactor = frame.shape[1]/300.0 
            
            xLeftBottom = int(widthFactor * xLeftBottom) 
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop   = int(widthFactor * xRightTop)
            yRightTop   = int(heightFactor * yRightTop)
            
            
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 0, 255),2) #rectangle for the object

            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
                #print(label) 

    cv2.putText(frame,'Actual Speed : ', position, font, fontScale, fontColor, lineType)
    cv2.putText(frame, str(train_speed[count]), position1, font, fontScale, fontColor, lineType)
    
    cv2.putText(frame,'Pred Speed  : ', position2, font, fontScale, fontColor, lineType)
    cv2.putText(frame, str(pred_speed[count]), position3, font, fontScale, fontColor, lineType)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    count+=1

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  #escape key to halt/break
        break
