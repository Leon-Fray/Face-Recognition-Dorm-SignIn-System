#main

import cv2 

#Opencv DNN 
net = cv2.dnn.readNet("/Users/leonfray/Desktop/projects/objectDetection/dnn_model/yolov4-tiny.weights", "/Users/leonfray/Desktop/projects/objectDetection/dnn_model/yolov4-tiny.cfg")
model =cv2.dnn_DetectionModel(net)

# 12) NOTES: the objects are always scaled down, while the 
# size should also be a multiple of 32, hence we use 
# 320x320. While a bigger size would be more accurate 
# it would also be far slower. So we need to find an 
# appropriate sweet spot. 

# 13) It is scaled to 1/255 because one of the things 
# has a range of 0-1 while the other must be multipled 
# of 255 (i think)

model.setInputParams(size=(320,320), scale =1/255 )

#Load Class list 

classes = []

with open ("/Users/leonfray/Desktop/projects/objectDetection/dnn_model/classes.txt", "r") as file_object: 
    for class_name in file_object.readlines(): 
        class_name = class_name.strip() 
        classes.append(class_name)


#Initialize Camera 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#full hd 1920 x 1080

#step 1: getting the frame from the webcam
# we use different indexes like 0,1,2 this is for the different web cams 
cap = cv2.VideoCapture(0)  


# 2) Notes: ret is a boolean variable that is used to indicate 
# wheteher the frame was succesfully read or not. 
# so it would be True if the frame is captured 
# and false if there are no frames to read 

# 3) cap.read() reads the next video frame from the video source 
# and frame is the variable that holds the acutal image data 
# of the capture frame 

while True:    # 10) loop to allow a live feed of frame capture 
    ret, frame = cap.read() 


    #object detection 
    (class_ids, scores, bboxes)= model.detect(frame) 
    for class_ids, scores, bboxes in zip(class_ids,scores,bboxes): 
        (x,y,w,h)  = bboxes
        class_name = classes[class_ids]

        cv2.putText(frame, str(class_name), (x,y-10),cv2.FONT_HERSHEY_PLAIN, 2, (200,0,50), 2) 
        cv2.rectangle(frame, (x,y), (x+w,y+h), (200,0,50), 3) 


    print("class ids", class_ids) 
    print("scores", scores)
    print("bboxes", bboxes)

# 4) "Frame" is simply the name of the window and you can name 
# it anything you like 

# 5)This line is actually displaying the captured image 

# 6) it is using imshow() to do this. 
# 7) imshow() takes two parameters, the Title and the iamge you
# you want to be displayed 

    cv2.imshow("Frame", frame )


# 8) The program needs a waitkey as otherwise it will execute and 
# then close immediately after 



    cv2.waitKey(1) 

# with waitKey(0), it is waiting for a key press to change 
# frames, hence it is moving a bit choppy. 
# We can change it to waitKey(1) and instead it will only 
# wait 1ms before changing frames



# 9) NOTES: waitKey() allows the frame to stay longer, however it 
# only captures one image, what we want is a live feed. 
# Hence we would need to run the cap.read() through a loop 