#experimental


import cv2

# Opencv DNN
net = cv2.dnn.readNet("/Users/leonfray/Desktop/projects/objectDetection/dnn_model/yolov4-tiny.weights", "/Users/leonfray/Desktop/projects/objectDetection/dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)

model.setInputParams(size=(320, 320), scale=1/255)

classes = []
with open("/Users/leonfray/Desktop/projects/objectDetection/dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()

    # Object detection
    (class_ids, scores, bboxes) = model.detect(frame)

    # Draw detected objects
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        x, y, w, h = bbox
        class_name = classes[class_id[0]]

        # Draw frame and text for each detected object
        cv2.putText(frame, str(class_name), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

    cv2.imshow("Frame", frame)

    # Wait for a key press, change frames every 1ms
    cv2.waitKey(1)
