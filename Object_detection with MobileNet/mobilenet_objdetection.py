import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'ssd_mobilenet_v3_large_coco_2020_01_14.pb')

# Initialize the webcam
cap = cv2.VideoCapture(0)

def detect_objects(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and get the detections
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the prediction on the frame
            label = f"{CLASSES[idx]}: {confidence:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_objects(frame)
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
