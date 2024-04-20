import cv2
import numpy as np

def main():
    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame from BGR to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range of green color in HSV
        lower_green = np.array([50, 100, 100])
        upper_green = np.array([70, 255, 255])

        # Create a mask with the specified green range
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding box around the detected green object
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 400:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the result
        cv2.imshow('Green object tracking', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
