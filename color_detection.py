import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Open the webcam
stream = cv2.VideoCapture(0)

while True:
    # Capture frame from the webcam
    ret, frame = stream.read()

    # Convert the frame from BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the yellow color
    lowerlimit = np.array([20, 75, 75])
    upperlimit = np.array([30, 255, 255])

    # Apply the mask to detect the yellow object
    mask = cv2.inRange(hsv_frame, lowerlimit, upperlimit)

    # Find the contours of the yellow object
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a bounding box around the yellow object
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all windows
stream.release()
cv2.destroyAllWindows()
