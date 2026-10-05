import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:

    ret, frame = cam.read()

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(frame, (21, 21), 0)
    #blurred = cv2.resize(blurred, [0, 0], fx= 1.5, fy= 1.5)

    # Apply Sharpening filter
    kernel = np.array([[-1, -1, -1],
                      [-1, 15, -1],
                      [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(frame, ddepth=-1, kernel=sharpen_filter)

    # Apply Sobel filter
    sobel = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #sobel = cv2.GaussianBlur(frame,ksize=(1,1), sigmaX=0, sigmaY=0)
    sobel_filter = cv2.Laplacian(sobel, ddepth=-1)

    # Apply Brightness Filter
    brightness = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    brightness = brightness * 2
    brightness_filter = cv2.filter2D(frame, ddepth=-1, kernel=brightness)


    #Apply Negative Filter
    negative = 255 - frame

    cv2.imshow('Original', frame)
    cv2.imshow('Blurred', blurred)
    cv2.imshow('Sharpened', sharpened)
    cv2.imshow('Sobel', sobel_filter)
    cv2.imshow('Brightness', brightness_filter)
    cv2.imshow('Negative', negative)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
