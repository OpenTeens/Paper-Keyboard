import cv2
from locate import process

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# To capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()

    # Process the frame
    process(img)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:     # ESC
        break

# Release the VideoCapture object
cap.release()
