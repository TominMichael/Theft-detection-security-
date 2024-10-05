import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import cv2
import numpy as np
import os
from fer import FER  # Import the FER library for emotion detection

# Initialize the face recognizer and emotion detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "C:/Users/Naveen/Python/OpenCV-Face-Recognition/FacialRecognition/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize id counter
id = 0

# Names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Naveen', 'enemy', 'Ajmal', 'Kavya', 'W'] 

# Initialize the emotion detector
emotion_detector = FER()

# Initialize and start real-time video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Flip vertically

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Recognize face
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less than 100 ==> "0" is a perfect match 
        if confidence < 100:
            if id < len(names):
                id = names[id]  # Safely access the names list
            else:
                id = "Unknown"  # Handle case where id exceeds the names list
            confidence_text = "  {0}%".format(round(100 - confidence))
        else:
            id = "Unknown"
            confidence_text = "  {0}%".format(round(100 - confidence))

        # Crop the face for emotion detection
        face_img = img[y:y + h, x:x + w]
        # Detect emotions
        emotions = emotion_detector.detect_emotions(face_img)
        # Get the dominant emotion if any emotions are detected
        if emotions:
            dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
        else:
            dominant_emotion = "None"

        # Display the name, confidence, and emotion on the image
        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        cv2.putText(img, f'Emotion: {dominant_emotion}', (x + 5, y + h + 25), font, 1, (36, 255, 12), 2)

    cv2.imshow('camera', img)

    # Press 'ESC' for exiting video
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# Cleanup resources
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
