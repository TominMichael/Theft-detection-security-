import cv2
import os

# Set path to Haar cascade file (make sure the path is correct)
cascade_path = 'C:/Users/Naveen/Python/OpenCV-Face-Recognition/FacialRecognition/haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cascade_path)

# Check if Haar Cascade file loaded correctly
if face_detector.empty():
    print(f"[ERROR] Could not load cascade classifier at {cascade_path}.")
    exit()

# Initialize webcam and set properties
cam = cv2.VideoCapture(0)  # Change to 1 or 2 if 0 doesn't work
if not cam.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# For each person, enter a numeric face ID
face_id = input('\nEnter user ID and press <return>: ')

# Create dataset directory if it doesn't exist
if not os.path.exists("dataset"):
    os.makedirs("dataset")

print("\n[INFO] Initializing face capture. Look at the camera and wait...")
count = 0

while True:
    # Read frame from camera
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Failed to capture image from camera.")
        break

    # Flip the frame horizontally
    img = cv2.flip(img, 1) 

    # Convert frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    print(f"[INFO] Number of faces detected: {len(faces)}")

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save captured face images to the dataset folder
        img_path = f"dataset/User.{face_id}.{count}.jpg"
        cv2.imwrite(img_path, gray[y:y + h, x:x + w])
        print(f"[INFO] Saved image: {img_path}")

        # Show the image with rectangle around the face
        cv2.imshow('image', img)

    # Break the loop if 'ESC' key is pressed or 30 samples are taken
    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27 or count >= 300:  # Take 30 face samples and stop video
        break

# Cleanup resources
print("\n[INFO] Exiting Program and cleaning up...")
cam.release()
cv2.destroyAllWindows()

''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18    

'''

import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('C:/Users/Naveen/Python/OpenCV-Face-Recognition/FacialRecognition/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


'''
Real Time Face Recognition
    ==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
    ==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

'''

import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "C:/Users/Naveen/Python/OpenCV-Face-Recognition/FacialRecognition/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize id counter
id = 0

# Names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Naveen', 'enemy', 'Ajmal', 'Kavya', 'W'] 

# Initialize and start real-time video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1) # Flip vertically

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less than 100 ==> "0" is a perfect match 
        if confidence < 100:
            if id < len(names):
                id = names[id]  # Safely access the names list
            else:
                id = "Unknown"  # Handle case where id exceeds the names list
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "Unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)  
    
    cv2.imshow('camera', img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
