import cv2
import os
import numpy as np
from PIL import Image

# Paths for saving model and dataset
cascade_path = 'C:/Users/Naveen/Python/OpenCV-Face-Recognition/FacialRecognition/haarcascade_frontalface_default.xml'
dataset_path = 'dataset'
trainer_dir = 'trainer'
trainer_file = os.path.join(trainer_dir, 'trainer.yml')

# Create trainer directory if it doesn't exist
if not os.path.exists(trainer_dir):
    os.makedirs(trainer_dir)

# Initialize face recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier(cascade_path)

# Function to train the model
def train_model():
    # Get the images and label data
    print("\n[INFO] Training faces. This may take a few seconds...")
    faces, ids = getImagesAndLabels(dataset_path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer.yml
    recognizer.write(trainer_file)
    print(f"\n[INFO] {len(np.unique(ids))} faces trained. Exiting Program")

# Function to get images and labels for training
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
    faceSamples, ids = [], []

    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert('L')  # Convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = face_detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        except Exception as e:
            print(f"[ERROR] Could not process file {imagePath}: {e}")
            continue

    return faceSamples, ids

# Function to capture new face data for unknown faces
def capture_new_face(face_id):
    print("\n[INFO] Initializing face capture. Look at the camera and wait...")
    cam = cv2.VideoCapture(0)  # Open camera
    cam.set(3, 640)  # Set video width
    cam.set(4, 480)  # Set video height

    count = 0
    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Failed to capture image from camera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # Check if any faces are detected
        if len(faces) > 0:
            print(f"[INFO] Face detected. Number of faces: {len(faces)}")

        for (x, y, w, h) in faces:
            count += 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Save the captured image into the dataset folder
            img_path = f"dataset/User.{face_id}.{count}.jpg"
            cv2.imwrite(img_path, gray[y:y + h, x:x + w])
            print(f"[INFO] Captured image: {img_path} - Total images captured: {count}")

            # Display the image with the detected face
            cv2.imshow('New Face Registration', img)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' to exit
        if k == 27 or count >= 150:  # Capture 150 face samples
            break

    print("\n[INFO] Exiting capture and releasing camera...")
    cam.release()
    cv2.destroyAllWindows()  # Close the registration window

# Function to recognize faces
def recognize_and_register():
    # Initialize webcam and set properties
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # Set video width
    cam.set(4, 480)  # Set video height

    recognizer.read(trainer_file)  # Load the trained model

    print("\n[INFO] Starting face recognition...")
    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Failed to capture image from camera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less than 50 --> "Recognized", otherwise "Unknown"
            if confidence < 50:
                name = f"User {id}"
                confidence_text = f"  {round(100 - confidence)}%"
            else:
                name = "Unknown"
                confidence_text = f"  {round(100 - confidence)}%"

            cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, confidence_text, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

            # If face is unknown, register it
            if name == "Unknown":
                print("\n[INFO] Unknown face detected. Switching to registration...")
                cam.release()
                cv2.destroyAllWindows()  # Close the recognition window
                new_id = input("\nEnter a new user ID for this face: ")
                capture_new_face(new_id)  # Capture new face data
                train_model()  # Retrain model with new data
                print("\n[INFO] Switching back to recognition...")
                recognize_and_register()  # Restart recognition
                return  # Exit the function to avoid duplicate windows

        # Show recognition window
        cv2.imshow('Face Recognition', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' to exit
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()  # Close all windows

# Start the program
recognize_and_register()
