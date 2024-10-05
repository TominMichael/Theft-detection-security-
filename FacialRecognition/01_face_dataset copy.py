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
    if k == 27 or count >= 1000:  # Take 30 face samples and stop video
        break

# Cleanup resources
print("\n[INFO] Exiting Program and cleaning up...")
cam.release()
cv2.destroyAllWindows()
