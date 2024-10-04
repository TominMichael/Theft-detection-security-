import tensorflow as tf
import cv2
import numpy as np

# Load the .h5 model (TensorFlow/Keras)
model = tf.keras.models.load_model("new.h5")

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Function to preprocess and perform inference
def run_inference(frame):
    # Preprocess the frame to the required input shape
    input_image = cv2.resize(frame, (150, 150))
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    input_image = input_image.astype(np.float32) / 255.0  # Normalize

    # Perform inference using the loaded .h5 model
    result = model.predict(input_image)
    return np.argmax(result, axis=1)[0]  # Get the class with the highest probability

# Start Video Capture and Process Frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame and run inference
    prediction = run_inference(frame)

    # Postprocess and annotate the frame based on the prediction
    if prediction == 0:
        label = "Weapon Detected"
    elif prediction == 1:
        label = "None"
    elif prediction == 2:
        label = "Mask Detected"

    # Display the label on the video frame
    cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the result
    cv2.imshow("Detection", frame)

    # Check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

