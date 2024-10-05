import tensorflow as tf
import cv2
import numpy as np
import time
from telegram import Bot

# Load the .h5 model (TensorFlow/Keras)
model = tf.keras.models.load_model("newnew.h5")

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Telegram bot initialization
telegram_bot_token = "7948180888:AAFZcoOo_FJY2VeE2NY5kWWc9geeygUI0IQ"  # Replace with your bot's token
chat_id = "1161319551"  # Replace with your chat ID
bot = Bot(token=telegram_bot_token)

# VideoWriter object to record video when a weapon is detected
out = None
recording = False

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
        
        # Send a Telegram message if a weapon is detected
        bot.send_message(chat_id=chat_id, text="Warning: Weapon detected!")
        
        # Start recording if not already recording
        if not recording:
            recording = True
            # Set up the VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = time.strftime("%Y%m%d-%H%M%S")  # Create a unique filename based on the timestamp
            out = cv2.VideoWriter(f'weapon_detected_{timestamp}.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            print(f"Started recording: weapon_detected_{timestamp}.avi")

        # Write the frame to the video file
        if out:
            out.write(frame)

    elif prediction == 1:
        label = "None"
        
        # Stop recording when no weapon is detected and was previously recording
        if recording:
            recording = False
            if out:
                out.release()
                out = None
            print("Stopped recording")

    elif prediction == 2:
        label = "Mask Detected"
        
        # Send a Telegram message if a mask is detected
        bot.send_message(chat_id=chat_id, text="Mask detected.")

    # Display the label on the video frame
    cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the result
    cv2.imshow("Detection", frame)

    # Check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
if out:
    out.release()
cap.release()
cv2.destroyAllWindows()
