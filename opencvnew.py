import cv2
import time
import numpy as np
import tensorflow as tf
from telegram import Bot
import asyncio
import os

# Define Telegram bot token and chat_id
bot_token = '7948180888:AAFZcoOo_FJY2VeE2NY5kWWc9geeygUI0IQ'
chat_id = '1161319551'

# Load the pre-trained model
model = tf.keras.models.load_model('mask_weapon_none_detection_modelwithmaazanew.h5')

# Initialize Telegram bot
bot = Bot(token=bot_token)

# Create directories for storing videos if they don't exist
os.makedirs("weapon_detected", exist_ok=True)
os.makedirs("mask_detected", exist_ok=True)

# Function to send a message via Telegram
async def send_telegram_message(message):
    try:
        await bot.send_message(chat_id=chat_id, text=message)
        print("Telegram message sent!")
    except Exception as e:
        print(f"Failed to send message: {e}")

# Initialize webcam
cap = cv2.VideoCapture(0)



# Initialize variables for recording control
weapon_detected_count = 0
mask_detected_count = 0
detection_threshold = 20  # Frames before recording starts
no_detection_threshold = 10  # Frames to stop recording if no detection

recording_weapon = False
recording_mask = False
weapon_frames_without_detection = 0
mask_frames_without_detection = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Main loop to capture frames and perform detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize and normalize the frame
    img = cv2.resize(frame, (150, 150))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Perform prediction
    prediction = model.predict(img)
    label = np.argmax(prediction, axis=1)

    # Define the label text based on the prediction
    if label == 1:  # Weapon detected
        label_text = "mask"
    elif label == 0:  # Mask detected
        label_text = "none"
    else:
        label_text = "weapon"

    # Overlay the label on the frame
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Logic for handling weapon detection
    if label == 1:  # Weapon detected
        weapon_detected_count += 1
        mask_detected_count = 0  # Reset mask detection count
        weapon_frames_without_detection = 0

        if weapon_detected_count >= detection_threshold and not recording_weapon:
            print("Starting weapon detection recording...")
            video_writer = cv2.VideoWriter(f'weapon_detected/weapon_{int(time.time())}.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording_weapon = True
            asyncio.run(send_telegram_message("Weapon detected!"))

        if recording_weapon:
            video_writer.write(frame)
    else:
        weapon_detected_count = 0
        if recording_weapon:
            weapon_frames_without_detection += 1
            if weapon_frames_without_detection >= no_detection_threshold:
                print("Stopping weapon detection recording...")
                video_writer.release()
                recording_weapon = False

    # Logic for handling mask detection
    if label == 0:  # Mask detected
        mask_detected_count += 1
        weapon_detected_count = 0  # Reset weapon detection count
        mask_frames_without_detection = 0

        if mask_detected_count >= detection_threshold and not recording_mask:
            print("Starting mask detection recording...")
            video_writer = cv2.VideoWriter(f'mask_detected/mask_{int(time.time())}.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording_mask = True
            asyncio.run(send_telegram_message("Mask detected!"))

        if recording_mask:
            video_writer.write(frame)
    else:
        mask_detected_count = 0
        if recording_mask:
            mask_frames_without_detection += 1
            if mask_frames_without_detection >= no_detection_threshold:
                print("Stopping mask detection recording...")
                video_writer.release()
                recording_mask = False

    # Show the frame with the label
    cv2.imshow('Frame', frame)

    # Add some exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

