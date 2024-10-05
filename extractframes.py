
#for extracting frames from video to make dataset



import cv2
import os

# Load the video
video_path = 'maskapi.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Create a directory to save frames
output_dir = 'extracted_frames'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save each frame as an image file
    frame_filename = f"{output_dir}/maskframesssss_{frame_count}.jpg"
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
print(f"Extracted {frame_count} frames from the video.")

