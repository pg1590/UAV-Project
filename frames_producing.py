import cv2
import os

video_path = "/home/prakhargupta/Desktop/M26_Spring/UAV/Project/video_2026-05-06_23-54-22 (online-video-cutter.com).mp4"   # <-- change this
output_dir = "extracted_frames"

num_frames_to_extract = 1000

# Create output folder
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Total frames in video: {total_frames}")
print(f"FPS: {fps}")

# Calculate interval
frame_interval = total_frames // num_frames_to_extract
print(f"Saving every {frame_interval} frame")

count = 0
saved_count = 0

while cap.isOpened() and saved_count < num_frames_to_extract:
    ret, frame = cap.read()

    if not ret:
        break

    if count % frame_interval == 0:
        filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    count += 1

cap.release()
print(f"Saved {saved_count} frames to '{output_dir}'")