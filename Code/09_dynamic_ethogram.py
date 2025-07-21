# Adjust the file paths to match your project directory location.
## Adjust the names of behaviours that are relevant for your project. This will need to be done for the "ALL_CLASSES" argument below. 
## Specify the trained model you wish to deploy using the "model" argument below.
## Specify the input video path you would like to run analysis on in the "video_path" argument below.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ultralytics import YOLO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

matplotlib.use('Agg')

video_path = r"C:\Users\lchen\Desktop\PantherAI - Final Scripts\Vids4Testing\more for testing\c31_a_11_4_2024 10_18_51 AM (UTC-05_00).mkv"
output_video_path = "ethogram_output.mp4"

model = YOLO(r"C:\Users\lchen\Desktop\PantherAI - Final Scripts\best_1280.pt") #or use best_1280_march_v2.pt"

ALL_CLASSES = ["T1_REST", "T1_LOCO", "T1_FEED", "T1_OBMAN", "T1_STEREO", "No Detections"]
color_map = {
    "T1_REST": "gold",
    "T1_LOCO": "mediumspringgreen",
    "T1_FEED": "red", 
    "T1_OBMAN": "fuchsia",
    "T1_STEREO": "dodgerblue",
    "No Detections": "black"
}

frame_behaviors = []
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit()

# Standard size (HD resolution)
output_width = 1280
output_height = 720
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (output_width, output_height))

def generate_ethogram_image(behaviors, max_frames=100):
    fig, ax = plt.subplots(figsize=(12, 6))
    canvas = FigureCanvas(fig)

    spacing_factor = 0.2
    behavior_positions = {behavior: idx * spacing_factor for idx, behavior in enumerate(ALL_CLASSES)}

    start_idx = max(0, len(behaviors) - max_frames)
    x_vals, y_vals, colors = [], [], []

    for i, behavior_set in enumerate(behaviors[start_idx:], start=start_idx):
        for behavior in behavior_set:
            x_vals.append(i)
            y_vals.append(behavior_positions[behavior])
            
            # Force color for T1_FEED to be 'orangered'
            if behavior == "T1_FEED":
                colors.append("red")  # Use 'orangered' explicitly for T1_FEED
            else:
                colors.append(color_map[behavior])  # Use mapped color for other behaviors

    ax.scatter(x_vals, y_vals, c=colors, s=50, marker='s')
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Behaviors")
    ax.set_title("Real-Time Ethogram")
    ax.set_yticks(list(behavior_positions.values()))
    ax.set_yticklabels(list(behavior_positions.keys()))
    ax.set_xlim(start_idx, len(behaviors) + 10)
    ax.set_ylim(-0.1, max(behavior_positions.values()) + 0.1)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    canvas.draw()
    ethogram_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    ethogram_image = ethogram_image.reshape(canvas.get_width_height()[::-1] + (3,))
    return ethogram_image

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop if no more frames

    results = model(frame, conf=0.6)
    detected_behaviors = set()  # Store multiple behaviors

    for result in results:
        for detection in result.boxes:
            cls = int(detection.cls[0].cpu().numpy())
            detected_behaviors.add(model.names[cls].strip())  # Ensures label matches color map key
           # print(f"Detected behavior: {model.names[cls].strip()}") #if yo uwant the detected behavior printed; although it is already printed by default...


    if not detected_behaviors:
        detected_behaviors.add("No Detections")  # If nothing detected, add default

    frame_behaviors.append(list(detected_behaviors))  # Store all detected behaviors for the frame
    ethogram_image = generate_ethogram_image(frame_behaviors)

    # Resize ethogram to fit standard output size while keeping aspect ratio
    ethogram_resized = cv2.resize(ethogram_image, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
    
    out.write(ethogram_resized)  # Write ethogram-only frame

    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print(f"Video saved as {output_video_path}")
