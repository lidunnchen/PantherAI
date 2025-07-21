# Specify the path to the video input or folder of videos you would like to analyze by adjusting the "video_folder" argument below.
## Adjust the behavioural categories depending on your study needs, do this for anywhere below you see "ALL_CLASSES" or "color_map" or "class_labels".
### Adjust the colours associated with each behaviour to your preference.
#### A activity budget barplot will be produced as well as a .csv file indicating the number of frames and time spent engaged in each respective behavioural class. 
##### Refer to the associated article (Chen et al., 2025) and Supplemental Table 1 for descriptions of each script's intended usage and parameters. 


import cv2
import csv
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict
import time
import numpy as np

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Define all behavior categories
ALL_CLASSES = ["T1_REST", "T1_LOCO", "T1_FEED", "T1_OBMAN", "T1_STEREO", "No Detections"]

def yolo_video_class_time(model_path, video_path, output_csv):
    start_time = time.time()
    model = YOLO(model_path)
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps

    class_times = defaultdict(float)
    class_frames = defaultdict(int)
    no_detection_frames = 0
    total_frames = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model.predict(frame, stream=True)
        detections_made = False

        for result in results:
            for box in result.boxes.data.tolist():
                detections_made = True
                class_id = int(box[5])
                class_name = model.names[class_id]
                class_times[class_name] += frame_duration
                class_frames[class_name] += 1

        if not detections_made:
            no_detection_frames += 1

        total_frames += 1

    video.release()
    total_time_seconds = total_frames * frame_duration
    total_time_minutes = total_time_seconds / 60

    # Ensure all categories are included in percent_budget
    percent_budget = {
        class_name: (class_times[class_name] / 60) / total_time_minutes * 100 if class_times[class_name] > 0 else 0.0
        for class_name in ALL_CLASSES
    }

    no_detection_time_minutes = (no_detection_frames * frame_duration) / 60
    percent_budget["No Detections"] = (no_detection_time_minutes / total_time_minutes) * 100

    end_time = time.time()
    inference_time = end_time - start_time

    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Class', 'Time (minutes)', 'Percent budget (%)', 'Number of Frames'])
        writer.writerow(['Total', f"{total_time_minutes:.2f}", "100.00", total_frames])
        for class_name in ALL_CLASSES:
            writer.writerow([class_name, f"{class_times[class_name]:.2f}", f"{percent_budget[class_name]:.2f}", class_frames[class_name]])
        writer.writerow(['Inference Time (seconds)', f"{inference_time:.2f}", "", ""])

    print(f"Results saved to {output_csv}")
    return class_times, percent_budget

# Example usage; CHANGE TO REFLECT THE APPROPRIATE DIRECTORY FOR YOUR PROJECT

video_folder = r"C:\Users\lchen\Desktop\PantherAI - Final Scripts\Vids4Testing\11_4_2024 10_18_51 AM (UTC-05_00).mkv"
class_times, percent_budget = yolo_video_class_time('best_1280.pt', video_folder, 'output.csv')

# Define color mapping
color_map = {
    "T1_REST": "gold",
    "T1_LOCO": "mediumspringgreen",
    "T1_FEED": "red",
    "T1_OBMAN": "fuchsia",
    "T1_STEREO": "dodgerblue",
    "No Detections": "black"
} #colors for each class WERE gold, mediumspringgreen, red, fuchsia, dogerblue, black

# Relabel behavior classes (wrapped for better display)
class_labels = {
    "T1_REST": "Rest",
    "T1_FEED": "Feed",
    "T1_OBMAN": "Object\nManipulation",
    "T1_LOCO": "Locomotion",
    "T1_STEREO": "Pace",
    "No Detections": "No Detections"
}

# Plot horizontal bar chart with all categories
classes = ALL_CLASSES
percentages = [percent_budget[c] for c in ALL_CLASSES]
colors = [color_map[c] for c in ALL_CLASSES]
labels = [class_labels[c] for c in ALL_CLASSES]

plt.figure(figsize=(11, 6))  # Consistent figure size

y_pos = np.arange(len(classes))
plt.barh(y_pos, percentages, color=colors, edgecolor='black', height=0.7)

# Font size adjustments
font_size = 16
percent_label_size = 14

plt.xlabel("Percentage of Time (%)", fontsize=font_size)
plt.ylabel("Behavior", fontsize=font_size)
plt.title("Percentage of Time Spent in Each Behavior", fontsize=font_size + 1)
plt.yticks(y_pos, labels, fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.xlim(0, 100)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add percentage text labels
for i, percentage in enumerate(percentages):
    plt.text(percentage + 1, y_pos[i], f"{percentage:.1f}%", va='center', fontsize=percent_label_size)

plt.tight_layout()
plt.savefig("activity_budget_barplot.png", dpi=600)
