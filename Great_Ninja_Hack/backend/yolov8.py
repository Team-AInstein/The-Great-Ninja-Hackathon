import cv2 as cv
import time
from collections import deque
import numpy as np
from scipy.signal import find_peaks
from ultralytics import YOLO

def detect_cars(video_file):
    # Set thresholds
    Conf_threshold = 0.4
    NMS_threshold = 0.4

    # Define colors for different classes
    COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), 
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    # Load class names from file
    class_name = []
    with open('classes.txt', 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]

    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt', 'yolov8m.pt', etc.

    # Open the video file
    cap = cv.VideoCapture(video_file)
    starting_time = time.time()
    frame_counter = 0

    # Create a named window and set it to full screen
    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.setWindowProperty('frame', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    # To keep track of car counts and timestamps
    car_counts = deque()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # Perform detection
        results = model(frame)[0]
        car_count = 0
        
        for result in results.boxes:
            classid = int(result.cls.item())
            score = result.conf.item()
            box = result.xyxy[0].cpu().numpy().astype(int)
            
            if class_name[classid] == "car":
                car_count += 1
                color = COLORS[classid % len(COLORS)]
                label = f"{class_name[classid]} : {score:.2f}"
                cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv.putText(frame, label, (box[0], box[1]-10), 
                           cv.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

        # Record the car count with the current timestamp
        current_time = time.time()
        car_counts.append((current_time, car_count))

        # Remove counts that are older than 30 seconds
        while car_counts and car_counts[0][0] < current_time - 30:
            car_counts.popleft()

        # Extract car count values
        car_count_values = [count for _, count in car_counts]
        peaks, _ = find_peaks(car_count_values)

        mean_peak_value = np.mean([car_count_values[i] for i in peaks]) if peaks.size > 0 else 0

        # Calculate and display FPS
        ending_time = time.time()
        fps = frame_counter / (ending_time - starting_time)
        cv.putText(frame, f'FPS: {fps:.2f}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(frame, f'Mean Peak Cars : {mean_peak_value:.2f}', (20, 80), 
                   cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)

        # Display the frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    return mean_peak_value
