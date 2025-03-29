import cv2 as cv
import numpy as np
from ultralytics import YOLO
from collections import deque
from scipy.signal import find_peaks
import pytesseract

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Use appropriate YOLO model

def detect_cars(image):
    """
    Detects cars in an image using YOLOv8 and returns the car count.
    """
    results = model(image)[0]
    car_count = 0

    for result in results.boxes:
        classid = int(result.cls.item())
        if classid == 2:  # YOLO class ID for cars (ensure correct ID)
            car_count += 1

    return car_count

def detect_emergency_vehicle(image):
    """
    Uses OCR to detect emergency vehicle text (e.g., 'AMBULANCE', 'FIRE').
    Returns True if an emergency vehicle is detected, otherwise False.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    
    emergency_keywords = ["AMBULANCE", "FIRE", "POLICE"]
    for keyword in emergency_keywords:
        if keyword in text.upper():
            return True
    return False

def calculate_signal_times(vehicle_counts, emergency_flags):
    """
    Determines the optimal signal duration for each side based on traffic density.
    Emergency vehicles get priority.
    """
    base_time = 10  # Minimum green light duration in seconds

    total_cars = sum(vehicle_counts)
    if total_cars == 0:
        return [base_time] * 4  # Default if no cars are detected

    # Calculate proportional times
    signal_times = [(count / total_cars) * 60 for count in vehicle_counts]

    # Ensure no signal is less than the base_time
    signal_times = [max(base_time, t) for t in signal_times]

    # If an emergency vehicle is detected, prioritize that side
    for i in range(4):
        if emergency_flags[i]:
            signal_times[i] = max(signal_times[i], 30)  # Give emergency lane priority

    return signal_times

def process_intersection(images):
    """
    Processes 4 images (one for each side of the intersection) and returns signal times.
    """
    vehicle_counts = []
    emergency_flags = []

    for img in images:
        vehicle_counts.append(detect_cars(img))
        emergency_flags.append(detect_emergency_vehicle(img))

    signal_times = calculate_signal_times(vehicle_counts, emergency_flags)

    return signal_times

if __name__ == "__main__":
    # Load four images
    image_paths = ["side1.jpg", "side2.jpg", "side3.jpg", "side4.jpg"]
    images = [cv.imread(path) for path in image_paths]

    # Ensure all images are loaded correctly
    if any(img is None for img in images):
        print("Error: One or more images could not be loaded.")
    else:
        signal_times = process_intersection(images)
        for i, time in enumerate(signal_times):
            print(f"Side {i+1} should have {time:.2f} seconds of green light.")

 