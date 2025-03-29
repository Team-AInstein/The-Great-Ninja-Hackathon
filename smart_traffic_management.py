import cv2 as cv
import numpy as np
from ultralytics import YOLO
from collections import deque
import pytesseract

# Import the genetic algorithm traffic optimization module
from traffic_optimization import optimize_traffic

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
        if classid == 2:  
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

def process_intersection(images):
    """
    Processes 4 images (one for each side of the intersection) and returns signal times.
    """
    vehicle_counts = []
    emergency_flags = []

    for img in images:
        vehicle_counts.append(detect_cars(img))
        emergency_flags.append(detect_emergency_vehicle(img))

    
    optimized_times = optimize_traffic(vehicle_counts)
    return optimized_times

if __name__ == "__main__":
   
    image_paths = ["1mg.jpg", "2mg.jpg", "3mg.jpg", "4mg.jpg"]
    images = [cv.imread(path) for path in image_paths]

    
    if any(img is None for img in images):
        print("Error: One or more images could not be loaded.")
    else:
        optimized_signal_times = process_intersection(images)
        print("Optimized Traffic Signal Timings:")
        for direction, time in optimized_signal_times.items():
            print(f"{direction.capitalize()} should have {time} seconds of green light.")
