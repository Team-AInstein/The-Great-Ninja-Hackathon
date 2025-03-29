from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback
from yolov8 import detect_cars
from algo import optimize_traffic
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)


try:
    ambulance_model = YOLO("./ambulance_detection.pt")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    ambulance_model = None

def detect_ambulance(image_path):
    try:
        if ambulance_model is None:
            print("Warning: Ambulance model not loaded")
            return False
            
        # Run detection with confidence threshold
        results = ambulance_model(image_path, conf=0.3)[0]
        
        # Add more detailed logging
        print(f"Processing image for ambulance detection: {image_path}")
        
        for box in results.boxes:
            confidence = float(box.conf)
            class_id = int(box.cls)
            
            print(f"Detection found: Class {class_id} with confidence {confidence}")
            
            if confidence > 0.3:
                print(f"⚠️ Ambulance detected in {image_path} with {confidence:.2f} confidence")
                return True
                
        return False
    except Exception as e:
        print(f"Error in ambulance detection: {str(e)}")
        print(traceback.format_exc())
        return False

def process_traffic_data(image_paths):
    try:
        num_cars_list = []
        ambulance_lanes = []
        lane_map = {0: 'north', 1: 'south', 2: 'west', 3: 'east'}
        
        # First pass: detect ambulances and vehicles
        for i, image_path in enumerate(image_paths):
            try:
                # Check for ambulance first
                if detect_ambulance(image_path):
                    print(f"Ambulance detected in lane {i}")
                    ambulance_lanes.append(i)
                
                # Then count vehicles
                num_cars = detect_cars(image_path)
                num_cars_list.append(num_cars)
                
            except Exception as e:
                print(f"Error processing lane {i}: {str(e)}")
                print(traceback.format_exc())
                num_cars_list.append(0)
        
        # Get initial timings from traffic optimization
        initial_result = optimize_traffic(num_cars_list)
        
        # Convert to integer values
        green_times = {
            'north': int(initial_result['north']),
            'south': int(initial_result['south']),
            'west': int(initial_result['west']),
            'east': int(initial_result['east'])
        }
        
        print("Original timings:", green_times)
        
        # Remove all timing modifications and use original timings
        result = {
            'timings': {
                'north': green_times['north'],
                'south': green_times['south'],
                'west': green_times['west'],
                'east': green_times['east']
            },
            'ambulance_info': {
                'detected': len(ambulance_lanes) > 0,
                'lanes': ambulance_lanes
            },
            'status': 'success'
        }
        
        print("Final timings being sent:", result['timings'])
        return result
        
    except Exception as e:
        print(f"Error in process_traffic_data: {str(e)}")
        print(traceback.format_exc())
        raise

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        # Check if files were uploaded
        if 'images' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
            
        files = request.files.getlist('images')
        if len(files) != 4:
            return jsonify({'error': 'Please upload exactly 4 Images'}), 400

        # Ensure upload directory exists
        os.makedirs('uploads', exist_ok=True)

        image_paths = []
        for i, file in enumerate(files):
            if file.filename == '':
                return jsonify({'error': f'File {i+1} is empty'}), 400
                
            image_path = os.path.join('uploads', f'image_{i}.jpg')
            file.save(image_path)
            image_paths.append(image_path)

        # Process traffic data
        result = process_traffic_data(image_paths)
        return jsonify(result)

    except Exception as e:
        print(f"Error in upload_files: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Internal server error occurred', 'details': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)