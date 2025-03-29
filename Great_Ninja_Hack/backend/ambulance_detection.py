import os
import glob
import cv2 
import matplotlib.pyplot as plt
from ultralytics import YOLO


model = YOLO("./ambulance_detection.pt")


image_path = "./abmul.png"
results = model(image_path, conf=0.3, save=True) 

prediction_dirs = sorted(glob.glob("runs/detect/predict*"), key=os.path.getmtime, reverse=True)
latest_pred_dir = prediction_dirs[0] if prediction_dirs else "runs/detect/predict"


print(f"Latest YOLO output directory: {latest_pred_dir}")


detected_image = None
for file in os.listdir(latest_pred_dir):
    if file.lower().startswith(os.path.basename(image_path).split(".")[0]):  
        detected_image = os.path.join(latest_pred_dir, file)
        break


if detected_image and os.path.exists(detected_image):
    print(f"Detected image path: {detected_image}")
    
    img = cv2.imread(detected_image)
    
    if img is not None:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    else:
        print("Error: Image couldn't be loaded. Check the file format.")
else:
    print(f"Error: Processed image not found in {latest_pred_dir}")