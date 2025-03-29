import cv2 as cv
import time
from ultralytics import YOLO

Conf_threshold = 0.6
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use other variants like 'yolov8s.pt'

cap = cv.VideoCapture('pexels-alex-pelsh-6896028.mp4')
frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
dim = (int(frame_width/4), int(frame_height/4))
print(dim)
out = cv.VideoWriter('OutputVideo3.avi', fourcc, 30.0, dim)
starting_time = time.time()
frame_counter = 0

while True:
    ret, frame = cap.read()
    frame_counter += 1
    if not ret:
        break
    
    frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    results = model(frame)[0]
    
    for result in results.boxes:
        classid = int(result.cls.item())
        score = result.conf.item()
        box = result.xyxy[0].cpu().numpy().astype(int)
        
        color = COLORS[classid % len(COLORS)]
        label = "%s : %f" % (class_name[classid], score)
        cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 1)
        cv.rectangle(frame, (box[0]-2, box[1]-20), (box[0]+120, box[1]-4), (100, 130, 100), -1)
        cv.putText(frame, label, (box[0], box[1]-10),
                   cv.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
    
    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime
    cv.line(frame, (18, 43), (140, 43), (0, 0, 0), 27)
    cv.putText(frame, f'FPS: {round(fps,2)}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
    
    cv.imshow('frame', frame)
    out.write(frame)
    
    if cv.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv.destroyAllWindows()
print('done')
