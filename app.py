from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO("yolov8n.pt")

def calculate_distance(box, frame_width):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    obj_width = x2 - x1
    distance = round(1000 / (obj_width + 1), 2)
    return distance

def get_position(frame_width, x1):
    if x1 < frame_width // 3:
        return "left"
    elif x1 < 2 * frame_width // 3:
        return "center"
    else:
        return "right"

@app.get("/")
def home():
    return {"status": "backend running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(frame)[0]

    nearest = None
    min_dist = 9999

    for box in results.boxes:
        label = results.names[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        dist = calculate_distance(box, frame.shape[1])

        if dist < min_dist:
            min_dist = dist
            nearest = (label, dist, x1)

    if nearest:
        pos = get_position(frame.shape[1], nearest[2])
        return {
            "object": nearest[0],
            "distance": nearest[1],
            "position": pos
        }
    else:
        return {"message": "No object detected"}
