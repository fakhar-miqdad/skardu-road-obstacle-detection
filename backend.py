
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI(title="Skardu Road Obstacle Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("runs/detect/train/weights/best.pt")

TARGET_CLASSES = [0, 1, 2, 3, 5, 7]


@app.get("/")
def home():
    return {"message": "Skardu Road AI Backend is Running"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}

    results = model.predict(
        image,
        classes=TARGET_CLASSES,
        conf=0.25
    )

    detections = []

    for result in results:
        for box in result.boxes:
            detections.append({
                "object": model.names[int(box.cls[0])],
                "confidence": round(
                    float(box.conf[0]) * 100, 2
                )
            })

    # Draw bounding boxes
    annotated_image = results[0].plot()

    # Convert image into Base64
    success, buffer = cv2.imencode(
        ".jpg", annotated_image
    )

    image_base64 = base64.b64encode(
        buffer
    ).decode("utf-8")

    return {
        "total_objects": len(detections),
        "detections": detections,
        "annotated_image": image_base64
    }
