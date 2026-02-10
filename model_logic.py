import cv2
from ultralytics import YOLO

class FruitAI:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')

    def process_frame(self, frame):
        results = self.model(frame, stream=True)
        for r in results:
            return r.plot()
        return frame

    def predict_image(self, path):
        results = self.model(path)
        res = results[0]
        grade = "Grade A (Excellent)"
        detections = []
        for box in res.boxes:
            label = res.names[int(box.cls[0])]
            conf = float(box.conf[0])
            detections.append({'label': label, 'conf': f"{conf:.2f}"})
            if any(x in label.lower() for x in ["rotten", "defect", "bad"]):
                grade = "Grade C (Rejected)"
            elif "bruise" in label.lower() and "C" not in grade:
                grade = "Grade B (Standard)"
        return detections, grade, res.plot()