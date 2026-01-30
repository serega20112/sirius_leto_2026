import torch
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def track_people(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.7,
            verbose=False,
            device=self.device,
            tracker="bytetrack.yaml",
        )

        output = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for box, track_id in zip(boxes, ids):
                output.append({"bbox": box, "track_id": track_id})
        return output
