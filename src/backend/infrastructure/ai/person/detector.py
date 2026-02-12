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
        try:
            if results and len(results) > 0 and getattr(results[0].boxes, 'id', None) is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                for box, track_id in zip(boxes, ids):
                    bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                    output.append({"bbox": bbox, "track_id": int(track_id)})
        except Exception:
            pass
        return output
