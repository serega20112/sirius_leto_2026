import torch
from ultralytics import YOLO
import cv2
import numpy as np


class PoseEstimator:
    def __init__(self):
        self.model = YOLO("yolov8n-pose.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # 3D точки (без изменений)
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-450.0, 0.0, -400.0),
                (450.0, 0.0, -400.0),
                (-700.0, -600.0, -450.0),
                (700.0, -600.0, -450.0),
            ],
            dtype=np.float64,
        )

    def estimate_engagement(self, frame, bbox):
        if bbox is None:
            return "unknown"
        x1, y1, x2, y2 = map(int, bbox)
        face_roi = frame[max(0, y1) : y2, max(0, x1) : x2]
        if face_roi.size == 0:
            return "unknown"

        # Указываем девайс GPU
        results = self.model(face_roi, verbose=False, conf=0.5, device=self.device)[0]

        if results.keypoints is None or len(results.keypoints.xy) == 0:
            return "unknown"

        kp = results.keypoints.xy[0].cpu().numpy()
        if len(kp) < 7 or np.any(kp[:5] == 0):
            return "unknown"

        image_points = np.array(
            [
                (kp[0][0] + x1, kp[0][1] + y1),
                (kp[1][0] + x1, kp[1][1] + y1),
                (kp[2][0] + x1, kp[2][1] + y1),
                (kp[3][0] + x1, kp[3][1] + y1),
                (kp[4][0] + x1, kp[4][1] + y1),
                (kp[5][0] + x1, kp[5][1] + y1),
                (kp[6][0] + x1, kp[6][1] + y1),
            ],
            dtype=np.float64,
        )

        focal_length = frame.shape[1]
        cam_matrix = np.array(
            [
                [focal_length, 0, frame.shape[1] / 2],
                [0, focal_length, frame.shape[0] / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        success, rot_vec, trans_vec = cv2.solvePnP(
            self.model_points, image_points, cam_matrix, np.zeros((4, 1))
        )
        if not success:
            return "unknown"

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw = angles[0], angles[1]

        if abs(yaw) > 35 or pitch > 20:
            return "low"
        elif abs(yaw) > 15 or pitch > 10:
            return "medium"
        return "high"
