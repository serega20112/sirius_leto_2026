import cv2
import mediapipe as mp
import numpy as np


class PoseEstimator:
    """
    Анализ вовлеченности на основе направления взгляда (поворота головы).
    Использует MediaPipe FaceMesh.
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )

    def estimate_engagement(self, frame: np.ndarray, bbox: list) -> str:
        """
        Определяет статус вовлеченности: 'high', 'medium', 'low', 'unknown'.

        Args:
            frame: Полный кадр видео (numpy array BGR).
            bbox: Координаты лица/человека [x1, y1, x2, y2].
        """
        x1, y1, x2, y2 = map(int, bbox)

        h_img, w_img, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return "unknown"

        face_roi = frame[y1:y2, x1:x2]

        results = self.face_mesh.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return "unknown"

        landmarks = results.multi_face_landmarks[0]

        img_h, img_w, _ = face_roi.shape

        face_2d = []
        face_3d = []

        key_points = [33, 263, 1, 61, 291, 199]

        for idx, lm in enumerate(landmarks.landmark):
            if idx in key_points:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1]
        ])

        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        if not success:
            return "unknown"

        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Q, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)


        pitch = angles[0] * 360
        yaw = angles[1] * 360


        is_looking_away = abs(yaw) > 35
        is_looking_down = pitch > 20

        if is_looking_away or is_looking_down:
            return "low"

        if abs(yaw) > 15 or pitch > 10:
            return "medium"

        return "high"