from __future__ import annotations

from collections import OrderedDict
from math import hypot
from pathlib import Path
from typing import List

import cv2
import numpy as np
import os

from src.backend.dependencies import settings


class _CentroidTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: float = 75.0):
        self.next_object_id = 1
        self.objects: OrderedDict[int, tuple[int, int]] = OrderedDict()
        self.bboxes: OrderedDict[int, list[int]] = OrderedDict()
        self.disappeared: dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: tuple[int, int], bbox: list[int]):
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def unregister(self, object_id: int):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.bboxes:
            del self.bboxes[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects: List[list[int]]) -> List[dict]:
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.unregister(oid)
            return [
                {"bbox": list(self.bboxes[oid]), "track_id": oid}
                for oid in list(self.bboxes.keys())
            ]

        input_centroids = []
        for x1, y1, x2, y2 in rects:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids.append((cX, cY))

        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.zeros((len(object_centroids), len(input_centroids)), dtype="float")
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    D[i, j] = hypot(
                        object_centroids[i][0] - input_centroids[j][0],
                        object_centroids[i][1] - input_centroids[j][1],
                    )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            assigned_rows = set()
            assigned_cols = set()

            for row, col in zip(rows, cols):
                if row in assigned_rows or col in assigned_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = rects[col]
                self.disappeared[object_id] = 0
                assigned_rows.add(row)
                assigned_cols.add(col)

            for row in range(0, D.shape[0]):
                if row not in assigned_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.unregister(object_id)

            for col in range(0, D.shape[1]):
                if col not in assigned_cols:
                    self.register(input_centroids[col], rects[col])

        return [
            {"bbox": list(self.bboxes[oid]), "track_id": oid}
            for oid in list(self.bboxes.keys())
        ]


class PersonDetector:
    """Lightweight person detector + centroid tracker.

    Detection backends (in order):
    - OpenCV DNN with ONNX (if provided)
    - OpenCV DNN MobileNet-SSD Caffe (if provided)
    - HOG + SVM fallback
    """

    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.45):
        self.conf_threshold = float(conf_threshold)
        self.net = None
        self.backend = "None"
        self.input_size = (300, 300)
        self.mp = None
        self.mp_face_detector = None
        self.mp_face_detector_low = None
        self.mp_face_mesh = None
        self.mp_pose = None

        # OpenVINO core and compiled models (YOLOv8 for person detection, SCRFD for face detection)
        self.ov_core = None
        self.yolo_model = None
        self.scrfd_model = None
        self.yolo_input_name = None
        self.scrfd_input_name = None
        self.yolo_input_shape = None
        self.scrfd_input_shape = None

        # paths for MediaPipe tasks models (optional)
        self.face_task_path = Path(
            getattr(settings, "MP_FACE_LANDMARKER_MODEL_PATH", "")
        )
        self.pose_task_path = Path(
            getattr(settings, "MP_POSE_LANDMARKER_MODEL_PATH", "")
        )
        self.face_landmarker = None
        self.pose_landmarker = None

        onnx_path = Path(getattr(settings, "MOBILENET_SSD_ONNX", ""))
        proto = Path(getattr(settings, "MOBILENET_SSD_PROTOTXT", ""))
        caffe = Path(getattr(settings, "MOBILENET_SSD_CAFFEMODEL", ""))

        try:
            if onnx_path.exists():
                self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
                self.backend = "onnx"
            elif proto.exists() and caffe.exists():
                self.net = cv2.dnn.readNetFromCaffe(str(proto), str(caffe))
                self.backend = "caffe"
            else:
                self.net = None
                self.backend = "hog"
        except Exception as e:
            print(f"[AI] light detector init error: {e}")
            self.net = None
            self.backend = "hog"

        if self.backend == "hog":
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        else:
            self.hog = None

        # Initialize OpenVINO models if available
        try:
            from openvino.runtime import Core
            self.ov_core = Core()
            # YOLOv8 person detection model (OpenVINO IR)
            yolo_path = getattr(settings, "YOLOV8_MODEL_PATH", "")
            if yolo_path and os.path.exists(yolo_path):
                self.yolo_model = self.ov_core.compile_model(yolo_path, device_name="CPU")
                # Assume single input
                self.yolo_input_name = next(iter(self.yolo_model.inputs)).get_any_name()
                self.yolo_input_shape = self.yolo_model.inputs[0].shape
                print(f"[AI] YOLOv8 OpenVINO model loaded: {yolo_path}")
            # SCRFD face detection model (OpenVINO IR)
            scrfd_path = getattr(settings, "SCRFD_MODEL_PATH", "")
            if scrfd_path and os.path.exists(scrfd_path):
                self.scrfd_model = self.ov_core.compile_model(scrfd_path, device_name="CPU")
                self.scrfd_input_name = next(iter(self.scrfd_model.inputs)).get_any_name()
                self.scrfd_input_shape = self.scrfd_model.inputs[0].shape
                print(f"[AI] SCRFD OpenVINO model loaded: {scrfd_path}")
        except Exception as e:
            print(f"[AI] OpenVINO initialization error: {e}")

        # Initialize YuNet face detector if model path exists (fast OpenCV DNN)
        self.yunet_net = None
        yunet_path = Path(getattr(settings, "YUNET_MODEL_PATH", ""))
        if yunet_path.exists():
            try:
                self.yunet_net = cv2.dnn.readNetFromONNX(str(yunet_path))
                # YuNet outputs normalized coordinates; confidence threshold can be tuned via env if needed
                self.yunet_conf_threshold = float(
                    getattr(settings, "YUNET_CONF_THRESHOLD", 0.5)
                )
                print("[AI] YuNet face detector initialized")
            except Exception as e:
                print(f"[AI] YuNet init error: {e}")
                self.yunet_net = None

        # Try to initialize MediaPipe (tasks API preferred, legacy solutions as fallback)
        try:
            import mediapipe as mp

            self.mp = mp
            mp_conf = float(getattr(settings, "MP_MIN_DETECTION_CONFIDENCE", 0.5))
            mp_track_conf = float(getattr(settings, "MP_MIN_TRACKING_CONFIDENCE", 0.5))

            # Prefer new tasks API when model assets are available
            if self._try_init_mediapipe_tasks(mp, mp_conf, mp_track_conf):
                print(f"[AI] person_detector initialized with MediaPipe tasks backends")
            # Otherwise try the legacy mp.solutions API
            elif self._try_init_mediapipe_legacy(mp, mp_conf, mp_track_conf):
                print(
                    f"[AI] person_detector initialized with MediaPipe legacy backends (pose={self.mp_pose is not None})"
                )
            else:
                # no usable mediapipe backend
                self.mp = None
                self.mp_face_detector = None
                self.mp_face_detector_low = None
                self.mp_face_mesh = None
        except Exception as e:
            print(f"[AI] MediaPipe init failed in person_detector: {e}")
            self.mp = None
            self.mp_face_detector = None
            self.mp_face_detector_low = None
            self.mp_face_mesh = None

        self.tracker = _CentroidTracker(max_disappeared=30, max_distance=100.0)

        # Frame counter for detection skipping (to improve FPS on low‑power devices)
        self._frame_counter: int = 0
        self._skip_frames: int = getattr(settings, "DETECTION_SKIP_FRAMES", 0)

        # smoothing state for bounding boxes (EWMA)
        self.smoothed_bboxes: dict[int, list[float]] = {}
        self.smooth_alpha = float(getattr(settings, "BBOX_SMOOTH_ALPHA", 0.45) or 0.45)

        # last detection backend used (for debug overlays)
        self.last_backend: str | None = None

        # NMS threshold for overlapping bbox suppression
        self.nms_threshold = float(getattr(settings, "NMS_THRESHOLD", 0.45) or 0.45)

        # Flag to indicate whether OpenVINO models are ready for the combined stack 8
        self.use_openvino_stack = self.yolo_model is not None and self.scrfd_model is not None

        # Prepare input shape information for OpenVINO models
        if self.yolo_model is not None:
            # YOLOv8 expects input in NCHW format
            self.yolo_input_height = self.yolo_input_shape[2]
            self.yolo_input_width = self.yolo_input_shape[3]
        if self.scrfd_model is not None:
            self.scrfd_input_height = self.scrfd_input_shape[2]
            self.scrfd_input_width = self.scrfd_input_shape[3]

    # ---------------------------------------------------------------------
    # Detection helpers for the OpenVINO stack (YOLOv8 + SCRFD [+ RTMPose])
    # ---------------------------------------------------------------------
    def _preprocess_for_yolo(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalize frame for YOLOv8 OpenVINO model.

        YOLOv8 expects BGR images scaled to 0‑255, then normalized to 0‑1 and
        transposed to NCHW.
        """
        resized = cv2.resize(frame, (self.yolo_input_width, self.yolo_input_height))
        blob = resized.astype(np.float32) / 255.0
        # Add batch dimension and transpose to NCHW
        blob = np.expand_dims(blob, axis=0).transpose(0, 3, 1, 2)
        return blob

    def _compute_iou(self, a: list[int], b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
        area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
        return inter / (area_a + area_b - inter + 1e-9)

    def _nms_rects(self, rects: List[list[int]], iou_thresh: float | None = None) -> List[list[int]]:
        """Greedy NMS by area (keeps larger boxes first) to suppress overlapping detections."""
        if not rects:
            return []
        if iou_thresh is None:
            iou_thresh = float(self.nms_threshold)

        boxes = list(rects)
        areas = [max(1, (r[2] - r[0]) * (r[3] - r[1])) for r in boxes]
        order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
        keep: List[int] = []
        suppressed = set()

        for idx in order:
            if idx in suppressed:
                continue
            keep.append(idx)
            for j in order:
                if j == idx or j in suppressed:
                    continue
                iou = self._compute_iou(boxes[idx], boxes[j])
                if iou > iou_thresh:
                    suppressed.add(j)

        return [boxes[i] for i in keep]

    def _detect_with_openvino_yolo(self, frame: np.ndarray) -> List[list[int]]:
        """Run YOLOv8 person detection using the compiled OpenVINO model.

        Returns a list of bounding boxes ``[x1, y1, x2, y2]`` for detections
        whose confidence exceeds ``self.conf_threshold`` and class id == 0
        (person).
        """
        if self.yolo_model is None:
            return []
        blob = self._preprocess_for_yolo(frame)
        try:
            result = self.yolo_model.infer({self.yolo_input_name: blob})
            # The model may have a single output tensor
            output = next(iter(result.values()))
        except Exception as e:
            print(f"[AI] YOLOv8 inference error: {e}")
            return []

        # YOLOv8 output shape: (batch, N, 85) where 85 = 4 bbox + 1 obj + 80 class
        detections = []
        for det in output[0]:
            confidence = float(det[4])  # objectness score
            if confidence < self.conf_threshold:
                continue
            class_scores = det[5:]
            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])
            if class_id != 0:  # person class in COCO
                continue
            if class_conf * confidence < self.conf_threshold:
                continue
            # bbox format: cx, cy, w, h (relative to input size)
            cx, cy, w, h = det[0:4]
            # Convert to absolute coordinates in original frame size
            frame_h, frame_w = frame.shape[:2]
            x1 = int((cx - w / 2) * frame_w)
            y1 = int((cy - h / 2) * frame_h)
            x2 = int((cx + w / 2) * frame_w)
            y2 = int((cy + h / 2) * frame_h)
            # Clamp to frame bounds
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w))
            y2 = max(0, min(y2, frame_h))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append([x1, y1, x2, y2])
        return detections

    def _preprocess_for_scrfd(self, face_img: np.ndarray) -> np.ndarray:
        """Resize and normalize face crop for SCRFD OpenVINO model."""
        resized = cv2.resize(face_img, (self.scrfd_input_width, self.scrfd_input_height))
        blob = resized.astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0).transpose(0, 3, 1, 2)
        return blob

    def _detect_face_scrfd(self, frame: np.ndarray, person_bbox: list[int]) -> list[int] | None:
        """Detect a face inside the given person bounding box using SCRFD.

        Returns a face bbox ``[x1, y1, x2, y2]`` in the original frame coordinates
        or ``None`` if no face is detected.
        """
        if self.scrfd_model is None:
            return None
        x1, y1, x2, y2 = person_bbox
        # Crop the person region; ensure we have a non‑empty crop
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        blob = self._preprocess_for_scrfd(crop)
        try:
            result = self.scrfd_model.infer({self.scrfd_input_name: blob})
            output = next(iter(result.values()))
        except Exception as e:
            print(f"[AI] SCRFD inference error: {e}")
            return None
        # SCRFD output format: (1, N, 5) where 5 = [x1, y1, x2, y2, score] (relative)
        best = None
        best_score = 0.0
        for det in output[0]:
            score = float(det[4])
            if score < self.conf_threshold:
                continue
            fx1 = int(det[0] * (x2 - x1)) + x1
            fy1 = int(det[1] * (y2 - y1)) + y1
            fx2 = int(det[2] * (x2 - x1)) + x1
            fy2 = int(det[3] * (y2 - y1)) + y1
            if score > best_score:
                best_score = score
                best = [fx1, fy1, fx2, fy2]
        return best

    # Placeholder for RTMPose – full implementation would require model‑specific post‑processing.
    def _detect_pose_rtmpose(self, frame: np.ndarray, person_bbox: list[int]):
        """Run RTMPose on the person crop. Currently returns an empty list.

        The detailed keypoint extraction is out of scope for this quick integration;
        the method exists so that future extensions can plug in the proper logic.
        """
        return []

    def _detect_with_dnn(self, frame: np.ndarray) -> List[list[int]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, self.input_size, 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        rects = []

        if (
            detections.ndim == 4
            and detections.shape[2] > 0
            and detections.shape[3] >= 7
        ):
            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence < self.conf_threshold:
                    continue
                class_id = int(detections[0, 0, i, 1])
                if class_id != 15:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    continue
                rects.append([x1, y1, x2, y2])
            return rects

        try:
            out = np.squeeze(detections)
            if out.ndim == 2 and out.shape[1] >= 6:
                for det in out:
                    score = float(det[2])
                    if score < self.conf_threshold:
                        continue
                    class_id = int(det[1])
                    if class_id != 15:
                        continue
                    x1 = int(det[3] * w)
                    y1 = int(det[4] * h)
                    x2 = int(det[5] * w)
                    y2 = int(det[6] * h)
                    rects.append([x1, y1, x2, y2])
                return rects
        except Exception:
            pass

        return []

    def _detect_with_mediapipe_faces(self, frame: np.ndarray) -> List[list[int]]:
        """Detect faces with MediaPipe and expand to approximate person bboxes."""
        if self.mp_face_detector is None:
            return []

        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_detector.process(rgb)
            rects = []

            # primary detector
            if results and getattr(results, "detections", None):
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x1 = int(max(0, min((bbox.xmin) * w, w - 1)))
                    y1 = int(max(0, min((bbox.ymin) * h, h - 1)))
                    x2 = int(max(0, min((bbox.xmin + bbox.width) * w, w)))
                    y2 = int(max(0, min((bbox.ymin + bbox.height) * h, h)))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    rects.append(self._expand_face_to_person(w, h, x1, y1, x2, y2))

                return [r for r in rects if r]

            # low-conf fallback detector
            if self.mp_face_detector_low is not None:
                try:
                    results_low = self.mp_face_detector_low.process(rgb)
                    if results_low and getattr(results_low, "detections", None):
                        for det in results_low.detections:
                            bbox = det.location_data.relative_bounding_box
                            x1 = int(max(0, min((bbox.xmin) * w, w - 1)))
                            y1 = int(max(0, min((bbox.ymin) * h, h - 1)))
                            x2 = int(max(0, min((bbox.xmin + bbox.width) * w, w)))
                            y2 = int(max(0, min((bbox.ymin + bbox.height) * h, h)))
                            if x2 <= x1 or y2 <= y1:
                                continue

                            rects.append(
                                self._expand_face_to_person(w, h, x1, y1, x2, y2)
                            )

                        if rects:
                            print(
                                f"[AI] person_detector: used low-conf mediapipe detector -> {len(rects)}"
                            )
                            return [r for r in rects if r]
                except Exception:
                    pass

            # last resort: try face_mesh landmarks -> bbox
            if self.mp_face_mesh is not None:
                try:
                    mesh_res = self.mp_face_mesh.process(rgb)
                    if getattr(mesh_res, "multi_face_landmarks", None):
                        for lm in mesh_res.multi_face_landmarks:
                            xs = [p.x for p in lm.landmark]
                            ys = [p.y for p in lm.landmark]
                            minx, maxx = min(xs), max(xs)
                            miny, maxy = min(ys), max(ys)
                            fx1 = int(max(0, minx * w))
                            fy1 = int(max(0, miny * h))
                            fx2 = int(min(w, maxx * w))
                            fy2 = int(min(h, maxy * h))
                            if fx2 <= fx1 or fy2 <= fy1:
                                continue
                            rects.append(
                                self._expand_face_to_person(w, h, fx1, fy1, fx2, fy2)
                            )

                        if rects:
                            print(
                                f"[AI] person_detector: used face_mesh fallback -> {len(rects)}"
                            )
                            return [r for r in rects if r]
                except Exception:
                    pass

            return []
        except Exception:
            return []

    def _expand_face_to_person(
        self, frame_w: int, frame_h: int, x1: int, y1: int, x2: int, y2: int
    ) -> list[int] | None:
        """Expand a face bbox to approximate a person bbox with stable heuristics."""
        fw = max(1, x2 - x1)
        fh = max(1, y2 - y1)

        # Heuristics tuned for classroom: widen horizontally ~1.1x, extend downward ~2.0x
        ex_left = int(fw * 1.1)
        ex_right = int(fw * 1.1)
        ex_top = int(fh * 0.25)
        ex_bottom = int(fh * 2.0)

        px1 = max(0, x1 - ex_left)
        py1 = max(0, y1 - ex_top)
        px2 = min(frame_w, x2 + ex_right)
        py2 = min(frame_h, y2 + ex_bottom)

        if px2 <= px1 or py2 <= py1:
            return None

        return [px1, py1, px2, py2]

    def _detect_with_hog(self, frame: np.ndarray) -> List[list[int]]:
        rects, weights = self.hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        results = []
        for x, y, w, h in rects:
            results.append([int(x), int(y), int(x + w), int(y + h)])
        return results

    def _detect_with_yunet_faces(self, frame: np.ndarray) -> List[list[int]]:
        """Detect faces using YuNet ONNX model and expand to person bboxes.

        YuNet is a lightweight face detector from OpenCV model zoo. It outputs
        normalized bounding boxes (center x/y, width, height) and a confidence
        score. We filter by a configurable confidence threshold and then expand
        the face bbox to a person bbox using the existing heuristic.
        """
        if self.yunet_net is None:
            return []
        try:
            h, w = frame.shape[:2]
            # YuNet expects 320x320 input, values normalized to [0, 1]
            blob = cv2.dnn.blobFromImage(
                frame, 1.0 / 255.0, (320, 320), (0, 0, 0), swapRB=True, crop=False
            )
            self.yunet_net.setInput(blob)
            detections = self.yunet_net.forward()
            rects: List[list[int]] = []
            # Expected shape: (1, 1, N, 15)
            if detections.ndim == 4:
                for i in range(detections.shape[2]):
                    det = detections[0, 0, i]
                    # det[4] is confidence score according to YuNet spec
                    score = float(det[4])
                    if score < getattr(self, "yunet_conf_threshold", 0.5):
                        continue
                    cx, cy, width, height = (
                        float(det[0]),
                        float(det[1]),
                        float(det[2]),
                        float(det[3]),
                    )
                    # Convert normalized coordinates to absolute pixel values
                    x1 = int((cx - width / 2) * w)
                    y1 = int((cy - height / 2) * h)
                    x2 = int((cx + width / 2) * w)
                    y2 = int((cy + height / 2) * h)
                    # Clamp to frame bounds
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    expanded = self._expand_face_to_person(w, h, x1, y1, x2, y2)
                    if expanded:
                        rects.append(expanded)
            return [r for r in rects if r]
        except Exception as e:
            print(f"[AI] yunet detection error: {e}")
            return []

    # ---------------------------------------------------------------------
    # OpenVINO based detection helpers (YOLOv8 for person, SCRFD for face)
    # ---------------------------------------------------------------------
    def _detect_with_openvino_yolo(self, frame: np.ndarray) -> List[list[int]]:
        """Run YOLOv8 person detection using an OpenVINO compiled model.

        The model is expected to output detections in the standard YOLO format:
        [batch, num_boxes, 85] where the last dimension contains ``[x, y, w, h,
        confidence, class_scores...]``. We filter for class ``0`` (person) and
        apply the confidence threshold from ``self.conf_threshold``.
        """
        if self.yolo_model is None:
            return []
        try:
            h, w = frame.shape[:2]
            # Resize frame to model input size (assume NCHW)
            input_h, input_w = self.yolo_input_shape[2], self.yolo_input_shape[3]
            resized = cv2.resize(frame, (input_w, input_h))
            blob = cv2.dnn.blobFromImage(resized, 1.0 / 255.0, (input_w, input_h), swapRB=True, crop=False)
            # Perform inference using the compiled OpenVINO model
            result = self.yolo_model.infer({self.yolo_input_name: blob})
            # Get the first (and typically only) output tensor
            output = next(iter(result.values()))
            detections = np.squeeze(output)
            candidates: List[tuple[list[int], float]] = []
            if detections.ndim == 2:
                for det in detections:
                    # YOLOv8 format: [x, y, w, h, conf, class0, class1, ...]
                    conf = float(det[4])
                    if conf < self.conf_threshold:
                        continue
                    class_scores = det[5:]
                    if class_scores.size == 0:
                        class_id = 0
                        class_conf = 1.0
                    else:
                        class_id = int(np.argmax(class_scores))
                        class_conf = float(class_scores[class_id])
                    if class_id != 0:
                        continue
                    score = conf * class_conf
                    # Convert normalized coordinates back to original frame size
                    cx, cy, bw, bh = det[0:4]
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    # Clamp to frame bounds
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    candidates.append(([x1, y1, x2, y2], float(score)))

            # Apply NMS using detection scores (if available)
            rects: List[list[int]] = []
            if candidates:
                try:
                    boxes_cv = [[bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]] for bx, _ in candidates]
                    scores = [s for _, s in candidates]
                    indices = cv2.dnn.NMSBoxes(boxes_cv, scores, float(self.conf_threshold), float(self.nms_threshold))
                    # Normalize indices returned by different OpenCV versions
                    idxs = []
                    if isinstance(indices, (list, tuple, np.ndarray)):
                        for x in np.array(indices).flatten():
                            try:
                                i = int(x)
                                idxs.append(i)
                            except Exception:
                                pass
                    for i in idxs:
                        rects.append(candidates[i][0])
                except Exception:
                    # fallback to greedy NMS if OpenCV NMS fails
                    rects = self._nms_rects([c[0] for c in candidates], iou_thresh=self.nms_threshold)

            return [r for r in rects if r]
        except Exception as e:
            print(f"[AI] YOLOv8 OpenVINO detection error: {e}")
            return []

    def _detect_with_scrfd_openvino(self, frame: np.ndarray) -> List[list[int]]:
        """Run SCRFD face detection using an OpenVINO compiled model and expand to person bboxes.

        SCRFD outputs face bounding boxes in the format ``[x1, y1, x2, y2, score]``
        (absolute pixel coordinates). We filter by ``self.conf_threshold`` and then
        expand each face bbox to an approximate person bbox using the existing
        ``_expand_face_to_person`` heuristic.
        """
        if self.scrfd_model is None:
            return []
        try:
            h, w = frame.shape[:2]
            input_h, input_w = self.scrfd_input_shape[2], self.scrfd_input_shape[3]
            resized = cv2.resize(frame, (input_w, input_h))
            blob = cv2.dnn.blobFromImage(resized, 1.0 / 255.0, (input_w, input_h), swapRB=True, crop=False)
            result = self.scrfd_model.infer({self.scrfd_input_name: blob})
            # SCRFD model may have a single output tensor
            output = next(iter(result.values()))
            detections = np.squeeze(output)
            rects: List[list[int]] = []
            if detections.ndim == 2:
                for det in detections:
                    # det format: [x1, y1, x2, y2, score]
                    score = float(det[4])
                    if score < self.conf_threshold:
                        continue
                    # Coordinates are relative to the resized input; scale back
                    fx1 = int(det[0] * w / input_w)
                    fy1 = int(det[1] * h / input_h)
                    fx2 = int(det[2] * w / input_w)
                    fy2 = int(det[3] * h / input_h)
                    expanded = self._expand_face_to_person(w, h, fx1, fy1, fx2, fy2)
                    if expanded:
                        rects.append(expanded)
            return [r for r in rects if r]
        except Exception as e:
            print(f"[AI] SCRFD OpenVINO detection error: {e}")
            return []

    def detect(self, frame: np.ndarray) -> List[list[int]]:
        if frame is None or frame.size == 0:
            return []

        # Increment frame counter for skipping logic (edge devices)
        self._frame_counter += 1

        # Helper to filter detections based on heuristics (must be defined before use)
        def _filter_rects(rects: List[list[int]]) -> List[list[int]]:
            filtered = []
            for r in rects:
                if self._is_likely_person(r, frame):
                    filtered.append(r)
                else:
                    if getattr(settings, "DEBUG_DETECTION", False):
                        print(f"[AI] person_detector: filtered bbox {r}")
            return filtered

        # If OpenVINO stack is available, use it as the primary backend.
        # Apply frame skipping to reduce CPU load on weak hardware.
        if self.use_openvino_stack:
            if self._skip_frames == 0 or (self._frame_counter % (self._skip_frames + 1) == 0):
                # Run YOLOv8 person detection via OpenVINO
                yolo_rects = self._detect_with_openvino_yolo(frame)
                # Filter detections
                filtered = _filter_rects(yolo_rects)
                if filtered:
                    self.last_backend = "openvino_yolo"
                    print(f"[AI] person_detector detected {len(filtered)} rects (backend=openvino_yolo)")
                    return filtered
            # If skipping this frame, continue to other backends for robustness.

        # Prefer MediaPipe tasks-based pose detection when available
        try:
            if (
                getattr(self, "pose_landmarker", None) is not None
                and self.mp is not None
            ):
                rects = self._detect_with_mediapipe_pose_tasks(frame)
                rects = _filter_rects(rects)
                if rects:
                    self.last_backend = "mediapipe_pose_tasks"
                    print(
                        f"[AI] person_detector detected {len(rects)} rects (backend=mediapipe_pose_tasks)"
                    )
                    return rects
        except Exception as e:
            print(f"[AI] mediapipe pose tasks detector error: {e}")

        # Prefer MediaPipe legacy pose-based detection (more stable for full-body/frontal)
        if self.mp_pose is not None:
            try:
                rects = self._detect_with_mediapipe_pose(frame)
                rects = _filter_rects(rects)
                if rects:
                    self.last_backend = "mediapipe_pose"
                    print(
                        f"[AI] person_detector detected {len(rects)} rects (backend=mediapipe_pose)"
                    )
                    return rects
            except Exception as e:
                print(f"[AI] mediapipe pose detector error: {e}")

        # Prefer MediaPipe tasks-based face detections for realtime CPU performance
        try:
            if (
                getattr(self, "face_landmarker", None) is not None
                and self.mp is not None
            ):
                rects = self._detect_with_mediapipe_faces_tasks(frame)
                rects = _filter_rects(rects)
                if rects:
                    self.last_backend = "mediapipe_face_tasks"
                    print(
                        f"[AI] person_detector detected {len(rects)} rects (backend=mediapipe_face_tasks)"
                    )
                    return rects
        except Exception as e:
            print(f"[AI] mediapipe face tasks detector error: {e}")

        # Prefer YuNet face detection if available (fast and accurate)
        if self.yunet_net is not None:
            try:
                rects = self._detect_with_yunet_faces(frame)
                rects = _filter_rects(rects)
                if rects:
                    self.last_backend = "yunet"
                    print(
                        f"[AI] person_detector detected {len(rects)} rects (backend=yunet)"
                    )
                    return rects
            except Exception as e:
                print(f"[AI] yunet detection error: {e}")

        # Prefer MediaPipe face-based detections for realtime CPU performance
        if self.mp_face_detector is not None:
            try:
                rects = self._detect_with_mediapipe_faces(frame)
                rects = _filter_rects(rects)
                if rects:
                    self.last_backend = "mediapipe_face"
                    print(
                        f"[AI] person_detector detected {len(rects)} rects (backend=mediapipe_face)"
                    )
                    return rects
            except Exception as e:
                print(f"[AI] mediapipe face detector error: {e}")

        # Next try DNN (if ONNX/Caffe provided)
        if self.net is not None:
            try:
                rects = self._detect_with_dnn(frame)
                rects = _filter_rects(rects)
                if rects:
                    self.last_backend = str(self.backend)
                    print(
                        f"[AI] person_detector detected {len(rects)} rects (backend={self.backend})"
                    )
                    return rects
            except Exception as e:
                print(f"[AI] light detector DNN error: {e}")

        # Fallback to HOG detector
        if self.hog is not None:
            try:
                rects = self._detect_with_hog(frame)
                rects = _filter_rects(rects)
                if rects:
                    self.last_backend = "hog"
                    print(
                        f"[AI] person_detector detected {len(rects)} rects (backend=hog)"
                    )
                    return rects
            except Exception as e:
                print(f"[AI] HOG detector error: {e}")

        return []

    def track_people(self, frame: np.ndarray) -> List[dict]:
        rects = self.detect(frame)
        tracked = self.tracker.update(rects)

        # apply EWMA smoothing to tracker bboxes and attach debug metadata
        enhanced = []
        for item in tracked:
            tid = item.get("track_id")
            raw_bbox = item.get("bbox")
            if tid is None or raw_bbox is None:
                continue

            # convert to floats for smoothing
            prev = self.smoothed_bboxes.get(tid)
            if prev is None:
                sm = [float(c) for c in raw_bbox]
            else:
                alpha = max(0.0, min(1.0, self.smooth_alpha))
                sm = [
                    alpha * float(raw_bbox[i]) + (1.0 - alpha) * float(prev[i])
                    for i in range(4)
                ]

            # save smoothed bbox for next frame
            self.smoothed_bboxes[tid] = sm

            enhanced.append(
                {
                    "bbox": [int(sm[0]), int(sm[1]), int(sm[2]), int(sm[3])],
                    "raw_bbox": raw_bbox,
                    "track_id": tid,
                    "detector_backend": self.last_backend,
                }
            )

        return enhanced

    def _detect_with_mediapipe_pose(self, frame: np.ndarray) -> List[list[int]]:
        """Detect person bounding box using MediaPipe Pose landmarks."""
        if self.mp_pose is None:
            return []

        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.mp_pose.process(rgb)
            if not res or getattr(res, "pose_landmarks", None) is None:
                return []

            lm = res.pose_landmarks.landmark
            xs = []
            ys = []
            for p in lm:
                # landmarks are normalized
                try:
                    # filter out invisible landmarks if possible
                    vis = getattr(p, "visibility", None)
                    if vis is not None and vis < 0.2:
                        continue
                except Exception:
                    pass

                xs.append(p.x)
                ys.append(p.y)

            if not xs or not ys:
                return []

            minx = max(0.0, min(xs))
            maxx = min(1.0, max(xs))
            miny = max(0.0, min(ys))
            maxy = min(1.0, max(ys))

            if maxx <= minx or maxy <= miny:
                return []

            fx1 = int(minx * w)
            fy1 = int(miny * h)
            fx2 = int(maxx * w)
            fy2 = int(maxy * h)

            # expand to include shoulders/torso
            pad_x = int((fx2 - fx1) * 0.35)
            pad_y = int((fy2 - fy1) * 0.9)

            x1 = max(0, fx1 - pad_x)
            y1 = max(0, fy1 - pad_y)
            x2 = min(w, fx2 + pad_x)
            y2 = min(h, fy2 + int(pad_y * 0.25))

            if x2 <= x1 or y2 <= y1:
                return []

            return [[x1, y1, x2, y2]]
        except Exception as e:
            print(f"[AI] mediapipe_pose detection failed: {e}")
            return []

    def _try_init_mediapipe_tasks(self, mp, mp_conf, mp_track_conf) -> bool:
        """Try to initialize MediaPipe Tasks (FaceLandmarker / PoseLandmarker).

        Returns True if any tasks-based detector was initialized.
        """
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except Exception as error:
            print(f"[AI] mediapipe tasks import error in person_detector: {error}")
            return False

        try:
            # Face landmarker (used for face-based detections -> expand to person bbox)
            if self.face_task_path.exists():
                self.face_landmarker = vision.FaceLandmarker.create_from_options(
                    vision.FaceLandmarkerOptions(
                        base_options=BaseOptions(
                            model_asset_path=str(self.face_task_path)
                        ),
                        running_mode=vision.RunningMode.IMAGE,
                        num_faces=3,
                        min_face_detection_confidence=mp_conf,
                        min_face_presence_confidence=mp_conf,
                        min_tracking_confidence=mp_track_conf,
                    )
                )
            else:
                self.face_landmarker = None

            # Pose landmarker (optional, used for full-body detection)
            if self.pose_task_path.exists():
                self.pose_landmarker = vision.PoseLandmarker.create_from_options(
                    vision.PoseLandmarkerOptions(
                        base_options=BaseOptions(
                            model_asset_path=str(self.pose_task_path)
                        ),
                        running_mode=vision.RunningMode.IMAGE,
                        num_poses=1,
                        min_pose_detection_confidence=mp_conf,
                        min_pose_presence_confidence=mp_conf,
                        min_tracking_confidence=mp_track_conf,
                    )
                )
            else:
                self.pose_landmarker = None
        except Exception as error:
            print(f"[AI] mediapipe tasks init error in person_detector: {error}")
            self.face_landmarker = None
            self.pose_landmarker = None
            return False

        return True

    def _try_init_mediapipe_legacy(self, mp, mp_conf, mp_track_conf) -> bool:
        """Try to initialize legacy `mp.solutions` components."""
        mp_solutions = getattr(mp, "solutions", None)
        if mp_solutions is None:
            return False

        # primary detector (configured threshold)
        try:
            self.mp_face_detector = mp_solutions.face_detection.FaceDetection(
                min_detection_confidence=mp_conf
            )
        except Exception:
            self.mp_face_detector = None

        # low-confidence fallback detector
        low_conf = max(0.25, mp_conf * 0.6)
        try:
            self.mp_face_detector_low = mp_solutions.face_detection.FaceDetection(
                min_detection_confidence=low_conf
            )
        except Exception:
            self.mp_face_detector_low = None

        # face_mesh fallback
        try:
            self.mp_face_mesh = mp_solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=3,
                refine_landmarks=False,
                min_detection_confidence=mp_conf,
                min_tracking_confidence=mp_track_conf,
            )
        except Exception:
            self.mp_face_mesh = None

        # pose-based detector
        try:
            self.mp_pose = mp_solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=False,
                min_detection_confidence=mp_conf,
                min_tracking_confidence=mp_track_conf,
            )
        except Exception:
            self.mp_pose = None

        return True

    def _detect_with_mediapipe_faces_tasks(self, frame: np.ndarray) -> List[list[int]]:
        """Detect faces using MediaPipe Tasks FaceLandmarker and expand to person bboxes."""
        if self.face_landmarker is None or self.mp is None:
            return []

        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.face_landmarker.detect(image)
            rects = []
            if getattr(res, "face_landmarks", None):
                for lm in res.face_landmarks:
                    try:
                        pts = getattr(lm, "landmark", lm)
                        xs = [p.x for p in pts]
                        ys = [p.y for p in pts]
                    except Exception:
                        continue

                    minx, maxx = min(xs), max(xs)
                    miny, maxy = min(ys), max(ys)
                    fx1 = int(max(0, minx * w))
                    fy1 = int(max(0, miny * h))
                    fx2 = int(min(w, maxx * w))
                    fy2 = int(min(h, maxy * h))
                    if fx2 <= fx1 or fy2 <= fy1:
                        continue
                    rects.append(self._expand_face_to_person(w, h, fx1, fy1, fx2, fy2))

            return [r for r in rects if r]
        except Exception:
            return []

    def _detect_with_mediapipe_pose_tasks(self, frame: np.ndarray) -> List[list[int]]:
        """Detect person bbox using MediaPipe Tasks PoseLandmarker."""
        if self.pose_landmarker is None or self.mp is None:
            return []

        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.pose_landmarker.detect(image)
            if not res or getattr(res, "pose_landmarks", None) is None:
                return []

            rects = []
            for pl in res.pose_landmarks:
                try:
                    pts = getattr(pl, "landmark", pl)
                    xs = []
                    ys = []
                    for p in pts:
                        try:
                            vis = getattr(p, "visibility", None)
                            if vis is not None and vis < 0.2:
                                continue
                        except Exception:
                            pass
                        xs.append(p.x)
                        ys.append(p.y)
                except Exception:
                    continue

                if not xs or not ys:
                    continue

                minx = max(0.0, min(xs))
                maxx = min(1.0, max(xs))
                miny = max(0.0, min(ys))
                maxy = min(1.0, max(ys))

                if maxx <= minx or maxy <= miny:
                    continue

                fx1 = int(minx * w)
                fy1 = int(miny * h)
                fx2 = int(maxx * w)
                fy2 = int(maxy * h)

                pad_x = int((fx2 - fx1) * 0.35)
                pad_y = int((fy2 - fy1) * 0.9)

                x1 = max(0, fx1 - pad_x)
                y1 = max(0, fy1 - pad_y)
                x2 = min(w, fx2 + pad_x)
                y2 = min(h, fy2 + int(pad_y * 0.25))

                if x2 <= x1 or y2 <= y1:
                    continue

                rects.append([x1, y1, x2, y2])

            return rects
        except Exception as e:
            print(f"[AI] mediapipe_pose tasks detection failed: {e}")
            return []

    def _is_likely_person(self, bbox: list[int], frame: np.ndarray) -> bool:
        """Heuristic filter to remove small/odd detections (hands, objects).

        Uses relative thresholds from settings so values can be tuned via env.
        """
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)

            # relative checks
            min_h_ratio = float(getattr(settings, "MIN_PERSON_HEIGHT_RATIO", 0.12))
            if height < int(h * min_h_ratio):
                return False

            area_ratio = float(width * height) / float(max(1, w * h))
            if area_ratio < float(getattr(settings, "MIN_PERSON_AREA_RATIO", 0.003)):
                return False

            ar = float(width) / float(height)
            if ar < float(
                getattr(settings, "MIN_PERSON_ASPECT_RATIO", 0.25)
            ) or ar > float(getattr(settings, "MAX_PERSON_ASPECT_RATIO", 2.5)):
                return False

            return True
        except Exception:
            return False
