#!/usr/bin/env python3

import argparse
import asyncio
import json
import math
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from aiohttp import WSMsgType, web

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:
    mp = None
    mp_python = None
    mp_vision = None


COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

FER_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def clamp(value, low, high):
    return max(low, min(value, high))


def softmax(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    if np.all(values >= 0) and 0.9 <= float(values.sum()) <= 1.1:
        return values / max(float(values.sum()), 1e-6)
    exps = np.exp(values - float(values.max()))
    return exps / max(float(exps.sum()), 1e-6)


class YoloDetector:
    def __init__(self, model_path, labels, conf=0.35, nms=0.45, size=640):
        self.labels = labels
        self.conf = conf
        self.nms = nms
        self.size = size
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.size / w, self.size / h)
        rw, rh = int(round(w * scale)), int(round(h * scale))
        dx, dy = (self.size - rw) // 2, (self.size - rh) // 2
        resized = cv2.resize(frame, (rw, rh))
        padded = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        padded[dy:dy + rh, dx:dx + rw] = resized

        blob = cv2.dnn.blobFromImage(padded, 1.0 / 255.0, (self.size, self.size), swapRB=True)
        self.net.setInput(blob)
        output = self.net.forward()
        rows = output[0]
        if rows.shape[0] < rows.shape[1]:
            rows = rows.T

        boxes, scores, class_ids = [], [], []
        for row in rows:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < self.conf:
                continue
            cx, cy, bw, bh = [float(x) for x in row[:4]]
            x1 = (cx - bw / 2 - dx) / scale
            y1 = (cy - bh / 2 - dy) / scale
            x2 = (cx + bw / 2 - dx) / scale
            y2 = (cy + bh / 2 - dy) / scale
            left = int(clamp(round(x1), 0, w - 1))
            top = int(clamp(round(y1), 0, h - 1))
            right = int(clamp(round(x2), 0, w - 1))
            bottom = int(clamp(round(y2), 0, h - 1))
            boxes.append([left, top, max(1, right - left), max(1, bottom - top)])
            scores.append(score)
            class_ids.append(class_id)

        keep = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.nms)
        detections = []
        for index in np.array(keep).reshape(-1) if len(keep) else []:
            class_id = class_ids[int(index)]
            detections.append(
                {
                    "box": boxes[int(index)],
                    "score": scores[int(index)],
                    "label": self.labels[class_id] if class_id < len(self.labels) else "unknown",
                    "class_id": class_id,
                }
            )
        return detections


class HandGestureRecognizer:
    def __init__(self, model_path):
        self.ready = False
        self.detector = None
        self.history = deque(maxlen=48)
        self.last_active = None
        self.last_score = 0.0
        self.last_seen = 0.0
        self.last_active_at = 0.0
        if mp is None or mp_python is None or mp_vision is None or not Path(model_path).exists():
            return

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)
        self.ready = True

    def detect(self, frame):
        if not self.ready:
            return {"hands": [], "events": []}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(image)
        hands = []
        for index, landmarks in enumerate(result.hand_landmarks):
            handedness = "unknown"
            if index < len(result.handedness) and result.handedness[index]:
                handedness = result.handedness[index][0].category_name
            hand = self._hand_features(landmarks, frame.shape[1], frame.shape[0], handedness)
            hands.append(hand)

        if not hands:
            self.history.clear()
            if time.time() - self.last_seen > 0.5:
                self.last_active = None
            return {"hands": [], "events": []}

        primary = max(hands, key=lambda hand: hand["area"])
        now = time.time()
        self.last_seen = now
        self.history.append(
            {
                "t": now,
                "x": primary["center"][0],
                "y": primary["center"][1],
                "area": primary["area"],
                "open": primary["open_ratio"],
                "open_count": primary["open_count"],
                "tip_z": primary["tip_z"],
            }
        )
        gesture, score = self._classify()
        events = []
        if gesture:
            self.last_active = gesture
            self.last_score = score
            self.last_active_at = now
            events.append({"gesture": gesture, "score": score, "handedness": primary["handedness"]})
        elif self.last_active and now - self.last_active_at < 0.45:
            events.append({"gesture": self.last_active, "score": self.last_score, "handedness": primary["handedness"]})
        return {"hands": hands, "events": events}

    def _hand_features(self, landmarks, width, height, handedness):
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        px_points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
        min_xy = points[:, :2].min(axis=0)
        max_xy = points[:, :2].max(axis=0)
        bbox_wh = np.maximum(max_xy - min_xy, 1e-4)
        area = float(bbox_wh[0] * bbox_wh[1])
        wrist = points[0, :2]
        mcp = points[[5, 9, 13, 17], :2]
        pips = points[[3, 6, 10, 14, 18], :2]
        tips = points[[4, 8, 12, 16, 20], :2]
        mcp_dist = np.linalg.norm(mcp - wrist, axis=1).mean()
        tip_dist = np.linalg.norm(tips - wrist, axis=1).mean()
        open_ratio = float(tip_dist / max(mcp_dist, 1e-4))
        finger_tip_dist = np.linalg.norm(tips - wrist, axis=1)
        finger_pip_dist = np.linalg.norm(pips - wrist, axis=1)
        open_count = int(np.sum(finger_tip_dist > (finger_pip_dist * 1.04)))
        return {
            "points": px_points,
            "norm_points": points,
            "bbox": [int(min_xy[0] * width), int(min_xy[1] * height), int(bbox_wh[0] * width), int(bbox_wh[1] * height)],
            "center": (float(points[:, 0].mean()), float(points[:, 1].mean())),
            "area": area,
            "open_ratio": open_ratio,
            "open_count": open_count,
            "tip_z": float(points[[4, 8, 12, 16, 20], 2].mean()),
            "handedness": handedness,
        }

    def _recent(self, seconds):
        now = time.time()
        return [item for item in self.history if now - item["t"] <= seconds]

    def _classify(self):
        samples = self._recent(2.0)
        if len(samples) < 5:
            return None, 0.0

        xs = np.array([item["x"] for item in samples], dtype=np.float32)
        ys = np.array([item["y"] for item in samples], dtype=np.float32)
        areas = np.array([item["area"] for item in samples], dtype=np.float32)
        opens = np.array([item["open"] for item in samples], dtype=np.float32)
        open_counts = np.array([item["open_count"] for item in samples], dtype=np.float32)
        tip_z = np.array([item["tip_z"] for item in samples], dtype=np.float32)

        x_range = float(xs.max() - xs.min())
        y_range = float(ys.max() - ys.min())
        open_range = float(opens.max() - opens.min())
        z_range = float(tip_z.max() - tip_z.min())
        area_range = float(areas.max() - areas.min())
        area_start = float(np.median(areas[: max(2, len(areas) // 4)]))
        area_end = float(np.median(areas[-max(2, len(areas) // 4):]))
        area_growth = area_end / max(area_start, 1e-4)
        recent_area_std = float(np.std(areas[-5:]))
        recent_xy_motion = float(
            np.std(xs[-min(5, len(xs)):]) + np.std(ys[-min(5, len(ys)):])
        )
        median_open = float(np.median(opens))
        median_open_count = float(np.median(open_counts))

        wave_turns = self._turn_count(xs, min_delta=0.010)
        curl_turns = self._turn_count(opens, min_delta=0.025)
        z_turns = self._turn_count(tip_z, min_delta=0.018)

        if x_range > 0.045 and wave_turns >= 1 and y_range < 0.32 and median_open_count >= 3:
            score = clamp((x_range / 0.14) + (0.10 * wave_turns), 0.35, 1.0)
            return "hi", score

        if (
            area_growth > 1.10
            and recent_area_std < 0.012
            and recent_xy_motion < 0.035
            and median_open_count >= 3
            and median_open > 1.12
        ):
            score = clamp((area_growth - 1.0) / 0.35, 0.35, 1.0)
            return "reject", score

        if (
            (open_range > 0.10 or z_range > 0.025 or area_range > 0.035)
            and (curl_turns >= 1 or z_turns >= 1 or open_range > 0.16)
            and x_range < 0.24
            and y_range < 0.32
        ):
            score = clamp((open_range / 0.28) + (z_range / 0.08) + (0.08 * curl_turns), 0.35, 1.0)
            return "invite", score

        return None, 0.0

    @staticmethod
    def _turn_count(values, min_delta):
        if len(values) < 4:
            return 0
        diffs = np.diff(values)
        signs = []
        for diff in diffs:
            if abs(float(diff)) < min_delta:
                continue
            signs.append(1 if diff > 0 else -1)
        turns = 0
        for prev, curr in zip(signs, signs[1:]):
            if prev != curr:
                turns += 1
        return turns


class LocalVisionRuntime:
    def __init__(
        self,
        root,
        camera_index=0,
        emotion_backend="pyfeat",
        stream_fps=20.0,
        inference_fps=5.0,
        jpeg_quality=76,
        pyfeat_interval=1.2,
        camera_width=960,
        camera_height=540,
    ):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.running = False
        self.camera_index = camera_index
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.websockets = set()
        self.loop = None
        self.latest_state = None
        self.latest_jpeg = None
        self.latest_frame = None
        self.latest_detections = []
        self.latest_faces = []
        self.latest_hands = []
        self.image_seq = 0
        self.available_cameras = self.scan_cameras()
        self.emotion_alpha = 0.25
        self.emotion_ema = 0.0
        self.emotion_ema_ready = False
        self.stream_interval = 1.0 / max(float(stream_fps), 1.0)
        self.inference_interval = 1.0 / max(float(inference_fps), 0.2)
        self.jpeg_quality = int(clamp(int(jpeg_quality), 45, 95))
        self.emotion_backend = emotion_backend
        self.pyfeat_detector = None
        self.pyfeat_ready = False
        self.pyfeat_last_result = None
        self.pyfeat_last_at = 0.0
        self.pyfeat_interval = max(float(pyfeat_interval), 0.2)

        model_dir = self.root / "models"
        self.object_detector = YoloDetector(model_dir / "yolov8n.onnx", COCO_LABELS)
        self.face_detector = cv2.FaceDetectorYN_create(
            str(model_dir / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.8, 0.3, 5000
        )
        self.emotion_net = cv2.dnn.readNetFromONNX(str(model_dir / "emotion-ferplus-12-int8.onnx"))
        self.hand_gestures = HandGestureRecognizer(model_dir / "hand_landmarker.task")
        self.load_pyfeat()

    def load_pyfeat(self):
        if self.emotion_backend != "pyfeat":
            return
        try:
            from feat import Detectorv2

            self.pyfeat_detector = Detectorv2(device="cpu")
            self.pyfeat_ready = True
            print("py-feat valence backend ready", flush=True)
        except Exception as exc:
            self.pyfeat_ready = False
            print(f"py-feat unavailable, falling back to FER+: {exc}", flush=True)

    def scan_cameras(self):
        cameras = []
        for index in range(8):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cameras.append(
                    {"index": index, "name": f"Local camera {index}", "source_type": "camera", "available": True}
                )
            cap.release()
        return cameras

    def start(self):
        if self.running:
            return
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.capture_thread.start()
        self.inference_thread.start()

    def attach_loop(self, loop):
        self.loop = loop

    def connect_camera(self, camera_index):
        with self.lock:
            self.camera_index = int(camera_index)
        return {"success": True, "message": f"switching to local camera {camera_index}"}

    def snapshot_state(self):
        with self.lock:
            return self.latest_state

    def snapshot_jpeg(self):
        with self.lock:
            return self.image_seq, self.latest_jpeg

    def capture_loop(self):
        active_index = None
        cap = None
        while self.running:
            loop_start = time.time()
            with self.lock:
                desired_index = self.camera_index
            if cap is None or desired_index != active_index:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(desired_index)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                active_index = desired_index
            if not cap.isOpened():
                time.sleep(0.5)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            frame = self.resize_frame(frame)

            with self.lock:
                self.latest_frame = frame.copy()
                detections = list(self.latest_detections)
                faces = list(self.latest_faces)
                hands = list(self.latest_hands)
                state = self.latest_state

            annotated = self.annotate(frame, detections, faces, hands, state)
            ok, encoded = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if ok:
                with self.lock:
                    self.latest_jpeg = bytes(encoded)
                    self.image_seq += 1
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, self.stream_interval - elapsed))

    def resize_frame(self, frame):
        if self.camera_width <= 0 or self.camera_height <= 0:
            return frame
        height, width = frame.shape[:2]
        if width == self.camera_width and height == self.camera_height:
            return frame
        return cv2.resize(frame, (self.camera_width, self.camera_height), interpolation=cv2.INTER_AREA)

    def inference_loop(self):
        while self.running:
            loop_start = time.time()
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            if frame is None:
                time.sleep(0.05)
                continue

            state, detections, faces, hand_result = self.process_frame(frame)
            with self.lock:
                self.latest_state = state
                self.latest_detections = detections
                self.latest_faces = faces
                self.latest_hands = hand_result["hands"]
            self.broadcast({"type": "state", "state": state})
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, self.inference_interval - elapsed))

    def process_frame(self, frame):
        detections = self.object_detector.detect(frame)
        faces = self.detect_faces(frame)
        hand_result = self.hand_gestures.detect(frame)
        state = self.compute_state(frame, detections, faces, hand_result["events"])
        return state, detections, faces, hand_result

    def detect_faces(self, frame):
        self.face_detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.face_detector.detect(frame)
        if faces is None:
            return []
        results = []
        for row in faces:
            x, y, w, h = [int(round(v)) for v in row[:4]]
            box = [max(0, x), max(0, y), max(1, min(frame.shape[1] - x, w)), max(1, min(frame.shape[0] - y, h))]
            landmarks = [(float(row[i]), float(row[i + 1])) for i in range(4, 14, 2)]
            orient = self.estimate_orientation(box, landmarks)
            emotion = self.emotion_result(frame, box)
            results.append(
                {
                    "box": box,
                    "score": float(row[14]),
                    "user_id": "stranger",
                    "user_role": "stranger",
                    "emotion_raw": emotion["score"],
                    "emotion_label": emotion["label"],
                    "emotion_confidence": emotion["confidence"],
                    "emotion_arousal": emotion.get("arousal", 0.0),
                    "emotion_backend": emotion.get("backend", "ferplus"),
                    "orientation": orient,
                }
            )
        results.sort(key=lambda face: face["box"][2] * face["box"][3], reverse=True)
        return results

    def emotion_result(self, frame, box):
        if self.pyfeat_ready:
            result = self.pyfeat_emotion_result(frame, box)
            if result is not None:
                return result

        x, y, w, h = box
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            return {"score": 0.0, "label": "unknown", "confidence": 0.0}
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blob = cv2.dnn.blobFromImage(gray, 1.0 / 255.0, (64, 64), swapRB=False, crop=False)
        self.emotion_net.setInput(blob)
        probs = softmax(self.emotion_net.forward())
        if probs.size < 8:
            return {"score": 0.0, "label": "unknown", "confidence": 0.0}
        best_index = int(np.argmax(probs))
        positive = float(probs[1] + 0.35 * probs[2])
        negative = float(0.80 * probs[3] + probs[4] + 0.80 * probs[5] + 0.60 * probs[6] + 0.50 * probs[7])
        return {
            "score": float(clamp(positive - negative, -1.0, 1.0)),
            "label": FER_LABELS[best_index] if best_index < len(FER_LABELS) else "unknown",
            "confidence": float(probs[best_index]),
            "arousal": 0.0,
            "backend": "ferplus",
        }

    def pyfeat_emotion_result(self, frame, box):
        now = time.time()
        if self.pyfeat_last_result is not None and now - self.pyfeat_last_at < self.pyfeat_interval:
            return self.pyfeat_last_result

        x, y, w, h = box
        margin = int(max(w, h) * 0.35)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            if not cv2.imwrite(path, crop):
                return None
            out = self.pyfeat_detector.detect(path, data_type="image", progress_bar=False)
            if out is None or len(out) == 0:
                return None
            row = out.iloc[0]
            valence = float(row.get("valence", 0.0))
            arousal = float(row.get("arousal", 0.0))
            emotion_columns = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
            scores = {name: float(row.get(name, 0.0)) for name in emotion_columns}
            label = max(scores, key=scores.get) if scores else "unknown"
            confidence = scores.get(label, 0.0)
            result = {
                "score": float(clamp(valence, -1.0, 1.0)),
                "label": label.lower(),
                "confidence": float(clamp(confidence, 0.0, 1.0)),
                "arousal": float(clamp(arousal, -1.0, 1.0)),
                "backend": "pyfeat",
            }
            self.pyfeat_last_result = result
            self.pyfeat_last_at = now
            return result
        except Exception as exc:
            print(f"py-feat emotion failed, using FER+: {exc}", flush=True)
            self.pyfeat_ready = False
            return None
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def estimate_orientation(self, box, landmarks):
        x, y, w, h = box
        right_eye, left_eye, nose, right_mouth, left_mouth = landmarks
        roll = math.degrees(math.atan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]))
        box_cx = x + w * 0.5
        yaw = clamp(((nose[0] - box_cx) / (w * 0.5)) * 45.0, -60.0, 60.0)
        eye_cy = (left_eye[1] + right_eye[1]) * 0.5
        mouth_cy = (left_mouth[1] + right_mouth[1]) * 0.5
        pitch = clamp(((nose[1] - ((eye_cy + mouth_cy) * 0.5)) / (h * 0.25)) * 30.0, -45.0, 45.0)
        return {"valid": True, "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll}

    def compute_state(self, frame, detections, faces, hand_events):
        frame_area = frame.shape[0] * frame.shape[1]
        persons = [d for d in detections if d["label"] == "person"]
        largest = max(persons, key=lambda d: d["box"][2] * d["box"][3], default=None)
        area_ratio = 0.0 if largest is None else (largest["box"][2] * largest["box"][3]) / frame_area
        novelty_labels = []
        novelty_scores = []
        for detection in detections:
            if detection["label"] in {"cat", "dog", "bottle"}:
                novelty_labels.append(detection["label"])
                novelty_scores.append(detection["score"])

        nearest = faces[0] if faces else None
        emotion_valid = False
        emotion_raw = 0.0
        emotion_score = 0.0
        emotion_label = "none"
        emotion_confidence = 0.0
        emotion_arousal = 0.0
        emotion_backend = "none"
        if nearest:
            emotion_raw = nearest["emotion_raw"]
            emotion_label = nearest["emotion_label"]
            emotion_confidence = nearest["emotion_confidence"]
            emotion_arousal = nearest["emotion_arousal"]
            emotion_backend = nearest["emotion_backend"]
            emotion_valid = self.emotion_quality_ok(frame, nearest)
            if emotion_valid:
                if not self.emotion_ema_ready:
                    self.emotion_ema = emotion_raw
                    self.emotion_ema_ready = True
                else:
                    self.emotion_ema = (self.emotion_alpha * emotion_raw) + ((1.0 - self.emotion_alpha) * self.emotion_ema)
                emotion_score = self.emotion_ema
            else:
                emotion_score = self.emotion_ema if self.emotion_ema_ready else emotion_raw
        else:
            self.emotion_ema_ready = False

        gestures = []
        if hand_events:
            user_id = nearest["user_id"] if nearest else "unknown"
            user_role = nearest["user_role"] if nearest else "unknown"
            for event in hand_events:
                gestures.append(
                    {
                        "user_id": user_id,
                        "user_role": user_role,
                        "gesture": f'{user_role}_{event["gesture"]}',
                        "score": event["score"],
                    }
                )
        elif nearest:
            gestures.append(
                {
                    "user_id": nearest["user_id"],
                    "user_role": nearest["user_role"],
                    "gesture": f'{nearest["user_role"]}_detected',
                    "score": nearest["score"],
                }
            )

        return {
            "stamp": time.time(),
            "near_human_present": bool(faces or area_ratio >= 0.12),
            "nearest_user_id": nearest["user_id"] if nearest else "none",
            "nearest_user_role": nearest["user_role"] if nearest else "none",
            "v_user_raw": emotion_score,
            "v_user_raw_raw": emotion_raw,
            "emotion_label": emotion_label,
            "emotion_confidence": emotion_confidence,
            "emotion_arousal": emotion_arousal,
            "emotion_backend": emotion_backend,
            "emotion_valid": emotion_valid,
            "face_orient": nearest["orientation"] if nearest else {"valid": False, "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
            "novelty": bool(novelty_labels),
            "novelty_labels": novelty_labels,
            "novelty_scores": novelty_scores,
            "gestures": gestures,
            "person_count": len(persons),
            "nearest_person_score": 0.0 if largest is None else largest["score"],
            "nearest_person_bbox_area_ratio": area_ratio,
        }

    def emotion_quality_ok(self, frame, face):
        h, w = frame.shape[:2]
        box = face["box"]
        face_area_ratio = (box[2] * box[3]) / max(1, w * h)
        orient = face["orientation"]
        return (
            face["score"] >= 0.85
            and face_area_ratio >= 0.01
            and orient.get("valid", False)
            and abs(orient["yaw_deg"]) <= 35.0
            and abs(orient["pitch_deg"]) <= 30.0
        )

    def annotate(self, frame, detections, faces, hands, state):
        out = frame.copy()
        for detection in detections:
            x, y, w, h = detection["box"]
            color = (0, 200, 255) if detection["label"] == "person" else (180, 180, 40)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(out, f'{detection["label"]} {int(detection["score"] * 100)}%', (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for face in faces:
            x, y, w, h = face["box"]
            color = (40, 40, 230)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                out,
                f'stranger {face["emotion_label"]} raw={face["emotion_raw"]:.2f}',
                (x, min(out.shape[0] - 8, y + h + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        for hand in hands:
            for start, end in HAND_CONNECTIONS:
                cv2.line(out, hand["points"][start], hand["points"][end], (60, 220, 180), 2)
            for point in hand["points"]:
                cv2.circle(out, point, 3, (30, 255, 210), -1)
            x, y, w, h = hand["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (60, 220, 180), 1)
            cv2.putText(
                out,
                f'hand open={hand["open_ratio"]:.2f}',
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (60, 220, 180),
                1,
            )
        return out

    def broadcast(self, payload):
        if self.loop is None or not self.websockets:
            return
        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(self.broadcast_async(payload)))

    async def broadcast_async(self, payload):
        stale = []
        for ws in self.websockets:
            if ws.closed:
                stale.append(ws)
                continue
            try:
                await ws.send_json(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.websockets.discard(ws)


def make_app(runtime, web_dir):
    routes = web.RouteTableDef()

    @routes.get("/")
    async def index(_request):
        return web.FileResponse(web_dir / "index.html", headers={"Cache-Control": "no-store"})

    @routes.get("/{name:index.js|styles.css}")
    async def asset(request):
        return web.FileResponse(web_dir / request.match_info["name"], headers={"Cache-Control": "no-store"})

    @routes.get("/api/state")
    async def api_state(_request):
        return web.json_response({"state": runtime.snapshot_state()})

    @routes.get("/api/cameras")
    async def api_cameras(_request):
        return web.json_response({"stamp": time.time(), "cameras": runtime.available_cameras})

    @routes.post("/api/connect")
    async def api_connect(request):
        data = await request.json()
        source_type = data.get("source_type", "camera")
        if source_type != "camera":
            return web.json_response({"success": False, "message": "local fallback supports local camera only"})
        return web.json_response(runtime.connect_camera(int(data.get("camera_index", 0))))

    @routes.get("/ws")
    async def websocket(request):
        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(request)
        runtime.websockets.add(ws)
        await ws.send_json({"type": "snapshot", "state": runtime.snapshot_state(), "cameras": {"stamp": time.time(), "cameras": runtime.available_cameras}})
        async for message in ws:
            if message.type == WSMsgType.ERROR:
                break
        runtime.websockets.discard(ws)
        return ws

    @routes.get("/stream.mjpg")
    async def stream(request):
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame", "Cache-Control": "no-store"},
        )
        await response.prepare(request)
        last_seq = -1
        while True:
            seq, jpeg = runtime.snapshot_jpeg()
            if jpeg is not None and seq != last_seq:
                last_seq = seq
                await response.write(b"--frame\r\nContent-Type: image/jpeg\r\n" + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii") + jpeg + b"\r\n")
            await asyncio.sleep(0.02)

    app = web.Application()
    app.add_routes(routes)
    return app


async def main_async(args):
    root = Path(__file__).resolve().parents[1]
    runtime = LocalVisionRuntime(
        root,
        args.camera_index,
        args.emotion_backend,
        args.stream_fps,
        args.inference_fps,
        args.jpeg_quality,
        args.pyfeat_interval,
        args.camera_width,
        args.camera_height,
    )
    runtime.start()
    runtime.attach_loop(asyncio.get_running_loop())
    app = make_app(runtime, root / "web")
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    print(f"local dashboard listening on http://{args.host}:{args.port}", flush=True)
    await asyncio.Event().wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--emotion-backend", choices=["pyfeat", "ferplus"], default="pyfeat")
    parser.add_argument("--stream-fps", type=float, default=20.0)
    parser.add_argument("--inference-fps", type=float, default=5.0)
    parser.add_argument("--jpeg-quality", type=int, default=76)
    parser.add_argument("--pyfeat-interval", type=float, default=1.2)
    parser.add_argument("--camera-width", type=int, default=960)
    parser.add_argument("--camera-height", type=int, default=540)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
