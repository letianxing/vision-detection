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
from aiohttp.client_exceptions import ClientConnectionResetError

from lip_landmarks import LipLandmarker
from flash_events import FlashDetector

from capture_sources import create_capture_source, list_realsense_sources

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
        self.last_debug = {}
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
        samples = self._recent(1.8)
        if len(samples) < 5:
            self.last_debug = {"samples": len(samples)}
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
        open_count_range = float(open_counts.max() - open_counts.min())
        area_start = float(np.median(areas[: max(2, len(areas) // 4)]))
        area_end = float(np.median(areas[-max(2, len(areas) // 4):]))
        area_growth = area_end / max(area_start, 1e-4)
        recent_area_std = float(np.std(areas[-5:]))
        recent_xy_motion = float(
            np.std(xs[-min(5, len(xs)):]) + np.std(ys[-min(5, len(ys)):])
        )
        median_open = float(np.median(opens))
        median_open_count = float(np.median(open_counts))
        median_area = float(np.median(areas))

        wave_turns = self._turn_count(xs, min_delta=0.010)
        curl_turns = self._turn_count(opens, min_delta=0.025)
        z_turns = self._turn_count(tip_z, min_delta=0.018)
        area_turns = self._turn_count(areas, min_delta=0.010)

        self.last_debug = {
            "samples": len(samples),
            "x_range": x_range,
            "y_range": y_range,
            "open_range": open_range,
            "open_count_range": open_count_range,
            "z_range": z_range,
            "area_range": area_range,
            "area_growth": area_growth,
            "median_area": median_area,
            "median_open": median_open,
            "median_open_count": median_open_count,
            "recent_xy_motion": recent_xy_motion,
            "wave_turns": wave_turns,
            "curl_turns": curl_turns,
            "z_turns": z_turns,
            "area_turns": area_turns,
        }

        if x_range > 0.040 and wave_turns >= 1 and y_range < 0.35 and median_open_count >= 3:
            score = clamp((x_range / 0.14) + (0.10 * wave_turns), 0.35, 1.0)
            return "hi", score

        invite_motion = (
            open_range > 0.08 or
            open_count_range >= 1.0 or
            z_range > 0.020 or
            area_range > 0.025
        )
        if (
            invite_motion
            and (curl_turns >= 1 or z_turns >= 1 or area_turns >= 1 or open_range > 0.12)
            and x_range < 0.26
            and y_range < 0.35
        ):
            score = clamp(
                (open_range / 0.24) +
                (open_count_range * 0.18) +
                (z_range / 0.07) +
                (area_range / 0.10),
                0.35,
                1.0,
            )
            return "invite", score

        if (
            median_open_count >= 3
            and median_open > 1.10
            and recent_xy_motion < 0.040
            and open_range < 0.12
            and x_range < 0.12
            and y_range < 0.18
            and (area_growth > 1.08 or median_area > 0.030)
        ):
            score = clamp(((area_growth - 1.0) / 0.25) + (median_area / 0.10), 0.35, 1.0)
            return "reject", score

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
        yolo_size=640,
        object_interval=0.5,
        source_type="opencv",
        source_id="",
        face_backend=None,
    ):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.running = False
        self.camera_index = camera_index
        self.source_type = source_type
        self.source_id = str(source_id or camera_index)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.websockets = set()
        self.loop = None
        self.latest_state = None
        self.latest_jpeg = None
        self.latest_frame = None
        self.latest_depth = None
        self.latest_depth_source = "none"
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
        self.emotion_net = None
        self.pyfeat_detector = None
        self.pyfeat_ready = False
        self.pyfeat_last_result = None
        self.pyfeat_last_at = 0.0
        self.pyfeat_interval = max(float(pyfeat_interval), 0.2)
        self.object_interval = max(float(object_interval), 0.05)
        self.last_object_at = 0.0
        self.cached_detections = []
        self.face_tracks = {}
        self.next_face_track_id = 1
        self.identity_store_path = self.root / "config" / "identities.local.json"
        self.identities = self.load_identities()

        model_dir = self.root / "models"
        self.object_detector = YoloDetector(model_dir / "yolov8n.onnx", COCO_LABELS, size=int(yolo_size))
        self.face_detector = cv2.FaceDetectorYN_create(
            str(model_dir / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.8, 0.3, 5000
        )
        import face_embedder
        self.face_embedder, self.face_backend = face_embedder.create(model_dir, face_backend)
        self.face_recognizer = self.face_embedder
        self.identity_cache = []
        self.identity_interval_ms = int(os.environ.get("VISION_IDENTITY_INTERVAL_MS", "300"))
        print(f"face backend: {self.face_backend.get('label', self.face_backend['backend'])} "
              f"({self.face_backend.get('reason')})", flush=True)
        if self.face_backend.get("warning"):
            print("face backend warning: " + self.face_backend["warning"], flush=True)
        self.hand_gestures = HandGestureRecognizer(model_dir / "hand_landmarker.task")
        self.flash_detector=FlashDetector()
        self.flash_events=deque(maxlen=30)
        self.registration_faces = deque(maxlen=100)
        self.lip_landmarker = LipLandmarker(model_dir / "face_landmarker.task")
        if self.emotion_backend == "ferplus":
            self.emotion_net = cv2.dnn.readNetFromONNX(str(model_dir / "emotion-ferplus-12-int8.onnx"))
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
            print(f"py-feat unavailable; emotion will be invalid until py-feat loads: {exc}", flush=True)

    def scan_cameras(self):
        cameras = []
        for index in range(8):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cameras.append(
                    {"index": index, "name": f"Local camera {index}", "source_type": "opencv", "available": True, "depth": False}
                )
            cap.release()
        cameras.extend(list_realsense_sources())
        return cameras

    def start(self):
        if self.running:
            return
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.capture_thread.start()
        self.inference_thread.start()

    def stop(self):
        self.running = False
        capture_thread = getattr(self, "capture_thread", None)
        inference_thread = getattr(self, "inference_thread", None)
        if capture_thread is not None:
            capture_thread.join(timeout=1.0)
        if inference_thread is not None:
            inference_thread.join(timeout=1.0)
        with self.lock:
            self.latest_frame = None
            self.latest_depth = None
            self.latest_jpeg = None
            self.latest_state = None

    def attach_loop(self, loop):
        self.loop = loop

    def connect_camera(self, camera_index, source_type="opencv"):
        with self.lock:
            self.source_type = "realsense" if source_type == "realsense" else "opencv"
            self.source_id = str(camera_index)
            if self.source_type == "opencv":
                self.camera_index = int(camera_index)
        return {"success": True, "message": f"switching to {self.source_type} source {camera_index}"}

    def snapshot_state(self):
        with self.lock:
            return self.latest_state

    def snapshot_jpeg(self):
        with self.lock:
            return self.image_seq, self.latest_jpeg

    def capture_loop(self):
        active_key = None
        source = None
        while self.running:
            loop_start = time.time()
            with self.lock:
                desired_key = (self.source_type, self.source_id)
            if source is None or desired_key != active_key:
                if source is not None:
                    source.release()
                try:
                    source = create_capture_source(
                        desired_key[0], desired_key[1], self.camera_width, self.camera_height
                    )
                except Exception as exc:
                    print(f"capture source unavailable: {exc}", flush=True)
                    source = None
                    time.sleep(0.5)
                    continue
                active_key = desired_key
            if not source.is_opened():
                time.sleep(0.5)
                continue
            ok, sample = source.read()
            if not ok or sample is None:
                time.sleep(0.05)
                continue
            frame = sample.color_bgr
            frame = self.resize_frame(frame)
            depth = sample.depth_m
            if depth is not None and depth.shape[:2] != frame.shape[:2]:
                depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_depth = None if depth is None else depth.copy()
                self.latest_depth_source = sample.depth_source
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
                depth = None if self.latest_depth is None else self.latest_depth.copy()
                depth_source = self.latest_depth_source
            if frame is None:
                time.sleep(0.05)
                continue

            state, detections, faces, hand_result = self.process_frame(frame, depth, depth_source)
            with self.lock:
                self.latest_state = state
                self.latest_detections = detections
                self.latest_faces = faces
                self.latest_hands = hand_result["hands"]
            self.broadcast({"type": "state", "state": state})
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, self.inference_interval - elapsed))

    def timed(self, name, function, *args):
        started = time.perf_counter()
        result = function(*args)
        if not hasattr(self, "model_timings"):
            self.model_timings = {}
        self.model_timings[name] = {"ms": round((time.perf_counter()-started)*1000, 2), "stamp_ms": int(time.time()*1000)}
        return result

    def process_frame(self, frame, depth=None, depth_source="none"):
        frame_started = time.perf_counter()
        now = time.time()
        if now - self.last_object_at >= self.object_interval:
            detections = self.timed("YOLOv8n", self.object_detector.detect, frame)
            self.cached_detections = detections
            self.last_object_at = now
        else:
            detections = list(self.cached_detections)
        faces = self.detect_faces(frame)
        hand_result = self.timed("MediaPipe Hand", self.hand_gestures.detect, frame)
        state = self.compute_state(frame, detections, faces, hand_result["events"], depth, depth_source)
        self.flash_events.extend(self.flash_detector.update(frame,int(now*1000)))
        state["flash_events"]=list(self.flash_events)
        state["model_timings"] = dict(getattr(self, "model_timings", {}))
        state["frame_processing_ms"] = round((time.perf_counter()-frame_started)*1000, 2)
        previous = getattr(self, "_last_measured_frame", now)
        state["inference_fps"] = round(1 / max(.001, now-previous), 1) if previous != now else None
        self._last_measured_frame = now
        return state, detections, faces, hand_result

    def detect_faces(self, frame):
        self.face_detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.timed("YuNet", self.face_detector.detect, frame)
        if faces is None:
            return []
        results = []
        for row in faces:
            x, y, w, h = [int(round(v)) for v in row[:4]]
            box = [max(0, x), max(0, y), max(1, min(frame.shape[1] - x, w)), max(1, min(frame.shape[0] - y, h))]
            landmarks = [(float(row[i]), float(row[i + 1])) for i in range(4, 14, 2)]
            orient = self.estimate_orientation(box, landmarks)
            emotion = self.timed("Emotion", self.emotion_result, frame, box)
            stamp_ms = int(time.time() * 1000)
            identity = self.cached_identity(row, stamp_ms)
            if identity is None:
                identity = self.timed(self.face_backend.get("backend", "face"), self.identify_face, frame, row)
                self.remember_identity(row, stamp_ms, identity)
            results.append(
                {
                    "box": box,
                    "score": float(row[14]),
                    "user_id": identity["user_id"],
                    "user_role": identity["user_role"],
                    "identity_similarity": identity["similarity"],
                    "_embedding": identity.get("_embedding"),
                    "emotion_raw": emotion["score"],
                    "emotion_label": emotion["label"],
                    "emotion_confidence": emotion["confidence"],
                    "emotion_arousal": emotion.get("arousal", 0.0),
                    "emotion_backend": emotion.get("backend", self.emotion_backend),
                    "emotion_model_valid": emotion.get("valid", True),
                    "orientation": orient,
                    "_landmarks": landmarks,
                    "_row": [float(value) for value in row],
                }
            )
        results.sort(key=lambda face: face["box"][2] * face["box"][3], reverse=True)
        self.assign_face_tracks(results)
        self.timed("MediaPipe Face", self.lip_landmarker.detect, frame, results)
        if len(results)==1 and results[0].get("_embedding") and results[0]["score"]>=.9:
            face=results[0]
            with self.lock:
                self.registration_faces.append((int(time.time()*1000),face["track_id"],face["_embedding"],bool(face.get("lip_motion_valid") and face.get("lip_motion"))))
        return results

    def load_identities(self):
        try:
            data = json.loads(self.identity_store_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}

    def save_identities(self):
        self.identity_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.identity_store_path.write_text(
            json.dumps(self.identities, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def face_embedding(self, frame, face_row):
        if self.face_embedder is None:
            return None
        try:
            vector = self.face_embedder.embed(frame, face_row)
            return None if vector is None else vector.tolist()
        except Exception:
            return None

    def cached_identity(self, face_row, stamp_ms):
        """Identity does not change between frames, so it is not recomputed every
        frame. The cache is keyed on where the face is, because face tracks are
        only assigned after identification."""
        row = np.asarray(face_row, dtype=np.float32).reshape(-1)
        centre = (float(row[0] + row[2] / 2), float(row[1] + row[3] / 2))
        span = max(float(row[2]), float(row[3]), 1.0)
        self.identity_cache = [item for item in self.identity_cache
                               if 0 <= stamp_ms - item["stamp_ms"] < self.identity_interval_ms]
        for item in self.identity_cache:
            if abs(item["centre"][0] - centre[0]) < span * 0.5 and abs(item["centre"][1] - centre[1]) < span * 0.5:
                return item["identity"]
        return None

    def remember_identity(self, face_row, stamp_ms, identity):
        row = np.asarray(face_row, dtype=np.float32).reshape(-1)
        self.identity_cache.append({"centre": (float(row[0] + row[2] / 2), float(row[1] + row[3] / 2)),
                                    "stamp_ms": stamp_ms, "identity": identity})
        if len(self.identity_cache) > 16:
            del self.identity_cache[:-16]

    def identify_face(self, frame, face_row):
        embedding = self.face_embedding(frame, face_row)
        if embedding is None:
            return {"user_id": "stranger", "user_role": "stranger", "similarity": 0.0}
        vector = np.asarray(embedding, dtype=np.float32)
        best = ("stranger", "stranger", 0.0)
        for user_id, entry in self.identities.items():
            saved = np.asarray(entry.get("embedding", []), dtype=np.float32)
            if saved.size != vector.size:
                continue
            similarity = float(np.dot(vector, saved) / (np.linalg.norm(saved) + 1e-6))
            if similarity > best[2]:
                best = (user_id, str(entry.get("role") or "known"), similarity)
        threshold = float(self.face_backend.get("threshold", 0.38))
        if best[2] < threshold:
            return {"user_id": "stranger", "user_role": "stranger", "similarity": best[2], "_embedding": embedding}
        return {"user_id": best[0], "user_role": best[1], "similarity": best[2], "_embedding": embedding}

    def enroll_nearest_face(self, user_id, user_role="known", require_single=False, dry_run=False, started_ms=0, ended_ms=0):
        user_id = str(user_id or "").strip()
        if not user_id:
            return {"success": False, "message": "user_id is required"}
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            faces = list(self.latest_faces)
        if frame is None or not faces:
            return {"success": False, "message": "no face is available"}
        if require_single and len(faces) != 1:
            return {"success": False, "message": "请只保留注册者一张人脸在镜头内"}
        face = faces[0]
        embedding = None
        if require_single:
            with self.lock:
                matched=[(e,moving) for t,k,e,moving in self.registration_faces if k==face.get("track_id") and max(started_ms,int(time.time()*1000)-5000)<=t<=ended_ms+250]
                samples=[e for e,moving in matched]
            if len(samples)<5 or sum(bool(moving) for e,moving in matched)<2:
                return {"success":False,"message":"请保持面对镜头，完整说一遍注册话术"}
            vectors=np.asarray(samples,dtype=np.float32)
            center=vectors.mean(axis=0);center/=max(float(np.linalg.norm(center)),1e-6)
            if float(np.min(vectors@center))<.7:
                return {"success":False,"message":"连续人脸样本不一致，请一人面对镜头重试"}
            embedding=center.tolist()
        else:
            embedding = self.face_embedding(frame, face.get("_row", []))
        if embedding is None:
            # Re-run detection to retain the original YuNet 15-value row.
            self.face_detector.setInputSize((frame.shape[1], frame.shape[0]))
            _, detected = self.face_detector.detect(frame)
            if detected is not None and len(detected):
                embedding = self.face_embedding(frame, detected[0])
        if embedding is None:
            return {"success": False, "message": "face recognizer is unavailable"}
        if not dry_run:
            self.identities[user_id] = {"role": str(user_role or "known"), "embedding": embedding}
            self.save_identities()
        return {
            "success": True,
            "message": f"enrolled {user_id}",
            "user_id": user_id,
            "user_role": str(user_role or "known"),
            "embedding": embedding,
            "embedding_model": self.face_backend.get("backend", "opencv-sface"),
        }

    def assign_face_tracks(self, faces):
        now = time.time()
        available = {
            track_id: state
            for track_id, state in self.face_tracks.items()
            if now - state["seen_at"] <= 2.0
        }
        used = set()
        for face in faces:
            best_id = None
            best_score = 0.0
            for track_id, state in available.items():
                if track_id in used:
                    continue
                score = self.face_box_match_score(face["box"], state["box"])
                if score > best_score:
                    best_id = track_id
                    best_score = score
            if best_id is None or best_score < 0.28:
                best_id = f"face_track_{self.next_face_track_id:04d}"
                self.next_face_track_id += 1
            face["track_id"] = best_id
            self.face_tracks[best_id] = {"box": list(face["box"]), "seen_at": now}
            used.add(best_id)
        self.face_tracks = {
            track_id: state
            for track_id, state in self.face_tracks.items()
            if now - state["seen_at"] <= 2.0
        }

    @staticmethod
    def face_box_match_score(left, right):
        lx, ly, lw, lh = left
        rx, ry, rw, rh = right
        intersection_w = max(0, min(lx + lw, rx + rw) - max(lx, rx))
        intersection_h = max(0, min(ly + lh, ry + rh) - max(ly, ry))
        intersection = intersection_w * intersection_h
        union = max(1, lw * lh + rw * rh - intersection)
        iou = intersection / union
        left_center = np.asarray([lx + lw * 0.5, ly + lh * 0.5], dtype=np.float32)
        right_center = np.asarray([rx + rw * 0.5, ry + rh * 0.5], dtype=np.float32)
        scale = max(30.0, 0.5 * (max(lw, lh) + max(rw, rh)))
        proximity = math.exp(-float(np.linalg.norm(left_center - right_center)) / scale)
        return float(0.65 * iou + 0.35 * proximity)

    def emotion_result(self, frame, box):
        if self.pyfeat_ready:
            result = self.pyfeat_emotion_result(frame, box)
            if result is not None:
                return result
        if self.emotion_backend == "pyfeat":
            return {
                "score": 0.0,
                "label": "pyfeat_unavailable",
                "confidence": 0.0,
                "arousal": 0.0,
                "backend": "pyfeat",
                "valid": False,
            }

        if self.emotion_net is None:
            return {
                "score": 0.0,
                "label": "emotion_unavailable",
                "confidence": 0.0,
                "arousal": 0.0,
                "backend": self.emotion_backend,
                "valid": False,
            }

        x, y, w, h = box
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            return {"score": 0.0, "label": "unknown", "confidence": 0.0, "valid": False}
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blob = cv2.dnn.blobFromImage(gray, 1.0, (64, 64), swapRB=False, crop=False)
        self.emotion_net.setInput(blob)
        probs = softmax(self.emotion_net.forward())
        if probs.size < 8:
            return {"score": 0.0, "label": "unknown", "confidence": 0.0, "valid": False}
        best_index = int(np.argmax(probs))
        positive = float(probs[1] + 0.35 * probs[2])
        negative = float(0.80 * probs[3] + probs[4] + 0.80 * probs[5] + 0.60 * probs[6] + 0.50 * probs[7])
        return {
            "score": float(clamp(positive - negative, -1.0, 1.0)),
            "label": FER_LABELS[best_index] if best_index < len(FER_LABELS) else "unknown",
            "confidence": float(probs[best_index]),
            "arousal": 0.0,
            "backend": "ferplus",
            "valid": bool(probs[best_index] >= .55 and np.sort(probs)[-1]-np.sort(probs)[-2] >= .15),
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
                "valid": True,
            }
            self.pyfeat_last_result = result
            self.pyfeat_last_at = now
            return result
        except Exception as exc:
            print(f"py-feat emotion failed; emotion will be invalid: {exc}", flush=True)
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

    def compute_state(self, frame, detections, faces, hand_events, depth=None, depth_source="none"):
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
            emotion_valid = nearest.get("emotion_model_valid", True) and self.emotion_quality_ok(frame, nearest)
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
        people = self.compute_people(frame, persons, faces, gestures, emotion_score, emotion_raw, emotion_label, emotion_confidence, emotion_arousal, emotion_valid, depth, depth_source)

        return {
            "objects": [{"label":d["label"],"confidence":d["score"],"bbox_area_ratio":self.area_ratio(frame,d["box"]),
                         "center":[(d["box"][0]+d["box"][2]/2)/frame.shape[1],(d["box"][1]+d["box"][3]/2)/frame.shape[0]],
                         "stamp_ms":int(self.last_object_at*1000)} for d in detections if d["label"]!="person"],
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
            "face_backend": {key: self.face_backend.get(key) for key in
                             ("backend", "label", "dimension", "threshold", "licence",
                              "commercial_use", "available", "calibrated", "reason", "warning")
                             if key in self.face_backend},
            "emotion_valid": emotion_valid,
            "face_orient": nearest["orientation"] if nearest else {"valid": False, "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
            "novelty": bool(novelty_labels),
            "novelty_labels": novelty_labels,
            "novelty_scores": novelty_scores,
            "gestures": gestures,
            "person_count": len(persons),
            "nearest_person_score": 0.0 if largest is None else largest["score"],
            "nearest_person_bbox_area_ratio": area_ratio,
            "people": people,
        }

    def compute_people(
        self,
        frame,
        person_detections,
        faces,
        gestures,
        nearest_emotion_score,
        nearest_emotion_raw,
        nearest_emotion_label,
        nearest_emotion_confidence,
        nearest_emotion_arousal,
        nearest_emotion_valid,
        depth,
        depth_source,
    ):
        people = []
        matched = set()
        for index, face in enumerate(faces):
            body_index = self.matching_person_detection(person_detections, face)
            if body_index is not None:
                matched.add(body_index)
            box = person_detections[body_index]["box"] if body_index is not None else face["box"]
            area_ratio = self.area_ratio(frame, box)
            distance_m, distance_confidence, resolved_depth_source = self.distance_for_box(
                frame, box, depth, depth_source, is_face=True
            )
            gaze = self.gaze_score(face)
            facing = self.body_facing_score(face)
            gesture, gesture_score = self.gesture_for_face(face, gestures)
            if index == 0:
                emotion_valence = nearest_emotion_score
                emotion_raw = nearest_emotion_raw
                emotion_label = nearest_emotion_label
                emotion_confidence = nearest_emotion_confidence
                emotion_arousal = nearest_emotion_arousal
                emotion_valid = nearest_emotion_valid
            else:
                emotion_valence = face["emotion_raw"]
                emotion_raw = face["emotion_raw"]
                emotion_label = face["emotion_label"]
                emotion_confidence = face["emotion_confidence"]
                emotion_arousal = face["emotion_arousal"]
                emotion_valid = face.get("emotion_model_valid", True) and self.emotion_quality_ok(frame, face)
            person_id = self.stable_person_id(face["user_id"], face["track_id"])
            people.append(
                {
                    "person_id": person_id,
                    "role": face["user_role"] or "unknown",
                    "face_id": f"{person_id}_face_obs",
                    "body_id": f"{person_id}_body_obs",
                    "voice_id": "",
                    "has_azimuth": True,
                    "azimuth_deg": self.rect_azimuth(frame, box),
                    "has_elevation": True,
                    "elevation_deg": self.rect_elevation(frame, box),
                    "has_distance": distance_m is not None,
                    "distance_m": float(distance_m or 0.0),
                    "distance_confidence": distance_confidence,
                    "depth_source": resolved_depth_source,
                    "reflex_area_ratio": self.area_ratio(frame, face["box"]),
                    "face_visible": True,
                    "face_confidence": clamp(face["score"], 0.0, 1.0),
                    "mouth_open_ratio": face.get("mouth_open_ratio", 0.0),
                    "lip_motion": face.get("lip_motion", False),
                    "lip_motion_valid": face.get("lip_motion_valid", False),
                    "lip_backend": face.get("lip_backend", "none"),
                    "lip_reason": face.get("lip_reason", "unavailable"),
                    "mouth_motion_score": face.get("mouth_motion_score", 0.0),
                    "mouth_roi_features": face.get("mouth_roi_features", []),
                    "gaze_score": gaze,
                    "body_facing_score": facing,
                    "bbox_area_ratio": area_ratio,
                    "engagement_status": self.engagement_status(gaze, facing, gesture),
                    "proxemic_space": self.proxemic_space(area_ratio, distance_m, distance_confidence),
                    "identity_confidence": float(face.get("identity_similarity", 0.0)),
                    "emotion_valence": emotion_valence,
                    "emotion_raw": emotion_raw,
                    "emotion_arousal": emotion_arousal,
                    "emotion_valid": emotion_valid,
                    "emotion_label": emotion_label,
                    "emotion_confidence": emotion_confidence,
                    "gesture": gesture,
                    "gesture_score": gesture_score,
                }
            )
        for index, detection in enumerate(person_detections):
            if index in matched:
                continue
            box = detection["box"]
            area_ratio = self.area_ratio(frame, box)
            distance_m, distance_confidence, resolved_depth_source = self.distance_for_box(
                frame, box, depth, depth_source, is_face=False
            )
            people.append(
                {
                    "person_id": f"vision_body_{index}",
                    "role": "unknown",
                    "face_id": "",
                    "body_id": f"body_{index}",
                    "voice_id": "",
                    "has_azimuth": True,
                    "azimuth_deg": self.rect_azimuth(frame, box),
                    "has_elevation": True,
                    "elevation_deg": self.rect_elevation(frame, box),
                    "has_distance": distance_m is not None,
                    "distance_m": float(distance_m or 0.0),
                    "distance_confidence": distance_confidence,
                    "depth_source": resolved_depth_source,
                    "face_visible": False,
                    "face_confidence": 0.0,
                    "mouth_open_ratio": 0.0,
                    "lip_motion": False,
                    "lip_motion_valid": False,
                    "lip_backend": "none",
                    "lip_reason": "no_face",
                    "mouth_roi_features": [],
                    "gaze_score": 0.0,
                    "body_facing_score": 0.25,
                    "bbox_area_ratio": area_ratio,
                    "engagement_status": "unengaged",
                    "proxemic_space": self.proxemic_space(area_ratio, distance_m, distance_confidence),
                    "identity_confidence": 0.0,
                    "emotion_valence": 0.0,
                    "emotion_raw": 0.0,
                    "emotion_arousal": 0.0,
                    "emotion_valid": False,
                    "emotion_label": "unknown",
                    "emotion_confidence": 0.0,
                    "gesture": "none",
                    "gesture_score": 0.0,
                }
            )
        return people

    def matching_person_detection(self, person_detections, face):
        fx, fy, fw, fh = face["box"]
        cx = fx + fw * 0.5
        cy = fy + fh * 0.5
        best_index = None
        best_score = 0.0
        for index, detection in enumerate(person_detections):
            x, y, w, h = detection["box"]
            contains = x <= cx <= x + w and y <= cy <= y + h
            overlap_w = max(0, min(x + w, fx + fw) - max(x, fx))
            overlap_h = max(0, min(y + h, fy + fh) - max(y, fy))
            score = (1.0 if contains else 0.0) + overlap_w * overlap_h
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    @staticmethod
    def stable_person_id(user_id, face_track_id):
        if user_id and user_id not in {"unknown", "stranger"}:
            return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in user_id)
        return f"vision_{face_track_id}"

    @staticmethod
    def area_ratio(frame, box):
        x, y, w, h = box
        frame_h, frame_w = frame.shape[:2]
        x1 = clamp(x, 0, frame_w)
        y1 = clamp(y, 0, frame_h)
        x2 = clamp(x + w, 0, frame_w)
        y2 = clamp(y + h, 0, frame_h)
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1, frame_w * frame_h))

    @staticmethod
    def rect_azimuth(frame, box):
        x, _y, w, _h = box
        frame_w = frame.shape[1]
        return float(((x + w * 0.5) / max(1, frame_w) - 0.5) * 70.0)

    @staticmethod
    def rect_elevation(frame, box):
        _x, y, _w, h = box
        frame_h = frame.shape[0]
        return float((0.5 - (y + h * 0.5) / max(1, frame_h)) * 43.0)

    @staticmethod
    def gaze_score(face):
        orient = face["orientation"]
        if not orient.get("valid", False):
            return 0.35 if face["score"] > 0.0 else 0.0
        yaw_score = 1.0 - abs(float(orient["yaw_deg"])) / 60.0
        pitch_score = 1.0 - abs(float(orient["pitch_deg"])) / 45.0
        return float(clamp(0.75 * yaw_score + 0.25 * pitch_score, 0.0, 1.0))

    @staticmethod
    def body_facing_score(face):
        orient = face["orientation"]
        if not orient.get("valid", False):
            return 0.35 if face["score"] > 0.0 else 0.25
        return float(clamp(1.0 - abs(float(orient["yaw_deg"])) / 90.0, 0.0, 1.0))

    @staticmethod
    def proxemic_space(area_ratio, distance_m=None, distance_confidence=0.0):
        if distance_m is not None and distance_confidence >= 0.5:
            if distance_m < 0.45:
                return "intimate"
            if distance_m < 1.2:
                return "personal"
            if distance_m < 3.6:
                return "social"
            return "public"
        if area_ratio >= 0.42:
            return "intimate"
        if area_ratio >= 0.20:
            return "personal"
        if area_ratio >= 0.06:
            return "social"
        if area_ratio > 0.0:
            return "public"
        return "unknown"

    @staticmethod
    def distance_for_box(frame, box, depth, depth_source, is_face):
        x, y, w, h = box
        if depth is not None:
            margin_x = int(w * 0.25)
            margin_y = int(h * 0.25)
            region = depth[
                max(0, y + margin_y) : min(depth.shape[0], y + h - margin_y),
                max(0, x + margin_x) : min(depth.shape[1], x + w - margin_x),
            ]
            valid = region[(region > 0.15) & (region < 12.0)]
            if valid.size >= 20:
                return float(np.median(valid)), min(1.0, 0.65 + valid.size / 4000.0), depth_source
        focal_px = frame.shape[1] / (2.0 * math.tan(math.radians(70.0) * 0.5))
        physical_width_m = 0.16 if is_face else 0.48
        estimate = physical_width_m * focal_px / max(1.0, float(w))
        return float(clamp(estimate, 0.25, 8.0)), 0.18 if is_face else 0.10, "monocular_size_proxy"

    @staticmethod
    def engagement_status(gaze_score, facing_score, gesture):
        if gaze_score >= 0.65 or gesture in {"wave", "invite", "call"}:
            return "engaged"
        if gaze_score >= 0.38 or facing_score >= 0.55:
            return "engaging"
        return "unengaged"

    @staticmethod
    def gesture_for_face(face, gestures):
        best = ("none", 0.0)
        for event in gestures:
            if str(event.get("gesture", "")).endswith("_detected"):
                continue
            if (
                event["user_id"] != face["user_id"]
                and event["user_id"] != "unknown"
                and face["user_id"] not in {"unknown", "stranger"}
            ):
                continue
            gesture = LocalVisionRuntime.normalize_gesture(event["gesture"])
            score = clamp(event["score"], 0.0, 1.0)
            if score >= best[1]:
                best = (gesture, score)
        return best

    @staticmethod
    def normalize_gesture(raw):
        gesture = str(raw).lower()
        if "wave" in gesture or "hi" in gesture:
            return "wave"
        if "invite" in gesture or "call" in gesture:
            return "invite"
        if "reject" in gesture or "stop" in gesture:
            return "reject"
        if "detected" in gesture:
            return "detected"
        return "none" if not raw else str(raw)

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
                f'{face.get("user_role", "unknown")} id={str(face.get("user_id", "unknown")) if str(face.get("user_id", "unknown")).isascii() else "face-" + str(face.get("track_id", "unknown"))} {face["emotion_label"]} raw={face["emotion_raw"]:.2f}',
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
        source_type = data.get("source_type", "opencv")
        source_id = data.get("camera_index", data.get("source_id", 0))
        return web.json_response(runtime.connect_camera(source_id, source_type))

    @routes.post("/api/start")
    async def api_start(_request):
        runtime.start()
        return web.json_response({"success": True, "message": "vision capture started"})

    @routes.post("/api/stop")
    async def api_stop(_request):
        runtime.stop()
        return web.json_response({"success": True, "message": "vision capture stopped"})

    @routes.post("/api/enroll")
    async def api_enroll(request):
        data = await request.json()
        result = runtime.enroll_nearest_face(data.get("user_id"), data.get("user_role", "known"), bool(data.get("require_single")), bool(data.get("dry_run")), int(data.get("started_ms") or 0), int(data.get("ended_ms") or 0))
        return web.json_response(result, status=200 if result.get("success") else 400)

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
        try:
            while True:
                seq, jpeg = runtime.snapshot_jpeg()
                if jpeg is not None and seq != last_seq:
                    last_seq = seq
                    await response.write(b"--frame\r\nContent-Type: image/jpeg\r\n" + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii") + jpeg + b"\r\n")
                await asyncio.sleep(0.02)
        except (ClientConnectionResetError, ConnectionResetError, asyncio.CancelledError):
            return response

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
        args.yolo_size,
        args.object_interval,
        args.source_type,
        args.source_id,
        args.face_backend,
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
    parser.add_argument("--yolo-size", type=int, default=640)
    parser.add_argument("--object-interval", type=float, default=0.5)
    parser.add_argument("--source-type", choices=["opencv", "realsense"], default="opencv")
    parser.add_argument("--source-id", default="")
    parser.add_argument("--face-backend", choices=["sface", "arcface"], default=None,
                        help="sface: Apache-2.0 权重，默认。arcface: buffalo_l w600k_r50，更准，"
                             "但权重仅限非商业研究用途。也可用 VISION_FACE_BACKEND 设置。")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
