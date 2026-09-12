#!/usr/bin/env python3

import os
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from vision_detection.msg import EmotionState, GestureEvents, GestureObservation

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:
    mp = None
    mp_python = None
    mp_vision = None


EMOTION_COLUMNS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]


def clamp(value, low, high):
    return max(low, min(value, high))


class PyFeatEmotionBackend:
    def __init__(self, logger, interval_sec):
        self.logger = logger
        self.interval_sec = max(0.1, float(interval_sec))
        self.detector = None
        self.ready = False
        self.last_result = None
        self.last_at = 0.0

    def load(self):
        try:
            from feat import Detectorv2

            self.detector = Detectorv2(device="cpu")
            self.ready = True
            self.logger.info("py-feat Detectorv2 emotion backend ready")
        except Exception as exc:
            self.ready = False
            self.logger.warn(f"py-feat unavailable; advanced emotion disabled: {exc}")

    def analyze(self, frame):
        if not self.ready:
            return None

        now = time.time()
        if self.last_result is not None and now - self.last_at < self.interval_sec:
            return self.last_result

        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            if not cv2.imwrite(path, frame):
                return None
            out = self.detector.detect(path, data_type="image", progress_bar=False)
            result = self._parse_result(out)
            self.last_result = result
            self.last_at = now
            return result
        except Exception as exc:
            self.logger.warn(f"py-feat emotion failed; advanced emotion disabled: {exc}")
            self.ready = False
            return None
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _parse_result(self, out):
        if out is None or len(out) == 0:
            return {
                "valid": False,
                "score": 0.0,
                "raw": 0.0,
                "label": "none",
                "confidence": 0.0,
                "arousal": 0.0,
                "backend": "pyfeat",
            }

        row = self._largest_face_row(out)
        valence = float(row.get("valence", self._fallback_valence(row)))
        arousal = float(row.get("arousal", 0.0))
        scores = {name: float(row.get(name, 0.0)) for name in EMOTION_COLUMNS}
        label = max(scores, key=scores.get) if scores else "unknown"
        confidence = scores.get(label, 0.0)
        score = float(clamp(valence, -1.0, 1.0))
        return {
            "valid": True,
            "score": score,
            "raw": score,
            "label": label.lower(),
            "confidence": float(clamp(confidence, 0.0, 1.0)),
            "arousal": float(clamp(arousal, -1.0, 1.0)),
            "backend": "pyfeat",
        }

    @staticmethod
    def _largest_face_row(out):
        width_col = next((name for name in ["FaceRectWidth", "facebox_w", "Width"] if name in out), None)
        height_col = next((name for name in ["FaceRectHeight", "facebox_h", "Height"] if name in out), None)
        if width_col is None or height_col is None:
            return out.iloc[0]
        areas = out[width_col].astype(float) * out[height_col].astype(float)
        return out.iloc[int(areas.argmax())]

    @staticmethod
    def _fallback_valence(row):
        happy = float(row.get("Happy", 0.0))
        negative = (
            float(row.get("Sad", 0.0))
            + float(row.get("Fear", 0.0))
            + float(row.get("Disgust", 0.0))
            + float(row.get("Anger", 0.0))
        )
        return clamp(happy - negative, -1.0, 1.0)


class HandGestureRecognizer:
    def __init__(self, logger, model_path):
        self.logger = logger
        self.ready = False
        self.detector = None
        self.history = deque(maxlen=48)
        self.last_active = None
        self.last_score = 0.0
        self.last_seen = 0.0
        self.last_active_at = 0.0

        if mp is None or mp_python is None or mp_vision is None:
            self.logger.warn("MediaPipe unavailable; advanced gestures disabled")
            return
        if not Path(model_path).exists():
            self.logger.warn(f"hand landmarker model not found: {model_path}")
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
        self.logger.info("MediaPipe hand landmarker gesture backend ready")

    def detect(self, frame):
        if not self.ready:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(image)
        hands = []
        for index, landmarks in enumerate(result.hand_landmarks):
            handedness = "unknown"
            if index < len(result.handedness) and result.handedness[index]:
                handedness = result.handedness[index][0].category_name
            hands.append(self._hand_features(landmarks, handedness))

        if not hands:
            self.history.clear()
            if time.time() - self.last_seen > 0.5:
                self.last_active = None
            return []

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
        if gesture:
            self.last_active = gesture
            self.last_score = score
            self.last_active_at = now
            return [{"gesture": gesture, "score": score, "handedness": primary["handedness"]}]
        if self.last_active and now - self.last_active_at < 0.45:
            return [
                {
                    "gesture": self.last_active,
                    "score": self.last_score,
                    "handedness": primary["handedness"],
                }
            ]
        return []

    @staticmethod
    def _hand_features(landmarks, handedness):
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        min_xy = points[:, :2].min(axis=0)
        max_xy = points[:, :2].max(axis=0)
        bbox_wh = np.maximum(max_xy - min_xy, 1e-4)
        wrist = points[0, :2]
        mcp = points[[5, 9, 13, 17], :2]
        pips = points[[3, 6, 10, 14, 18], :2]
        tips = points[[4, 8, 12, 16, 20], :2]
        mcp_dist = np.linalg.norm(mcp - wrist, axis=1).mean()
        tip_dist = np.linalg.norm(tips - wrist, axis=1).mean()
        finger_tip_dist = np.linalg.norm(tips - wrist, axis=1)
        finger_pip_dist = np.linalg.norm(pips - wrist, axis=1)
        return {
            "center": (float(points[:, 0].mean()), float(points[:, 1].mean())),
            "area": float(bbox_wh[0] * bbox_wh[1]),
            "open_ratio": float(tip_dist / max(mcp_dist, 1e-4)),
            "open_count": int(np.sum(finger_tip_dist > (finger_pip_dist * 1.04))),
            "tip_z": float(points[[4, 8, 12, 16, 20], 2].mean()),
            "handedness": handedness,
        }

    def _recent(self, seconds):
        now = time.time()
        return [item for item in self.history if now - item["t"] <= seconds]

    def _classify(self):
        samples = self._recent(1.8)
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

        if x_range > 0.040 and wave_turns >= 1 and y_range < 0.35 and median_open_count >= 3:
            return "hi", clamp((x_range / 0.14) + (0.10 * wave_turns), 0.35, 1.0)

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
            return "invite", clamp(
                (open_range / 0.24) +
                (open_count_range * 0.18) +
                (z_range / 0.07) +
                (area_range / 0.10),
                0.35,
                1.0,
            )

        if (
            median_open_count >= 3
            and median_open > 1.10
            and recent_xy_motion < 0.040
            and open_range < 0.12
            and x_range < 0.12
            and y_range < 0.18
            and (area_growth > 1.08 or median_area > 0.030)
        ):
            return "reject", clamp(((area_growth - 1.0) / 0.25) + (median_area / 0.10), 0.35, 1.0)

        return None, 0.0

    @staticmethod
    def _turn_count(values, min_delta):
        if len(values) < 4:
            return 0
        signs = []
        for diff in np.diff(values):
            if abs(float(diff)) < min_delta:
                continue
            signs.append(1 if diff > 0 else -1)
        return sum(1 for prev, curr in zip(signs, signs[1:]) if prev != curr)


class AdvancedPerceptionNode(Node):
    def __init__(self):
        super().__init__("vision_advanced_perception")
        self.declare_parameter("image_topic", "/vision/annotated_image")
        self.declare_parameter("emotion_topic", "/vision/advanced_emotion")
        self.declare_parameter("gesture_topic", "/vision/gesture_events_advanced")
        self.declare_parameter("hand_model_path", "")
        self.declare_parameter("emotion_backend", "pyfeat")
        self.declare_parameter("emotion_interval_sec", 0.7)
        self.declare_parameter("gesture_interval_sec", 0.12)

        self.bridge = CvBridge()
        self.emotion_pub = self.create_publisher(
            EmotionState, self.get_parameter("emotion_topic").value, 10
        )
        self.gesture_pub = self.create_publisher(
            GestureEvents, self.get_parameter("gesture_topic").value, 10
        )

        hand_model_path = str(self.get_parameter("hand_model_path").value)
        if not hand_model_path:
            hand_model_path = str(
                Path(get_package_share_directory("vision_detection"))
                / "models"
                / "hand_landmarker.task"
            )

        self.emotion_backend = None
        if str(self.get_parameter("emotion_backend").value) == "pyfeat":
            self.emotion_backend = PyFeatEmotionBackend(
                self.get_logger(), float(self.get_parameter("emotion_interval_sec").value)
            )
            self.emotion_backend.load()
        self.hand_gestures = HandGestureRecognizer(self.get_logger(), hand_model_path)
        self.gesture_interval_sec = max(
            0.02, float(self.get_parameter("gesture_interval_sec").value)
        )
        self.last_gesture_at = 0.0

        image_topic = str(self.get_parameter("image_topic").value)
        self.create_subscription(Image, image_topic, self._on_image, 10)
        self.get_logger().info(f"advanced perception subscribing to {image_topic}")

    def _on_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"image conversion failed: {exc}")
            return

        if self.emotion_backend is not None:
            result = self.emotion_backend.analyze(frame)
            if result is not None:
                self._publish_emotion(msg, result)

        now = time.time()
        if now - self.last_gesture_at < self.gesture_interval_sec:
            return
        self.last_gesture_at = now

        events = self.hand_gestures.detect(frame)
        gesture_msg = GestureEvents()
        gesture_msg.header = msg.header
        for event in events:
            observation = GestureObservation()
            observation.user_id = "unknown"
            observation.user_role = "unknown"
            observation.gesture = "unknown_" + event["gesture"]
            observation.score = float(event["score"])
            gesture_msg.gestures.append(observation)
        self.gesture_pub.publish(gesture_msg)

    def _publish_emotion(self, image_msg, result):
        msg = EmotionState()
        msg.header = image_msg.header
        msg.v_user_raw = float(result["score"])
        msg.v_user_raw_raw = float(result["raw"])
        msg.emotion_label = str(result["label"])
        msg.emotion_confidence = float(result["confidence"])
        msg.emotion_arousal = float(result["arousal"])
        msg.emotion_valid = bool(result["valid"])
        msg.emotion_backend = str(result["backend"])
        self.emotion_pub.publish(msg)


def main():
    rclpy.init()
    node = AdvancedPerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
