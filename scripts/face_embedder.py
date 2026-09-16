"""Swappable face embedding backends.

Identity accuracy now feeds the attention system directly: the memory channel is
scaled by recognition confidence, so a better face embedding makes remembered
context usable sooner, and a worse one makes the robot fall back to live
audio-visual evidence rather than mis-attributing somebody's history.

Two backends:

  sface    OpenCV FaceRecognizerSF, 128-D. Apache-2.0 weights from OpenCV Zoo.
           The default, because it carries no licence restriction.

  arcface  InsightFace buffalo_l w600k_r50, 512-D, ONNX Runtime. Substantially
           stronger on profile views and across age, but its weights and the
           data behind them are released for non-commercial research only.
           Opt in explicitly once the intended use is settled.

Cosine thresholds do not transfer between backends and neither default has been
calibrated on real faces here. Treat both as engineering starting points.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

# Canonical ArcFace 112x112 destination points, in the same order YuNet reports
# its landmarks: right eye, left eye, nose tip, right mouth corner, left mouth
# corner (right/left as seen by the subject).
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963],
                             [73.5318, 51.5014],
                             [56.0252, 71.7366],
                             [41.5493, 92.3655],
                             [70.7299, 92.2041]], dtype=np.float32)

BACKENDS = {
    "sface": {"model": "face_recognition_sface_2021dec.onnx", "dimension": 128, "threshold": 0.38,
              "licence": "Apache-2.0 (OpenCV Zoo)", "commercial_use": True,
              "label": "SFace 128 维"},
    "arcface": {"model": "w600k_r50.onnx", "dimension": 512, "threshold": 0.35,
                "licence": "InsightFace 权重与训练数据仅限非商业研究用途", "commercial_use": False,
                "label": "ArcFace w600k_r50 512 维"},
}


def _normalise(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else None


class SFaceEmbedder:
    name = "sface"

    def __init__(self, model_path):
        self.recognizer = cv2.FaceRecognizerSF_create(str(model_path), "")

    def embed(self, frame, face_row):
        aligned = self.recognizer.alignCrop(frame, np.asarray(face_row, dtype=np.float32))
        return _normalise(self.recognizer.feature(aligned))


class ArcFaceEmbedder:
    name = "arcface"

    def __init__(self, model_path, threads=2):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def align(self, frame, face_row):
        """Similarity transform from the five detected points to the template.

        Falls back to a plain resize of the box when the landmarks are missing,
        which is worse but honest: it never fabricates landmark positions.
        """
        row = np.asarray(face_row, dtype=np.float32).reshape(-1)
        if row.size >= 14:
            source = row[4:14].reshape(5, 2)
            matrix, _ = cv2.estimateAffinePartial2D(source, ARCFACE_TEMPLATE, method=cv2.LMEDS)
            if matrix is not None:
                return cv2.warpAffine(frame, matrix, (112, 112), borderValue=0.0)
        x, y, width, height = (int(round(value)) for value in row[:4])
        x, y = max(0, x), max(0, y)
        crop = frame[y:y + max(1, height), x:x + max(1, width)]
        return cv2.resize(crop, (112, 112)) if crop.size else None

    def embed(self, frame, face_row):
        aligned = self.align(frame, face_row)
        if aligned is None or aligned.size == 0:
            return None
        blob = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (blob - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        return _normalise(self.session.run(None, {self.input_name: blob})[0])


def create(model_dir, backend=None):
    """Build the requested backend, reporting exactly what is running and why.

    Returns (embedder_or_None, info). A missing model or a backend that will not
    load is reported rather than silently replaced by the other one.
    """
    model_dir = Path(model_dir)
    requested = str(backend or os.environ.get("VISION_FACE_BACKEND") or "sface").strip().lower()
    if requested not in BACKENDS:
        return None, {"backend": requested, "available": False, "reason": f"未知的人脸后端 {requested}"}

    spec = BACKENDS[requested]
    path = model_dir / spec["model"]
    info = {"backend": requested, "label": spec["label"], "dimension": spec["dimension"],
            "threshold": spec["threshold"], "licence": spec["licence"],
            "commercial_use": spec["commercial_use"], "model_path": str(path), "available": False,
            "calibrated": False,
            "note": "余弦阈值不跨后端通用，且两个默认值都未在真人数据上标定。"}
    if not path.exists():
        info["reason"] = f"缺少模型文件 {path.name}"
        return None, info
    try:
        embedder = ArcFaceEmbedder(path) if requested == "arcface" else SFaceEmbedder(path)
    except Exception as exc:
        info["reason"] = str(exc)
        return None, info
    info.update(available=True, reason="loaded")
    if not spec["commercial_use"]:
        info["warning"] = "该权重仅限非商业研究用途；用于产品前请改回 sface 或取得许可。"
    return embedder, info


# Quality gating and multi-frame templates. With a licence-clean but weaker
# backend this is where most of the usable accuracy comes from: averaging a few
# good frames cuts embedding noise, and refusing to decide on a bad frame is
# better than guessing. Thresholds are engineering starting points, not measured.
MIN_FACE_PIXELS = 72          # a face smaller than this carries too little detail
MIN_DETECTION_SCORE = 0.85
MIN_SHARPNESS = 25.0          # Laplacian variance; below this the crop is blurred
MAX_YAW_DEGREES = 40.0
MAX_PITCH_DEGREES = 30.0
TEMPLATE_FRAMES = 5


def face_quality(frame, face_row, orientation=None):
    """How much this particular frame is worth trusting, and why not if it is not."""
    row = np.asarray(face_row, dtype=np.float32).reshape(-1)
    width, height = float(row[2]), float(row[3])
    score = float(row[14]) if row.size >= 15 else 1.0
    reasons = []
    size = min(width, height)
    if size < MIN_FACE_PIXELS:
        reasons.append("face_too_small")
    if score < MIN_DETECTION_SCORE:
        reasons.append("weak_detection")

    yaw = abs(float((orientation or {}).get("yaw", 0.0) or 0.0))
    pitch = abs(float((orientation or {}).get("pitch", 0.0) or 0.0))
    if yaw > MAX_YAW_DEGREES or pitch > MAX_PITCH_DEGREES:
        reasons.append("not_frontal_enough")

    sharpness = 0.0
    x, y = max(0, int(row[0])), max(0, int(row[1]))
    crop = frame[y:y + max(1, int(height)), x:x + max(1, int(width))]
    if crop.size:
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        if sharpness < MIN_SHARPNESS:
            reasons.append("blurred")

    usable = not reasons
    quality = 0.0 if not usable else min(1.0, (min(size / (MIN_FACE_PIXELS * 2), 1.0) * 0.4 +
                                               min(sharpness / (MIN_SHARPNESS * 4), 1.0) * 0.3 +
                                               max(0.0, 1.0 - yaw / MAX_YAW_DEGREES) * 0.3))
    return {"usable": usable, "quality": round(quality, 4), "reasons": reasons,
            "size_px": round(size, 1), "sharpness": round(sharpness, 1),
            "yaw": round(yaw, 1), "detection_score": round(score, 3)}


class TemplateTracker:
    """Averages recent good embeddings of the same face into one template.

    A single frame's embedding is noisy; the mean of a few good ones is not. Only
    frames that passed the quality gate are admitted, so a blurred or profile
    frame cannot drag the template away from the person.
    """

    def __init__(self, frames=TEMPLATE_FRAMES, slots=8):
        self.frames = frames
        self.slots = slots
        self.entries = []

    def _match(self, centre, span):
        for entry in self.entries:
            if abs(entry["centre"][0] - centre[0]) < span * 0.5 and abs(entry["centre"][1] - centre[1]) < span * 0.5:
                return entry
        return None

    def update(self, face_row, embedding, stamp_ms, ttl_ms=2000):
        """Returns the averaged template for this face, or None if nothing yet."""
        row = np.asarray(face_row, dtype=np.float32).reshape(-1)
        centre = (float(row[0] + row[2] / 2), float(row[1] + row[3] / 2))
        span = max(float(row[2]), float(row[3]), 1.0)
        self.entries = [entry for entry in self.entries if 0 <= stamp_ms - entry["stamp_ms"] < ttl_ms]
        entry = self._match(centre, span)
        if entry is None:
            entry = {"centre": centre, "samples": []}
            self.entries.append(entry)
            del self.entries[:-self.slots]
        entry["centre"] = centre
        entry["stamp_ms"] = stamp_ms
        if embedding is not None:
            entry["samples"].append(np.asarray(embedding, dtype=np.float32))
            del entry["samples"][:-self.frames]
        if not entry["samples"]:
            return None
        return _normalise(np.mean(entry["samples"], axis=0))

    def depth(self, face_row, stamp_ms, ttl_ms=2000):
        row = np.asarray(face_row, dtype=np.float32).reshape(-1)
        centre = (float(row[0] + row[2] / 2), float(row[1] + row[3] / 2))
        span = max(float(row[2]), float(row[3]), 1.0)
        entry = self._match(centre, span)
        return len(entry["samples"]) if entry and 0 <= stamp_ms - entry["stamp_ms"] < ttl_ms else 0
