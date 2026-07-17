#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models"
mkdir -p "${MODEL_DIR}"

download() {
  local url="$1"
  local output="$2"
  if [[ -s "${output}" ]]; then
    echo "exists: ${output}"
    return
  fi
  echo "download: ${url}"
  curl -L --fail --retry 3 -o "${output}" "${url}"
}

download \
  "https://huggingface.co/inference4j/yolov8n/resolve/main/model.onnx?download=true" \
  "${MODEL_DIR}/yolov8n.onnx"

download \
  "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx?download=true" \
  "${MODEL_DIR}/face_detection_yunet_2023mar.onnx"

download \
  "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx?download=true" \
  "${MODEL_DIR}/face_recognition_sface_2021dec.onnx"

download \
  "https://huggingface.co/onnxmodelzoo/emotion-ferplus-12-int8/resolve/main/emotion-ferplus-12-int8.onnx?download=true" \
  "${MODEL_DIR}/emotion-ferplus-12-int8.onnx"

download \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
  "${MODEL_DIR}/hand_landmarker.task"

echo "models are ready in ${MODEL_DIR}"
