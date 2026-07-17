#include "vision_detection/object_detector.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <stdexcept>

namespace vision_detection
{
namespace
{

std::vector<std::string> cocoLabels()
{
  return {
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
    "hair drier", "toothbrush"};
}

float clampFloat(float value, float low, float high)
{
  return std::max(low, std::min(value, high));
}

}  // namespace

bool ObjectDetector::load(
  const std::string & model_path, float confidence_threshold, float nms_threshold,
  int input_width, int input_height)
{
  return load(
    model_path, confidence_threshold, nms_threshold, input_width, input_height,
    cocoLabels());
}

bool ObjectDetector::load(
  const std::string & model_path, float confidence_threshold, float nms_threshold,
  int input_width, int input_height, const std::vector<std::string> & labels)
{
  ready_ = false;
  labels_ = labels.empty() ? cocoLabels() : labels;
  confidence_threshold_ = confidence_threshold;
  nms_threshold_ = nms_threshold;
  input_width_ = input_width;
  input_height_ = input_height;

  if (model_path.empty() || !std::filesystem::exists(model_path)) {
    return false;
  }

  net_ = cv::dnn::readNetFromONNX(model_path);
  net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  ready_ = !net_.empty();
  return ready_;
}

bool ObjectDetector::ready() const
{
  return ready_;
}

const std::vector<std::string> & ObjectDetector::labels() const
{
  return labels_;
}

ObjectDetector::LetterboxInfo ObjectDetector::letterbox(const cv::Mat & image) const
{
  const float scale = std::min(
    static_cast<float>(input_width_) / static_cast<float>(image.cols),
    static_cast<float>(input_height_) / static_cast<float>(image.rows));
  const int resized_width = static_cast<int>(std::round(static_cast<float>(image.cols) * scale));
  const int resized_height = static_cast<int>(std::round(static_cast<float>(image.rows) * scale));
  const int dx = (input_width_ - resized_width) / 2;
  const int dy = (input_height_ - resized_height) / 2;

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(resized_width, resized_height));

  cv::Mat padded(input_height_, input_width_, image.type(), cv::Scalar(114, 114, 114));
  resized.copyTo(padded(cv::Rect(dx, dy, resized_width, resized_height)));
  return LetterboxInfo{padded, scale, dx, dy};
}

cv::Rect ObjectDetector::restoreBox(
  float cx, float cy, float w, float h, const LetterboxInfo & info,
  const cv::Size & original_size) const
{
  const float x1 = (cx - (w * 0.5F) - static_cast<float>(info.dx)) / info.scale;
  const float y1 = (cy - (h * 0.5F) - static_cast<float>(info.dy)) / info.scale;
  const float x2 = (cx + (w * 0.5F) - static_cast<float>(info.dx)) / info.scale;
  const float y2 = (cy + (h * 0.5F) - static_cast<float>(info.dy)) / info.scale;

  const int left = static_cast<int>(std::round(clampFloat(x1, 0.0F, static_cast<float>(original_size.width - 1))));
  const int top = static_cast<int>(std::round(clampFloat(y1, 0.0F, static_cast<float>(original_size.height - 1))));
  const int right = static_cast<int>(std::round(clampFloat(x2, 0.0F, static_cast<float>(original_size.width - 1))));
  const int bottom = static_cast<int>(std::round(clampFloat(y2, 0.0F, static_cast<float>(original_size.height - 1))));

  return cv::Rect(
    cv::Point(left, top),
    cv::Point(std::max(left + 1, right), std::max(top + 1, bottom)));
}

std::vector<Detection> ObjectDetector::detect(const cv::Mat & bgr_frame)
{
  if (!ready_ || bgr_frame.empty()) {
    return {};
  }

  const auto info = letterbox(bgr_frame);
  cv::Mat blob = cv::dnn::blobFromImage(
    info.image, 1.0 / 255.0, cv::Size(input_width_, input_height_), cv::Scalar(),
    true, false);
  net_.setInput(blob);

  std::vector<cv::Mat> outputs;
  net_.forward(outputs, net_.getUnconnectedOutLayersNames());
  if (outputs.empty()) {
    return {};
  }

  cv::Mat raw = outputs.front();
  cv::Mat rows;
  if (raw.dims == 3) {
    const int dim1 = raw.size[1];
    const int dim2 = raw.size[2];
    cv::Mat view(dim1, dim2, CV_32F, raw.ptr<float>());
    if (dim1 < dim2) {
      cv::transpose(view, rows);
    } else {
      rows = view;
    }
  } else if (raw.dims == 2) {
    rows = raw;
  } else {
    return {};
  }

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  for (int row_index = 0; row_index < rows.rows; ++row_index) {
    const float * row = rows.ptr<float>(row_index);
    const int class_count = rows.cols - 4;
    if (class_count <= 0) {
      continue;
    }

    int best_class = -1;
    float best_score = 0.0F;
    for (int class_index = 0; class_index < class_count; ++class_index) {
      const float class_score = row[4 + class_index];
      if (class_score > best_score) {
        best_score = class_score;
        best_class = class_index;
      }
    }

    if (best_class < 0 || best_score < confidence_threshold_) {
      continue;
    }

    boxes.push_back(restoreBox(row[0], row[1], row[2], row[3], info, bgr_frame.size()));
    scores.push_back(best_score);
    class_ids.push_back(best_class);
  }

  std::vector<int> keep;
  cv::dnn::NMSBoxes(boxes, scores, confidence_threshold_, nms_threshold_, keep);

  std::vector<Detection> detections;
  detections.reserve(keep.size());
  for (const int index : keep) {
    const int class_id = class_ids[index];
    Detection detection;
    detection.class_id = class_id;
    detection.label = class_id >= 0 && class_id < static_cast<int>(labels_.size()) ? labels_[class_id] : "unknown";
    detection.score = scores[index];
    detection.box = boxes[index];
    detections.push_back(detection);
  }

  return detections;
}

}  // namespace vision_detection
