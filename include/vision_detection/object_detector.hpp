#pragma once

#include "vision_detection/types.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <vector>

namespace vision_detection
{

class ObjectDetector
{
public:
  bool load(
    const std::string & model_path, float confidence_threshold, float nms_threshold,
    int input_width, int input_height);
  bool load(
    const std::string & model_path, float confidence_threshold, float nms_threshold,
    int input_width, int input_height, const std::vector<std::string> & labels);

  [[nodiscard]] bool ready() const;
  [[nodiscard]] std::vector<Detection> detect(const cv::Mat & bgr_frame);
  [[nodiscard]] const std::vector<std::string> & labels() const;

private:
  struct LetterboxInfo
  {
    cv::Mat image;
    float scale{1.0F};
    int dx{0};
    int dy{0};
  };

  [[nodiscard]] LetterboxInfo letterbox(const cv::Mat & image) const;
  [[nodiscard]] cv::Rect restoreBox(float cx, float cy, float w, float h, const LetterboxInfo & info, const cv::Size & original_size) const;

  cv::dnn::Net net_;
  std::vector<std::string> labels_;
  float confidence_threshold_{0.35F};
  float nms_threshold_{0.45F};
  int input_width_{640};
  int input_height_{640};
  bool ready_{false};
};

}  // namespace vision_detection
