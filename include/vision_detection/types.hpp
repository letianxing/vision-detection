#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <string>

namespace vision_detection
{

struct Detection
{
  int class_id{-1};
  std::string label;
  float score{0.0F};
  cv::Rect box;
};

struct FaceOrientation
{
  bool valid{false};
  float yaw_deg{0.0F};
  float pitch_deg{0.0F};
  float roll_deg{0.0F};
};

struct FaceObservation
{
  cv::Rect box;
  std::array<cv::Point2f, 5> landmarks{};
  float face_score{0.0F};
  cv::Mat embedding;
  std::string user_id{"stranger"};
  std::string user_role{"stranger"};
  float identity_similarity{0.0F};
  float emotion_score{0.0F};
  float emotion_raw_score{0.0F};
  std::string emotion_label{"unknown"};
  float emotion_confidence{0.0F};
  bool emotion_valid{false};
  FaceOrientation orientation;
};

}  // namespace vision_detection
