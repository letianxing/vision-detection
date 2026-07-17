#pragma once

#include "vision_detection/emotion_classifier.hpp"
#include "vision_detection/identity_store.hpp"
#include "vision_detection/types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/objdetect/face.hpp>

#include <string>
#include <vector>

namespace vision_detection
{

class FaceAnalyzer
{
public:
  bool load(
    const std::string & detector_model_path, const std::string & recognizer_model_path,
    float score_threshold, float nms_threshold, int top_k);

  [[nodiscard]] bool ready() const;
  [[nodiscard]] bool canExtractEmbeddings() const;

  [[nodiscard]] std::vector<FaceObservation> analyze(
    const cv::Mat & bgr_frame, const IdentityStore & identity_store,
    float identity_threshold, EmotionClassifier * emotion_classifier);

private:
  static FaceOrientation estimateOrientation(const cv::Rect & box, const std::array<cv::Point2f, 5> & landmarks);
  static cv::Rect clampRect(const cv::Rect & rect, const cv::Size & size);

  cv::Ptr<cv::FaceDetectorYN> detector_;
  cv::Ptr<cv::FaceRecognizerSF> recognizer_;
  float score_threshold_{0.9F};
  float nms_threshold_{0.3F};
  int top_k_{5000};
  bool ready_{false};
  bool recognizer_ready_{false};
};

}  // namespace vision_detection
