#include "vision_detection/face_analyzer.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace vision_detection
{
namespace
{

float radiansToDegrees(float radians)
{
  return radians * 180.0F / static_cast<float>(CV_PI);
}

float clampValue(float value, float low, float high)
{
  return std::max(low, std::min(value, high));
}

}  // namespace

bool FaceAnalyzer::load(
  const std::string & detector_model_path, const std::string & recognizer_model_path,
  float score_threshold, float nms_threshold, int top_k)
{
  ready_ = false;
  recognizer_ready_ = false;
  score_threshold_ = score_threshold;
  nms_threshold_ = nms_threshold;
  top_k_ = top_k;

  if (detector_model_path.empty() || !std::filesystem::exists(detector_model_path)) {
    return false;
  }

  detector_ = cv::FaceDetectorYN::create(
    detector_model_path, "", cv::Size(320, 320), score_threshold_, nms_threshold_,
    top_k_);
  ready_ = !detector_.empty();

  if (!recognizer_model_path.empty() && std::filesystem::exists(recognizer_model_path)) {
    recognizer_ = cv::FaceRecognizerSF::create(recognizer_model_path, "");
    recognizer_ready_ = !recognizer_.empty();
  }

  return ready_;
}

bool FaceAnalyzer::ready() const
{
  return ready_;
}

bool FaceAnalyzer::canExtractEmbeddings() const
{
  return recognizer_ready_;
}

std::vector<FaceObservation> FaceAnalyzer::analyze(
  const cv::Mat & bgr_frame, const IdentityStore & identity_store,
  float identity_threshold, EmotionClassifier * emotion_classifier)
{
  if (!ready_ || bgr_frame.empty()) {
    return {};
  }

  detector_->setInputSize(bgr_frame.size());
  cv::Mat faces;
  detector_->detect(bgr_frame, faces);

  std::vector<FaceObservation> observations;
  for (int i = 0; i < faces.rows; ++i) {
    const float * row = faces.ptr<float>(i);

    FaceObservation observation;
    observation.box = clampRect(
      cv::Rect(
        static_cast<int>(std::round(row[0])),
        static_cast<int>(std::round(row[1])),
        static_cast<int>(std::round(row[2])),
        static_cast<int>(std::round(row[3]))),
      bgr_frame.size());
    observation.landmarks = {
      cv::Point2f(row[4], row[5]),
      cv::Point2f(row[6], row[7]),
      cv::Point2f(row[8], row[9]),
      cv::Point2f(row[10], row[11]),
      cv::Point2f(row[12], row[13])};
    observation.face_score = row[14];
    observation.orientation = estimateOrientation(observation.box, observation.landmarks);

    const cv::Mat face_row = faces.row(i);
    if (recognizer_ready_) {
      cv::Mat aligned;
      cv::Mat feature;
      recognizer_->alignCrop(bgr_frame, face_row, aligned);
      recognizer_->feature(aligned, feature);
      observation.embedding = feature.clone();

      const auto match = identity_store.match(feature, identity_threshold);
      observation.identity_similarity = match.similarity;
      if (match.matched) {
        observation.user_id = match.user_id;
        observation.user_role = match.user_role;
      }
    }

    if (emotion_classifier != nullptr && emotion_classifier->ready()) {
      const cv::Rect roi = clampRect(observation.box, bgr_frame.size());
      const auto emotion = emotion_classifier->classify(bgr_frame(roi));
      observation.emotion_raw_score = emotion.score;
      observation.emotion_score = emotion.score;
      observation.emotion_label = emotion.label;
      observation.emotion_confidence = emotion.confidence;
      observation.emotion_valid = true;
    }

    observations.push_back(observation);
  }

  std::sort(
    observations.begin(), observations.end(),
    [](const FaceObservation & lhs, const FaceObservation & rhs) {
      return lhs.box.area() > rhs.box.area();
    });
  return observations;
}

FaceOrientation FaceAnalyzer::estimateOrientation(
  const cv::Rect & box, const std::array<cv::Point2f, 5> & landmarks)
{
  if (box.width <= 0 || box.height <= 0) {
    return {};
  }

  const cv::Point2f right_eye = landmarks[0];
  const cv::Point2f left_eye = landmarks[1];
  const cv::Point2f nose = landmarks[2];
  const cv::Point2f right_mouth = landmarks[3];
  const cv::Point2f left_mouth = landmarks[4];
  const cv::Point2f eye_center = (left_eye + right_eye) * 0.5F;
  const cv::Point2f mouth_center = (left_mouth + right_mouth) * 0.5F;
  const cv::Point2f box_center(
    static_cast<float>(box.x) + (static_cast<float>(box.width) * 0.5F),
    static_cast<float>(box.y) + (static_cast<float>(box.height) * 0.5F));

  FaceOrientation orientation;
  orientation.valid = true;
  orientation.roll_deg = radiansToDegrees(std::atan2(left_eye.y - right_eye.y, left_eye.x - right_eye.x));
  orientation.yaw_deg = clampValue(
    ((nose.x - box_center.x) / (static_cast<float>(box.width) * 0.5F)) * 45.0F,
    -60.0F, 60.0F);

  const float vertical_mid = (eye_center.y + mouth_center.y) * 0.5F;
  orientation.pitch_deg = clampValue(
    ((nose.y - vertical_mid) / (static_cast<float>(box.height) * 0.25F)) * 30.0F,
    -45.0F, 45.0F);

  return orientation;
}

cv::Rect FaceAnalyzer::clampRect(const cv::Rect & rect, const cv::Size & size)
{
  const int x = std::max(0, rect.x);
  const int y = std::max(0, rect.y);
  const int right = std::min(size.width, rect.x + rect.width);
  const int bottom = std::min(size.height, rect.y + rect.height);
  return cv::Rect(x, y, std::max(1, right - x), std::max(1, bottom - y));
}

}  // namespace vision_detection
