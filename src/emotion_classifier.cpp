#include "vision_detection/emotion_classifier.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

namespace vision_detection
{
namespace
{

const std::vector<std::string> & ferLabels()
{
  static const std::vector<std::string> labels = {
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt"};
  return labels;
}

}  // namespace

bool EmotionClassifier::load(const std::string & model_path)
{
  ready_ = false;
  if (model_path.empty() || !std::filesystem::exists(model_path)) {
    return false;
  }

  net_ = cv::dnn::readNetFromONNX(model_path);
  net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  ready_ = !net_.empty();
  return ready_;
}

bool EmotionClassifier::ready() const
{
  return ready_;
}

EmotionClassifier::Result EmotionClassifier::classify(const cv::Mat & bgr_face)
{
  if (!ready_ || bgr_face.empty()) {
    return {};
  }

  cv::Mat gray;
  cv::cvtColor(bgr_face, gray, cv::COLOR_BGR2GRAY);
  cv::resize(gray, gray, cv::Size(64, 64));

  cv::Mat blob = cv::dnn::blobFromImage(
    gray, 1.0 / 255.0, cv::Size(64, 64), cv::Scalar(), false, false, CV_32F);
  net_.setInput(blob);
  cv::Mat output = net_.forward();
  output = output.reshape(1, 1);

  if (output.cols < 8) {
    return {};
  }

  std::vector<float> logits(output.cols);
  for (int i = 0; i < output.cols; ++i) {
    logits[static_cast<std::size_t>(i)] = output.at<float>(0, i);
  }

  float raw_sum = 0.0F;
  bool all_non_negative = true;
  for (const float value : logits) {
    raw_sum += value;
    all_non_negative = all_non_negative && value >= 0.0F;
  }

  if (all_non_negative && raw_sum > 0.9F && raw_sum < 1.1F) {
    for (float & value : logits) {
      value /= raw_sum;
    }
  } else {
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    float denom = 0.0F;
    for (float & value : logits) {
      value = std::exp(value - max_logit);
      denom += value;
    }
    if (denom <= 0.0F) {
      return {};
    }
    for (float & value : logits) {
      value /= denom;
    }
  }

  return resultFromProbabilities(logits);
}

float EmotionClassifier::score(const cv::Mat & bgr_face)
{
  return classify(bgr_face).score;
}

EmotionClassifier::Result EmotionClassifier::resultFromProbabilities(
  const std::vector<float> & probabilities)
{
  Result result;
  if (probabilities.size() < 8) {
    return result;
  }

  auto best = std::max_element(probabilities.begin(), probabilities.end());
  const auto best_index = static_cast<std::size_t>(std::distance(probabilities.begin(), best));
  result.label = best_index < ferLabels().size() ? ferLabels()[best_index] : "unknown";
  result.confidence = *best;
  result.probabilities = probabilities;

  // FER+ order: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt.
  const auto & logits = probabilities;
  const float positive = logits[1] + (0.35F * logits[2]);
  const float negative =
    (0.80F * logits[3]) + logits[4] + (0.80F * logits[5]) +
    (0.60F * logits[6]) + (0.50F * logits[7]);
  result.score = std::clamp(positive - negative, -1.0F, 1.0F);
  return result;
}

}  // namespace vision_detection
