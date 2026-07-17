#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <vector>

namespace vision_detection
{

class EmotionClassifier
{
public:
  struct Result
  {
    float score{0.0F};
    std::string label{"unknown"};
    float confidence{0.0F};
    std::vector<float> probabilities;
  };

  bool load(const std::string & model_path);
  [[nodiscard]] bool ready() const;
  [[nodiscard]] Result classify(const cv::Mat & bgr_face);
  [[nodiscard]] float score(const cv::Mat & bgr_face);

private:
  static Result resultFromProbabilities(const std::vector<float> & probabilities);

  cv::dnn::Net net_;
  bool ready_{false};
};

}  // namespace vision_detection
