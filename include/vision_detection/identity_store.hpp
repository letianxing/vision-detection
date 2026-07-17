#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace vision_detection
{

class IdentityStore
{
public:
  struct Entry
  {
    std::string user_id;
    std::string user_role;
    cv::Mat embedding;
  };

  struct Match
  {
    bool matched{false};
    std::string user_id{"stranger"};
    std::string user_role{"stranger"};
    float similarity{0.0F};
  };

  bool load(const std::string & path);
  bool save(const std::string & path) const;
  void addOrUpdate(const std::string & user_id, const std::string & user_role, const cv::Mat & embedding);
  [[nodiscard]] Match match(const cv::Mat & embedding, float threshold) const;
  [[nodiscard]] std::size_t size() const;

private:
  static cv::Mat flattenFloat(const cv::Mat & embedding);
  static float cosineSimilarity(const cv::Mat & lhs, const cv::Mat & rhs);

  std::vector<Entry> entries_;
};

}  // namespace vision_detection
