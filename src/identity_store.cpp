#include "vision_detection/identity_store.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>

namespace vision_detection
{

bool IdentityStore::load(const std::string & path)
{
  entries_.clear();
  if (path.empty() || !std::filesystem::exists(path)) {
    return false;
  }

  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  }

  const cv::FileNode identities = fs["identities"];
  if (!identities.isSeq()) {
    return false;
  }

  for (const auto & node : identities) {
    Entry entry;
    node["user_id"] >> entry.user_id;
    node["user_role"] >> entry.user_role;
    node["embedding"] >> entry.embedding;
    if (!entry.user_id.empty() && !entry.embedding.empty()) {
      entry.embedding = flattenFloat(entry.embedding);
      if (entry.user_role.empty()) {
        entry.user_role = "known";
      }
      entries_.push_back(entry);
    }
  }

  return true;
}

bool IdentityStore::save(const std::string & path) const
{
  if (path.empty()) {
    return false;
  }

  const std::filesystem::path output_path(path);
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  cv::FileStorage fs(path, cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    return false;
  }

  fs << "identities" << "[";
  for (const auto & entry : entries_) {
    fs << "{";
    fs << "user_id" << entry.user_id;
    fs << "user_role" << entry.user_role;
    fs << "embedding" << entry.embedding;
    fs << "}";
  }
  fs << "]";
  return true;
}

void IdentityStore::addOrUpdate(
  const std::string & user_id, const std::string & user_role,
  const cv::Mat & embedding)
{
  if (user_id.empty() || embedding.empty()) {
    return;
  }

  const cv::Mat flattened = flattenFloat(embedding);
  auto iter = std::find_if(
    entries_.begin(), entries_.end(),
    [&](const Entry & entry) { return entry.user_id == user_id; });
  if (iter == entries_.end()) {
    entries_.push_back(Entry{user_id, user_role.empty() ? "known" : user_role, flattened});
    return;
  }

  iter->user_role = user_role.empty() ? iter->user_role : user_role;
  iter->embedding = flattened;
}

IdentityStore::Match IdentityStore::match(const cv::Mat & embedding, float threshold) const
{
  if (embedding.empty() || entries_.empty()) {
    return {};
  }

  const cv::Mat flattened = flattenFloat(embedding);
  Match best;
  best.similarity = -std::numeric_limits<float>::infinity();
  for (const auto & entry : entries_) {
    const float similarity = cosineSimilarity(flattened, entry.embedding);
    if (similarity > best.similarity) {
      best.similarity = similarity;
      best.user_id = entry.user_id;
      best.user_role = entry.user_role;
    }
  }

  if (best.similarity >= threshold) {
    best.matched = true;
    return best;
  }

  return {};
}

std::size_t IdentityStore::size() const
{
  return entries_.size();
}

cv::Mat IdentityStore::flattenFloat(const cv::Mat & embedding)
{
  cv::Mat float_embedding;
  embedding.convertTo(float_embedding, CV_32F);
  return float_embedding.reshape(1, 1).clone();
}

float IdentityStore::cosineSimilarity(const cv::Mat & lhs, const cv::Mat & rhs)
{
  if (lhs.empty() || rhs.empty() || lhs.total() != rhs.total()) {
    return 0.0F;
  }

  const float * lhs_ptr = lhs.ptr<float>(0);
  const float * rhs_ptr = rhs.ptr<float>(0);
  double dot = 0.0;
  double lhs_norm = 0.0;
  double rhs_norm = 0.0;

  for (int i = 0; i < lhs.cols; ++i) {
    dot += static_cast<double>(lhs_ptr[i]) * static_cast<double>(rhs_ptr[i]);
    lhs_norm += static_cast<double>(lhs_ptr[i]) * static_cast<double>(lhs_ptr[i]);
    rhs_norm += static_cast<double>(rhs_ptr[i]) * static_cast<double>(rhs_ptr[i]);
  }

  const double denom = std::sqrt(lhs_norm) * std::sqrt(rhs_norm);
  if (denom <= std::numeric_limits<double>::epsilon()) {
    return 0.0F;
  }

  return static_cast<float>(dot / denom);
}

}  // namespace vision_detection
