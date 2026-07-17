#include "vision_detection/emotion_classifier.hpp"
#include "vision_detection/face_analyzer.hpp"
#include "vision_detection/identity_store.hpp"
#include "vision_detection/object_detector.hpp"

#include "vision_detection/msg/camera_info.hpp"
#include "vision_detection/msg/camera_list.hpp"
#include "vision_detection/msg/emotion_state.hpp"
#include "vision_detection/msg/face_orientation.hpp"
#include "vision_detection/msg/gesture_observation.hpp"
#include "vision_detection/msg/gesture_events.hpp"
#include "vision_detection/msg/vision_signals.hpp"
#include "vision_detection/srv/set_input_source.hpp"

#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace vision_detection
{
namespace
{

std::set<std::string> toSet(const std::vector<std::string> & values)
{
  return {values.begin(), values.end()};
}

cv::Scalar colorForRole(const std::string & role)
{
  if (role == "owner") {
    return cv::Scalar(30, 180, 60);
  }
  if (role == "known") {
    return cv::Scalar(255, 160, 0);
  }
  return cv::Scalar(40, 40, 230);
}

std::string lowerCopy(const std::string & value)
{
  std::string out = value;
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

std::string gestureEventName(const std::string & role, const std::string & raw_label)
{
  const std::string label = lowerCopy(raw_label);
  std::string normalized = label;
  if (
    label.find("wave") != std::string::npos ||
    label.find("hello") != std::string::npos ||
    label.find("hi") != std::string::npos)
  {
    normalized = "wave";
  } else if (
    label.find("invite") != std::string::npos ||
    label.find("come") != std::string::npos ||
    label.find("call") != std::string::npos)
  {
    normalized = "invite";
  } else if (
    label.find("reject") != std::string::npos ||
    label.find("stop") != std::string::npos ||
    label.find("no") != std::string::npos)
  {
    normalized = "reject";
  }
  return role + "_" + normalized;
}

}  // namespace

class VisionDetectionNode final : public rclcpp::Node
{
public:
  VisionDetectionNode()
  : Node("vision_detection")
  {
    declareParameters();
    readParameters();
    loadModels();
    setupPublishers();
    setupFusionSubscriptions();
    setupInput();
    setupServices();
  }

private:
  void declareParameters()
  {
    declare_parameter<std::string>("source_type", "camera");
    declare_parameter<int>("camera_index", 0);
    declare_parameter<int>("camera_width", 1280);
    declare_parameter<int>("camera_height", 720);
    declare_parameter<double>("camera_fps", 15.0);
    declare_parameter<int>("camera_probe_count", 8);
    declare_parameter<std::string>("image_topic", "/camera/image_raw");
    declare_parameter<std::string>("video_path", "");

    declare_parameter<std::string>("yolo_model_path", "models/yolov8n.onnx");
    declare_parameter<std::string>("face_detector_model_path", "models/face_detection_yunet_2023mar.onnx");
    declare_parameter<std::string>("face_recognizer_model_path", "models/face_recognition_sface_2021dec.onnx");
    declare_parameter<std::string>("emotion_model_path", "models/emotion-ferplus-12-int8.onnx");
    declare_parameter<std::string>("gesture_model_path", "");
    declare_parameter<std::string>("identity_store_path", "config/identities.yml");

    declare_parameter<double>("object_confidence_threshold", 0.35);
    declare_parameter<double>("object_nms_threshold", 0.45);
    declare_parameter<int>("object_input_width", 640);
    declare_parameter<int>("object_input_height", 640);
    declare_parameter<double>("gesture_confidence_threshold", 0.35);
    declare_parameter<double>("gesture_nms_threshold", 0.45);
    declare_parameter<std::vector<std::string>>("gesture_labels", {"wave", "invite", "reject"});
    declare_parameter<double>("emotion_smoothing_alpha", 0.25);
    declare_parameter<double>("emotion_min_face_area_ratio", 0.01);
    declare_parameter<double>("emotion_max_abs_yaw_deg", 35.0);
    declare_parameter<double>("emotion_max_abs_pitch_deg", 30.0);
    declare_parameter<double>("emotion_min_face_score", 0.85);
    declare_parameter<double>("face_score_threshold", 0.8);
    declare_parameter<double>("face_nms_threshold", 0.3);
    declare_parameter<int>("face_top_k", 5000);
    declare_parameter<double>("identity_threshold", 0.38);
    declare_parameter<double>("near_person_min_area_ratio", 0.12);
    declare_parameter<std::vector<std::string>>("novelty_labels", {"cat", "dog", "bottle"});
    declare_parameter<bool>("publish_annotated_image", true);
    declare_parameter<bool>("use_advanced_emotion", true);
    declare_parameter<std::string>("advanced_emotion_topic", "/vision/advanced_emotion");
    declare_parameter<double>("advanced_emotion_max_age_sec", 1.5);
    declare_parameter<bool>("use_advanced_gestures", true);
    declare_parameter<std::string>(
      "advanced_gesture_events_topic", "/vision/gesture_events_advanced");
    declare_parameter<double>("advanced_gesture_max_age_sec", 1.5);

    declare_parameter<std::string>("signals_topic", "/vision/signals");
    declare_parameter<std::string>("camera_list_topic", "/vision/available_cameras");
    declare_parameter<std::string>("near_human_topic", "/vision/near_human_present");
    declare_parameter<std::string>("emotion_topic", "/vision/v_user_raw");
    declare_parameter<std::string>("novelty_topic", "/vision/novelty_present");
    declare_parameter<std::string>("face_orientation_topic", "/vision/face_orient");
    declare_parameter<std::string>("gesture_events_topic", "/vision/gesture_events");
    declare_parameter<std::string>("annotated_image_topic", "/vision/annotated_image");

    declare_parameter<std::string>("enroll_user_id", "owner");
    declare_parameter<std::string>("enroll_user_role", "owner");
  }

  void readParameters()
  {
    source_type_ = get_parameter("source_type").as_string();
    camera_index_ = static_cast<int>(get_parameter("camera_index").as_int());
    camera_width_ = static_cast<int>(get_parameter("camera_width").as_int());
    camera_height_ = static_cast<int>(get_parameter("camera_height").as_int());
    camera_fps_ = get_parameter("camera_fps").as_double();
    camera_probe_count_ = static_cast<int>(get_parameter("camera_probe_count").as_int());
    image_topic_ = get_parameter("image_topic").as_string();
    video_path_ = get_parameter("video_path").as_string();

    yolo_model_path_ = get_parameter("yolo_model_path").as_string();
    face_detector_model_path_ = get_parameter("face_detector_model_path").as_string();
    face_recognizer_model_path_ = get_parameter("face_recognizer_model_path").as_string();
    emotion_model_path_ = get_parameter("emotion_model_path").as_string();
    gesture_model_path_ = get_parameter("gesture_model_path").as_string();
    identity_store_path_ = get_parameter("identity_store_path").as_string();

    object_confidence_threshold_ = static_cast<float>(get_parameter("object_confidence_threshold").as_double());
    object_nms_threshold_ = static_cast<float>(get_parameter("object_nms_threshold").as_double());
    object_input_width_ = static_cast<int>(get_parameter("object_input_width").as_int());
    object_input_height_ = static_cast<int>(get_parameter("object_input_height").as_int());
    gesture_confidence_threshold_ = static_cast<float>(get_parameter("gesture_confidence_threshold").as_double());
    gesture_nms_threshold_ = static_cast<float>(get_parameter("gesture_nms_threshold").as_double());
    gesture_labels_ = get_parameter("gesture_labels").as_string_array();
    emotion_smoothing_alpha_ = static_cast<float>(get_parameter("emotion_smoothing_alpha").as_double());
    emotion_min_face_area_ratio_ = static_cast<float>(get_parameter("emotion_min_face_area_ratio").as_double());
    emotion_max_abs_yaw_deg_ = static_cast<float>(get_parameter("emotion_max_abs_yaw_deg").as_double());
    emotion_max_abs_pitch_deg_ = static_cast<float>(get_parameter("emotion_max_abs_pitch_deg").as_double());
    emotion_min_face_score_ = static_cast<float>(get_parameter("emotion_min_face_score").as_double());
    face_score_threshold_ = static_cast<float>(get_parameter("face_score_threshold").as_double());
    face_nms_threshold_ = static_cast<float>(get_parameter("face_nms_threshold").as_double());
    face_top_k_ = static_cast<int>(get_parameter("face_top_k").as_int());
    identity_threshold_ = static_cast<float>(get_parameter("identity_threshold").as_double());
    near_person_min_area_ratio_ = static_cast<float>(get_parameter("near_person_min_area_ratio").as_double());
    novelty_labels_ = get_parameter("novelty_labels").as_string_array();
    novelty_label_set_ = toSet(novelty_labels_);
    publish_annotated_image_ = get_parameter("publish_annotated_image").as_bool();
    use_advanced_emotion_ = get_parameter("use_advanced_emotion").as_bool();
    advanced_emotion_topic_ = get_parameter("advanced_emotion_topic").as_string();
    advanced_emotion_max_age_sec_ = get_parameter("advanced_emotion_max_age_sec").as_double();
    use_advanced_gestures_ = get_parameter("use_advanced_gestures").as_bool();
    advanced_gesture_events_topic_ = get_parameter("advanced_gesture_events_topic").as_string();
    advanced_gesture_max_age_sec_ = get_parameter("advanced_gesture_max_age_sec").as_double();

    signals_topic_ = get_parameter("signals_topic").as_string();
    camera_list_topic_ = get_parameter("camera_list_topic").as_string();
    near_human_topic_ = get_parameter("near_human_topic").as_string();
    emotion_topic_ = get_parameter("emotion_topic").as_string();
    novelty_topic_ = get_parameter("novelty_topic").as_string();
    face_orientation_topic_ = get_parameter("face_orientation_topic").as_string();
    gesture_events_topic_ = get_parameter("gesture_events_topic").as_string();
    annotated_image_topic_ = get_parameter("annotated_image_topic").as_string();
  }

  void loadModels()
  {
    if (object_detector_.load(
        yolo_model_path_, object_confidence_threshold_, object_nms_threshold_,
        object_input_width_, object_input_height_))
    {
      RCLCPP_INFO(get_logger(), "Loaded YOLO model: %s", yolo_model_path_.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "YOLO model not loaded: %s", yolo_model_path_.c_str());
    }

    if (face_analyzer_.load(
        face_detector_model_path_, face_recognizer_model_path_, face_score_threshold_,
        face_nms_threshold_, face_top_k_))
    {
      RCLCPP_INFO(get_logger(), "Loaded face detector: %s", face_detector_model_path_.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "Face detector not loaded: %s", face_detector_model_path_.c_str());
    }

    if (emotion_classifier_.load(emotion_model_path_)) {
      RCLCPP_INFO(get_logger(), "Loaded emotion model: %s", emotion_model_path_.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "Emotion model not loaded: %s", emotion_model_path_.c_str());
    }

    gesture_model_ready_ = gesture_detector_.load(
      gesture_model_path_, gesture_confidence_threshold_, gesture_nms_threshold_,
      object_input_width_, object_input_height_, gesture_labels_);
    if (gesture_model_ready_) {
      RCLCPP_INFO(get_logger(), "Loaded gesture model: %s", gesture_model_path_.c_str());
    } else if (!gesture_model_path_.empty()) {
      RCLCPP_WARN(get_logger(), "Gesture model not loaded: %s", gesture_model_path_.c_str());
    }

    if (identity_store_.load(identity_store_path_)) {
      RCLCPP_INFO(
        get_logger(), "Loaded %zu identities from %s",
        identity_store_.size(), identity_store_path_.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "No identity store loaded yet: %s", identity_store_path_.c_str());
    }
  }

  void setupPublishers()
  {
    signals_pub_ = create_publisher<msg::VisionSignals>(signals_topic_, rclcpp::SensorDataQoS());
    auto camera_list_qos = rclcpp::QoS(1);
    camera_list_qos.transient_local().reliable();
    camera_list_pub_ = create_publisher<msg::CameraList>(camera_list_topic_, camera_list_qos);
    near_human_pub_ = create_publisher<std_msgs::msg::Bool>(near_human_topic_, rclcpp::SensorDataQoS());
    emotion_pub_ = create_publisher<std_msgs::msg::Float32>(emotion_topic_, rclcpp::SensorDataQoS());
    novelty_pub_ = create_publisher<std_msgs::msg::Bool>(novelty_topic_, rclcpp::SensorDataQoS());
    face_orientation_pub_ = create_publisher<msg::FaceOrientation>(
      face_orientation_topic_, rclcpp::SensorDataQoS());
    gesture_events_pub_ = create_publisher<msg::GestureEvents>(
      gesture_events_topic_, rclcpp::SensorDataQoS());
    if (publish_annotated_image_) {
      annotated_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
        annotated_image_topic_, rclcpp::SensorDataQoS());
    }
  }

  void setupFusionSubscriptions()
  {
    if (use_advanced_emotion_) {
      advanced_emotion_sub_ = create_subscription<msg::EmotionState>(
        advanced_emotion_topic_, rclcpp::SensorDataQoS(),
        [this](msg::EmotionState::ConstSharedPtr msg) {
          std::lock_guard<std::mutex> lock(fusion_mutex_);
          last_advanced_emotion_ = *msg;
          has_advanced_emotion_ = true;
          last_advanced_emotion_received_ = now();
        });
      RCLCPP_INFO(
        get_logger(), "Listening for advanced emotion on %s",
        advanced_emotion_topic_.c_str());
    }

    if (use_advanced_gestures_) {
      advanced_gestures_sub_ = create_subscription<msg::GestureEvents>(
        advanced_gesture_events_topic_, rclcpp::SensorDataQoS(),
        [this](msg::GestureEvents::ConstSharedPtr msg) {
          std::lock_guard<std::mutex> lock(fusion_mutex_);
          last_advanced_gestures_ = *msg;
          has_advanced_gestures_ = true;
          last_advanced_gestures_received_ = now();
        });
      RCLCPP_INFO(
        get_logger(), "Listening for advanced gestures on %s",
        advanced_gesture_events_topic_.c_str());
    }
  }

  void setupInput()
  {
    probeLocalCameras();
    publishCameraList();
    camera_list_timer_ = create_wall_timer(
      std::chrono::seconds(5), [this]() {
        publishCameraList();
      });
    configureInput(source_type_, camera_index_, image_topic_, video_path_);
  }

  bool configureInput(
    const std::string & source_type, int camera_index, const std::string & image_topic,
    const std::string & video_path)
  {
    std::lock_guard<std::mutex> lock(input_mutex_);

    if (capture_timer_ != nullptr) {
      capture_timer_->cancel();
      capture_timer_.reset();
    }
    image_sub_.reset();
    if (capture_.isOpened()) {
      capture_.release();
    }

    source_type_ = source_type;
    camera_index_ = camera_index;
    image_topic_ = image_topic;
    video_path_ = video_path;

    if (source_type_ == "ros_topic") {
      image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic_, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
          handleImageMessage(msg);
        });
      RCLCPP_INFO(get_logger(), "Reading frames from ROS2 topic: %s", image_topic_.c_str());
      return true;
    }

    if (source_type_ == "video_file") {
      capture_.open(video_path_);
      RCLCPP_INFO(get_logger(), "Reading frames from video file: %s", video_path_.c_str());
    } else {
      capture_.open(camera_index_);
      capture_.set(cv::CAP_PROP_FRAME_WIDTH, camera_width_);
      capture_.set(cv::CAP_PROP_FRAME_HEIGHT, camera_height_);
      capture_.set(cv::CAP_PROP_FPS, camera_fps_);
      RCLCPP_INFO(get_logger(), "Reading frames from local camera index: %d", camera_index_);
    }

    if (!capture_.isOpened()) {
      RCLCPP_ERROR(get_logger(), "Unable to open input source type=%s", source_type_.c_str());
      return false;
    }

    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / std::max(1.0, camera_fps_)));
    capture_timer_ = create_wall_timer(period, [this]() {
      cv::Mat frame;
      {
        std::lock_guard<std::mutex> lock(input_mutex_);
        if (!capture_.read(frame) || frame.empty()) {
          if (source_type_ == "video_file") {
            capture_.set(cv::CAP_PROP_POS_FRAMES, 0);
          }
          return;
        }
      }
      processFrame(frame, now(), "bgr8");
    });
    return true;
  }

  void probeLocalCameras()
  {
    available_camera_indices_.clear();
    for (int index = 0; index < std::max(0, camera_probe_count_); ++index) {
      cv::VideoCapture probe(index);
      if (probe.isOpened()) {
        available_camera_indices_.push_back(index);
      }
      probe.release();
    }
  }

  void publishCameraList()
  {
    if (camera_list_pub_ == nullptr) {
      return;
    }

    msg::CameraList list;
    list.header.stamp = now();
    list.header.frame_id = "vision_detection";
    for (const int index : available_camera_indices_) {
      msg::CameraInfo camera;
      camera.index = index;
      camera.name = "Local camera " + std::to_string(index);
      camera.source_type = "camera";
      camera.available = true;
      list.cameras.push_back(camera);
    }
    camera_list_pub_->publish(list);
  }

  void setupServices()
  {
    enroll_service_ = create_service<std_srvs::srv::Trigger>(
      "/vision/enroll_nearest_face",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        enrollNearestFace(response);
      });
    set_input_service_ = create_service<srv::SetInputSource>(
      "/vision/set_input_source",
      [this](
        const std::shared_ptr<srv::SetInputSource::Request> request,
        std::shared_ptr<srv::SetInputSource::Response> response) {
        const bool ok = configureInput(
          request->source_type, request->camera_index, request->image_topic,
          request->video_path);
        response->success = ok;
        if (ok) {
          response->message = "connected source_type=" + request->source_type;
        } else {
          response->message = "failed to connect source_type=" + request->source_type;
        }
      });
  }

  void handleImageMessage(sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    try {
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
      processFrame(cv_ptr->image, msg->header.stamp, msg->encoding);
    } catch (const cv_bridge::Exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "cv_bridge conversion failed: %s", error.what());
    }
  }

  void processFrame(const cv::Mat & frame, const rclcpp::Time & stamp, const std::string & encoding)
  {
    (void)encoding;
    if (frame.empty()) {
      return;
    }

    const auto detections = object_detector_.detect(frame);
    const auto gesture_detections = gesture_detector_.detect(frame);
    const auto faces = face_analyzer_.analyze(
      frame, identity_store_, identity_threshold_,
      emotion_classifier_.ready() ? &emotion_classifier_ : nullptr);

    last_faces_ = faces;

    msg::VisionSignals state;
    state.header.stamp = stamp;
    state.header.frame_id = "vision_detection";

    fillObjectSignals(frame, detections, state);
    fillFaceSignals(frame, faces, gesture_detections, state);
    publishSignals(state);

    if (publish_annotated_image_ && annotated_image_pub_ != nullptr) {
      publishAnnotatedImage(frame, detections, gesture_detections, faces, state.header.stamp);
    }
  }

  void fillObjectSignals(
    const cv::Mat & frame, const std::vector<Detection> & detections,
    msg::VisionSignals & state) const
  {
    const float frame_area = static_cast<float>(frame.cols * frame.rows);
    float largest_person_area_ratio = 0.0F;
    float largest_person_score = 0.0F;

    for (const auto & detection : detections) {
      if (detection.label == "person") {
        ++state.person_count;
        const float area_ratio = static_cast<float>(detection.box.area()) / frame_area;
        if (area_ratio > largest_person_area_ratio) {
          largest_person_area_ratio = area_ratio;
          largest_person_score = detection.score;
        }
      }

      if (novelty_label_set_.find(detection.label) != novelty_label_set_.end()) {
        state.novelty = true;
        state.novelty_labels.push_back(detection.label);
        state.novelty_scores.push_back(detection.score);
      }
    }

    state.nearest_person_bbox_area_ratio = largest_person_area_ratio;
    state.nearest_person_score = largest_person_score;
    state.near_human_present = largest_person_area_ratio >= near_person_min_area_ratio_;
  }

  void fillFaceSignals(
    const cv::Mat & frame,
    const std::vector<FaceObservation> & faces,
    const std::vector<Detection> & gesture_detections,
    msg::VisionSignals & state)
  {
    if (faces.empty()) {
      state.nearest_user_id = "none";
      state.nearest_user_role = "none";
      state.v_user_raw = 0.0F;
      state.v_user_raw_raw = 0.0F;
      state.emotion_label = "none";
      state.emotion_confidence = 0.0F;
      state.emotion_arousal = 0.0F;
      state.emotion_valid = false;
      state.emotion_backend = "none";
      state.face_orient.valid = false;
      emotion_ema_initialized_ = false;
      applyAdvancedEmotionIfFresh(state);
      if (applyAdvancedGesturesIfFresh(state, "unknown", "unknown")) {
        return;
      }
      for (const auto & detection : gesture_detections) {
        msg::GestureObservation gesture;
        gesture.user_id = "unknown";
        gesture.user_role = "unknown";
        gesture.gesture = gestureEventName("unknown", detection.label);
        gesture.score = detection.score;
        state.gestures.push_back(gesture);
      }
      return;
    }

    const auto & nearest = faces.front();
    state.near_human_present = true;
    state.nearest_user_id = nearest.user_id;
    state.nearest_user_role = nearest.user_role;
    state.v_user_raw_raw = nearest.emotion_raw_score;
    state.emotion_label = nearest.emotion_label;
    state.emotion_confidence = nearest.emotion_confidence;
    state.emotion_arousal = 0.0F;
    state.emotion_valid = isEmotionQualityGood(frame, nearest);
    state.emotion_backend = "ferplus";
    if (state.emotion_valid) {
      const float alpha = std::clamp(emotion_smoothing_alpha_, 0.01F, 1.0F);
      if (!emotion_ema_initialized_) {
        emotion_ema_ = nearest.emotion_raw_score;
        emotion_ema_initialized_ = true;
      } else {
        emotion_ema_ = (alpha * nearest.emotion_raw_score) + ((1.0F - alpha) * emotion_ema_);
      }
      state.v_user_raw = emotion_ema_;
    } else {
      state.v_user_raw = emotion_ema_initialized_ ? emotion_ema_ : nearest.emotion_raw_score;
    }
    state.face_orient.valid = nearest.orientation.valid;
    state.face_orient.yaw_deg = nearest.orientation.yaw_deg;
    state.face_orient.pitch_deg = nearest.orientation.pitch_deg;
    state.face_orient.roll_deg = nearest.orientation.roll_deg;

    applyAdvancedEmotionIfFresh(state);
    if (applyAdvancedGesturesIfFresh(state, nearest.user_id, nearest.user_role)) {
      return;
    }

    if (!gesture_detections.empty()) {
      for (const auto & detection : gesture_detections) {
        const auto * face = nearestFaceForGesture(faces, detection.box);
        const std::string user_id = face == nullptr ? "unknown" : face->user_id;
        const std::string user_role = face == nullptr ? "unknown" : face->user_role;
        msg::GestureObservation gesture;
        gesture.user_id = user_id;
        gesture.user_role = user_role;
        gesture.gesture = gestureEventName(user_role, detection.label);
        gesture.score = detection.score;
        state.gestures.push_back(gesture);
      }
      return;
    }

    if (!gesture_model_ready_) {
      for (const auto & face : faces) {
        msg::GestureObservation gesture;
        gesture.user_id = face.user_id;
        gesture.user_role = face.user_role;
        gesture.gesture = face.user_role + "_detected";
        gesture.score = std::max(face.identity_similarity, face.face_score);
        state.gestures.push_back(gesture);
      }
    }
  }

  bool applyAdvancedEmotionIfFresh(msg::VisionSignals & state)
  {
    if (!use_advanced_emotion_) {
      return false;
    }

    msg::EmotionState advanced;
    {
      std::lock_guard<std::mutex> lock(fusion_mutex_);
      if (!has_advanced_emotion_) {
        return false;
      }
      if ((now() - last_advanced_emotion_received_).seconds() > advanced_emotion_max_age_sec_) {
        return false;
      }
      advanced = last_advanced_emotion_;
    }

    if (!advanced.emotion_valid) {
      return false;
    }

    state.near_human_present = true;
    state.v_user_raw = std::clamp(advanced.v_user_raw, -1.0F, 1.0F);
    state.v_user_raw_raw = std::clamp(advanced.v_user_raw_raw, -1.0F, 1.0F);
    state.emotion_label = advanced.emotion_label;
    state.emotion_confidence = std::clamp(advanced.emotion_confidence, 0.0F, 1.0F);
    state.emotion_arousal = std::clamp(advanced.emotion_arousal, -1.0F, 1.0F);
    state.emotion_valid = true;
    state.emotion_backend = advanced.emotion_backend.empty() ? "advanced" : advanced.emotion_backend;
    return true;
  }

  bool applyAdvancedGesturesIfFresh(
    msg::VisionSignals & state, const std::string & user_id,
    const std::string & user_role)
  {
    if (!use_advanced_gestures_) {
      return false;
    }

    msg::GestureEvents advanced;
    {
      std::lock_guard<std::mutex> lock(fusion_mutex_);
      if (!has_advanced_gestures_) {
        return false;
      }
      if ((now() - last_advanced_gestures_received_).seconds() > advanced_gesture_max_age_sec_) {
        return false;
      }
      advanced = last_advanced_gestures_;
    }

    if (advanced.gestures.empty()) {
      return false;
    }

    state.gestures.clear();
    for (const auto & incoming : advanced.gestures) {
      msg::GestureObservation gesture = incoming;
      if (gesture.user_id.empty() || gesture.user_id == "unknown") {
        gesture.user_id = user_id;
      }
      if (gesture.user_role.empty() || gesture.user_role == "unknown") {
        const std::string original_role = gesture.user_role.empty() ? "unknown" : gesture.user_role;
        gesture.user_role = user_role;
        const std::string original_prefix = original_role + "_";
        if (gesture.gesture.rfind(original_prefix, 0) == 0) {
          gesture.gesture = user_role + "_" + gesture.gesture.substr(original_prefix.size());
        }
      }
      if (gesture.gesture.find('_') == std::string::npos) {
        gesture.gesture = gestureEventName(gesture.user_role, gesture.gesture);
      }
      state.gestures.push_back(gesture);
    }
    return true;
  }

  bool isEmotionQualityGood(const cv::Mat & frame, const FaceObservation & face) const
  {
    if (frame.empty() || !face.emotion_valid || !face.orientation.valid) {
      return false;
    }
    const float frame_area = static_cast<float>(frame.cols * frame.rows);
    const float face_area_ratio = static_cast<float>(face.box.area()) / std::max(1.0F, frame_area);
    return face.face_score >= emotion_min_face_score_ &&
      face_area_ratio >= emotion_min_face_area_ratio_ &&
      std::abs(face.orientation.yaw_deg) <= emotion_max_abs_yaw_deg_ &&
      std::abs(face.orientation.pitch_deg) <= emotion_max_abs_pitch_deg_;
  }

  const FaceObservation * nearestFaceForGesture(
    const std::vector<FaceObservation> & faces, const cv::Rect & gesture_box) const
  {
    if (faces.empty()) {
      return nullptr;
    }

    const float gesture_center_x = static_cast<float>(gesture_box.x) +
      (static_cast<float>(gesture_box.width) * 0.5F);
    const FaceObservation * best = &faces.front();
    float best_distance = std::numeric_limits<float>::max();
    for (const auto & face : faces) {
      const float face_center_x = static_cast<float>(face.box.x) +
        (static_cast<float>(face.box.width) * 0.5F);
      const float distance = std::abs(face_center_x - gesture_center_x);
      if (distance < best_distance) {
        best_distance = distance;
        best = &face;
      }
    }
    return best;
  }

  void publishSignals(const msg::VisionSignals & state)
  {
    signals_pub_->publish(state);

    std_msgs::msg::Bool near_human;
    near_human.data = state.near_human_present;
    near_human_pub_->publish(near_human);

    std_msgs::msg::Float32 emotion;
    emotion.data = state.v_user_raw;
    emotion_pub_->publish(emotion);

    std_msgs::msg::Bool novelty;
    novelty.data = state.novelty;
    novelty_pub_->publish(novelty);

    face_orientation_pub_->publish(state.face_orient);

    msg::GestureEvents gestures;
    gestures.header = state.header;
    gestures.gestures = state.gestures;
    gesture_events_pub_->publish(gestures);
  }

  void publishAnnotatedImage(
    const cv::Mat & frame, const std::vector<Detection> & detections,
    const std::vector<Detection> & gesture_detections,
    const std::vector<FaceObservation> & faces, const rclcpp::Time & stamp)
  {
    cv::Mat annotated = frame.clone();

    for (const auto & detection : detections) {
      const cv::Scalar color = detection.label == "person" ? cv::Scalar(0, 200, 255) : cv::Scalar(180, 180, 40);
      cv::rectangle(annotated, detection.box, color, 2);
      cv::putText(
        annotated, detection.label + " " + std::to_string(static_cast<int>(detection.score * 100.0F)) + "%",
        cv::Point(detection.box.x, std::max(20, detection.box.y - 6)),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    for (const auto & face : faces) {
      const cv::Scalar color = colorForRole(face.user_role);
      cv::rectangle(annotated, face.box, color, 2);
      cv::putText(
        annotated,
        face.user_id + " emotion=" + std::to_string(face.emotion_score).substr(0, 5),
        cv::Point(face.box.x, std::min(annotated.rows - 8, face.box.y + face.box.height + 18)),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    for (const auto & detection : gesture_detections) {
      const cv::Scalar color(230, 80, 200);
      cv::rectangle(annotated, detection.box, color, 2);
      cv::putText(
        annotated,
        "gesture " + detection.label + " " +
          std::to_string(static_cast<int>(detection.score * 100.0F)) + "%",
        cv::Point(detection.box.x, std::max(20, detection.box.y - 6)),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    auto out_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", annotated).toImageMsg();
    out_msg->header.stamp = stamp;
    out_msg->header.frame_id = "vision_detection";
    annotated_image_pub_->publish(*out_msg);
  }

  void enrollNearestFace(std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (!face_analyzer_.canExtractEmbeddings()) {
      response->success = false;
      response->message = "face recognizer model is not loaded";
      return;
    }
    if (last_faces_.empty() || last_faces_.front().embedding.empty()) {
      response->success = false;
      response->message = "no face embedding is available yet";
      return;
    }

    const std::string user_id = get_parameter("enroll_user_id").as_string();
    const std::string user_role = get_parameter("enroll_user_role").as_string();
    identity_store_.addOrUpdate(user_id, user_role, last_faces_.front().embedding);
    if (!identity_store_.save(identity_store_path_)) {
      response->success = false;
      response->message = "failed to save identity store";
      return;
    }

    response->success = true;
    response->message = "enrolled nearest face as " + user_id + " role=" + user_role;
  }

  std::string source_type_;
  int camera_index_{0};
  int camera_width_{1280};
  int camera_height_{720};
  double camera_fps_{15.0};
  int camera_probe_count_{8};
  std::string image_topic_;
  std::string video_path_;

  std::string yolo_model_path_;
  std::string face_detector_model_path_;
  std::string face_recognizer_model_path_;
  std::string emotion_model_path_;
  std::string gesture_model_path_;
  std::string identity_store_path_;

  float object_confidence_threshold_{0.35F};
  float object_nms_threshold_{0.45F};
  int object_input_width_{640};
  int object_input_height_{640};
  float gesture_confidence_threshold_{0.35F};
  float gesture_nms_threshold_{0.45F};
  std::vector<std::string> gesture_labels_;
  bool gesture_model_ready_{false};
  float emotion_smoothing_alpha_{0.25F};
  float emotion_min_face_area_ratio_{0.01F};
  float emotion_max_abs_yaw_deg_{35.0F};
  float emotion_max_abs_pitch_deg_{30.0F};
  float emotion_min_face_score_{0.85F};
  float emotion_ema_{0.0F};
  bool emotion_ema_initialized_{false};
  float face_score_threshold_{0.8F};
  float face_nms_threshold_{0.3F};
  int face_top_k_{5000};
  float identity_threshold_{0.38F};
  float near_person_min_area_ratio_{0.12F};
  std::vector<std::string> novelty_labels_;
  std::set<std::string> novelty_label_set_;
  bool publish_annotated_image_{true};
  bool use_advanced_emotion_{true};
  std::string advanced_emotion_topic_;
  double advanced_emotion_max_age_sec_{1.5};
  bool use_advanced_gestures_{true};
  std::string advanced_gesture_events_topic_;
  double advanced_gesture_max_age_sec_{1.5};

  std::string signals_topic_;
  std::string camera_list_topic_;
  std::string near_human_topic_;
  std::string emotion_topic_;
  std::string novelty_topic_;
  std::string face_orientation_topic_;
  std::string gesture_events_topic_;
  std::string annotated_image_topic_;

  ObjectDetector object_detector_;
  ObjectDetector gesture_detector_;
  FaceAnalyzer face_analyzer_;
  EmotionClassifier emotion_classifier_;
  IdentityStore identity_store_;
  std::vector<FaceObservation> last_faces_;
  std::vector<int> available_camera_indices_;
  std::mutex input_mutex_;
  std::mutex fusion_mutex_;
  msg::EmotionState last_advanced_emotion_;
  bool has_advanced_emotion_{false};
  rclcpp::Time last_advanced_emotion_received_;
  msg::GestureEvents last_advanced_gestures_;
  bool has_advanced_gestures_{false};
  rclcpp::Time last_advanced_gestures_received_;

  cv::VideoCapture capture_;
  rclcpp::TimerBase::SharedPtr capture_timer_;
  rclcpp::TimerBase::SharedPtr camera_list_timer_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<msg::EmotionState>::SharedPtr advanced_emotion_sub_;
  rclcpp::Subscription<msg::GestureEvents>::SharedPtr advanced_gestures_sub_;
  rclcpp::Publisher<msg::VisionSignals>::SharedPtr signals_pub_;
  rclcpp::Publisher<msg::CameraList>::SharedPtr camera_list_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr near_human_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr emotion_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr novelty_pub_;
  rclcpp::Publisher<msg::FaceOrientation>::SharedPtr face_orientation_pub_;
  rclcpp::Publisher<msg::GestureEvents>::SharedPtr gesture_events_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annotated_image_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr enroll_service_;
  rclcpp::Service<srv::SetInputSource>::SharedPtr set_input_service_;
};

}  // namespace vision_detection

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<vision_detection::VisionDetectionNode>());
  rclcpp::shutdown();
  return 0;
}
