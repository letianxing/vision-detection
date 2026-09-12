from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("vision_detection")
    config_path = PathJoinSubstitution([package_share, "config", "vision_detection.yaml"])

    source_type = LaunchConfiguration("source_type")
    camera_index = LaunchConfiguration("camera_index")
    camera_horizontal_fov_deg = LaunchConfiguration("camera_horizontal_fov_deg")
    camera_vertical_fov_deg = LaunchConfiguration("camera_vertical_fov_deg")
    image_topic = LaunchConfiguration("image_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    depth_scale = LaunchConfiguration("depth_scale")
    depth_source = LaunchConfiguration("depth_source")
    publish_annotated_image = LaunchConfiguration("publish_annotated_image")
    advanced_perception = LaunchConfiguration("advanced_perception")
    advanced_image_topic = LaunchConfiguration("advanced_image_topic")
    advanced_emotion_topic = LaunchConfiguration("advanced_emotion_topic")
    advanced_gesture_events_topic = LaunchConfiguration("advanced_gesture_events_topic")
    advanced_emotion_backend = LaunchConfiguration("advanced_emotion_backend")
    dashboard = LaunchConfiguration("dashboard")
    dashboard_host = LaunchConfiguration("dashboard_host")
    dashboard_port = LaunchConfiguration("dashboard_port")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    face_detector_model_path = LaunchConfiguration("face_detector_model_path")
    face_recognizer_model_path = LaunchConfiguration("face_recognizer_model_path")
    emotion_model_path = LaunchConfiguration("emotion_model_path")
    use_builtin_emotion = LaunchConfiguration("use_builtin_emotion")
    gesture_model_path = LaunchConfiguration("gesture_model_path")
    identity_store_path = LaunchConfiguration("identity_store_path")
    people_topic = LaunchConfiguration("people_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument("source_type", default_value="camera"),
            DeclareLaunchArgument("camera_index", default_value="0"),
            DeclareLaunchArgument("camera_horizontal_fov_deg", default_value="70.0"),
            DeclareLaunchArgument("camera_vertical_fov_deg", default_value="43.0"),
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("depth_topic", default_value=""),
            DeclareLaunchArgument("depth_scale", default_value="0.001"),
            DeclareLaunchArgument("depth_source", default_value="ros_depth_topic"),
            DeclareLaunchArgument("publish_annotated_image", default_value="true"),
            DeclareLaunchArgument("advanced_perception", default_value="true"),
            DeclareLaunchArgument("advanced_image_topic", default_value="/vision/annotated_image"),
            DeclareLaunchArgument("advanced_emotion_topic", default_value="/vision/advanced_emotion"),
            DeclareLaunchArgument(
                "advanced_gesture_events_topic",
                default_value="/vision/gesture_events_advanced",
            ),
            DeclareLaunchArgument("advanced_emotion_backend", default_value="pyfeat"),
            DeclareLaunchArgument("dashboard", default_value="true"),
            DeclareLaunchArgument("dashboard_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("dashboard_port", default_value="8080"),
            DeclareLaunchArgument(
                "yolo_model_path",
                default_value=PathJoinSubstitution([package_share, "models", "yolov8n.onnx"]),
            ),
            DeclareLaunchArgument(
                "face_detector_model_path",
                default_value=PathJoinSubstitution(
                    [package_share, "models", "face_detection_yunet_2023mar.onnx"]
                ),
            ),
            DeclareLaunchArgument(
                "face_recognizer_model_path",
                default_value=PathJoinSubstitution(
                    [package_share, "models", "face_recognition_sface_2021dec.onnx"]
                ),
            ),
            DeclareLaunchArgument(
                "emotion_model_path",
                default_value=PathJoinSubstitution(
                    [package_share, "models", "emotion-ferplus-12-int8.onnx"]
                ),
            ),
            DeclareLaunchArgument("use_builtin_emotion", default_value="false"),
            DeclareLaunchArgument("gesture_model_path", default_value=""),
            DeclareLaunchArgument(
                "identity_store_path",
                default_value=PathJoinSubstitution([package_share, "config", "identities.yml"]),
            ),
            DeclareLaunchArgument("people_topic", default_value="/vision/people"),
            Node(
                package="vision_detection",
                executable="vision_detection_node",
                name="vision_detection",
                output="screen",
                parameters=[
                    config_path,
                    {
                        "source_type": source_type,
                        "camera_index": ParameterValue(camera_index, value_type=int),
                        "camera_horizontal_fov_deg": ParameterValue(
                            camera_horizontal_fov_deg, value_type=float
                        ),
                        "camera_vertical_fov_deg": ParameterValue(
                            camera_vertical_fov_deg, value_type=float
                        ),
                        "image_topic": image_topic,
                        "depth_topic": depth_topic,
                        "depth_scale": ParameterValue(depth_scale, value_type=float),
                        "depth_source": depth_source,
                        "publish_annotated_image": ParameterValue(
                            publish_annotated_image, value_type=bool
                        ),
                        "yolo_model_path": yolo_model_path,
                        "face_detector_model_path": face_detector_model_path,
                        "face_recognizer_model_path": face_recognizer_model_path,
                        "emotion_model_path": emotion_model_path,
                        "use_builtin_emotion": ParameterValue(
                            use_builtin_emotion, value_type=bool
                        ),
                        "gesture_model_path": gesture_model_path,
                        "identity_store_path": identity_store_path,
                        "use_advanced_emotion": ParameterValue(
                            advanced_perception, value_type=bool
                        ),
                        "advanced_emotion_topic": advanced_emotion_topic,
                        "use_advanced_gestures": ParameterValue(
                            advanced_perception, value_type=bool
                        ),
                        "advanced_gesture_events_topic": advanced_gesture_events_topic,
                        "people_topic": people_topic,
                    },
                ],
            ),
            Node(
                package="vision_detection",
                executable="advanced_perception_node.py",
                name="vision_advanced_perception",
                output="screen",
                condition=IfCondition(advanced_perception),
                parameters=[
                    {
                        "image_topic": advanced_image_topic,
                        "emotion_topic": advanced_emotion_topic,
                        "gesture_topic": advanced_gesture_events_topic,
                        "emotion_backend": advanced_emotion_backend,
                        "gesture_interval_sec": 0.12,
                    }
                ],
            ),
            Node(
                package="vision_detection",
                executable="vision_dashboard.py",
                name="vision_dashboard",
                output="screen",
                condition=IfCondition(dashboard),
                arguments=["--host", dashboard_host, "--port", dashboard_port],
            ),
        ]
    )
