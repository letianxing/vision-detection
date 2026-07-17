cmake_minimum_required(VERSION 3.8)
project(vision_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CameraInfo.msg"
  "msg/CameraList.msg"
  "msg/EmotionState.msg"
  "msg/FaceOrientation.msg"
  "msg/GestureObservation.msg"
  "msg/GestureEvents.msg"
  "msg/VisionSignals.msg"
  "srv/SetInputSource.srv"
  DEPENDENCIES std_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

include_directories(include)

add_library(vision_detection_core
  src/emotion_classifier.cpp
  src/face_analyzer.cpp
  src/identity_store.cpp
  src/object_detector.cpp
)
target_include_directories(vision_detection_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(vision_detection_core
  rclcpp
  sensor_msgs
  std_msgs
)
target_link_libraries(vision_detection_core PUBLIC ${OpenCV_LIBS})

add_executable(vision_detection_node src/vision_detection_node.cpp)
target_include_directories(vision_detection_node PRIVATE ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(vision_detection_node
  cv_bridge
  rclcpp
  sensor_msgs
  std_msgs
  std_srvs
)
target_link_libraries(vision_detection_node
  vision_detection_core
  ${OpenCV_LIBS}
  "${cpp_typesupport_target}"
)

install(PROGRAMS
  scripts/advanced_perception_node.py
  scripts/vision_dashboard.py
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  vision_detection_core
  vision_detection_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch scripts
  DESTINATION share/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

install(DIRECTORY web
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)

ament_export_include_directories(include)
ament_export_libraries(vision_detection_core)
ament_export_dependencies(rosidl_default_runtime)

ament_package()
