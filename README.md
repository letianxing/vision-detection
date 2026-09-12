# vision-detection

ROS2 C++ vision node for mixed camera inputs:

- local cameras on macOS/Linux through OpenCV `VideoCapture`
- ROS2 `sensor_msgs/Image` topics through `cv_bridge`
- open ONNX models loaded with OpenCV DNN

The node publishes compact perception signals that downstream ROS2 nodes can consume:

- `near_human_present`: whether a nearby human is visible
- `v_user_raw`: emotion score for the nearest detected face in `[-1, 1]`
- `novelty`: whether selected object categories such as `cat`, `dog`, or `bottle` are visible
- `face_orient`: estimated yaw/pitch/roll for the nearest face
- `gestures`: hand gesture events from the advanced MediaPipe node when enabled, otherwise an optional gesture ONNX model or identity events such as `owner_detected`
- `/vision/people`: per-person observations for attention and ROS4HRI bridging, including `person_id`, transient `face_id/body_id`, estimated azimuth/elevation, gaze, body-facing, proxemics, engagement, emotion, and gesture fields

## Models

Download the default models:

```bash
./scripts/download_models.sh
```

Run this before `colcon build` so the models are installed into the package share directory used by the launch file.

Default model sources:

- YOLOv8n ONNX object detector: <https://huggingface.co/inference4j/yolov8n>
- OpenCV YuNet face detector: <https://huggingface.co/opencv/face_detection_yunet>
- OpenCV SFace face embedding model: <https://huggingface.co/opencv/face_recognition_sface>
- FER+ emotion classifier: <https://huggingface.co/onnxmodelzoo/emotion-ferplus-12-int8>
- MediaPipe hand landmarker: <https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker>

The default YOLOv8n model is AGPL-3.0 licensed. Check model licenses before product use.

Optional hand gesture models can be plugged in with:

```bash
ros2 launch vision_detection vision_detection.launch.py \
  gesture_model_path:=/path/to/gesture_yolo.onnx
```

Set `gesture_labels` to match the model class order. Labels containing `wave`, `hello`, or `hi` publish `<role>_wave`; labels containing `invite`, `come`, or `call` publish `<role>_invite`; labels containing `reject`, `stop`, or `no` publish `<role>_reject`.

For local Mac debugging, `scripts/local_vision_dashboard.py` uses MediaPipe hand landmarks and temporal rules for:

- `hi`: lateral hand waving
- `invite`: finger curl/beckon motion in place
- `reject`: hand moves forward and briefly stops

Use Python 3.12 for the local dashboard because MediaPipe wheels may not support newer Python versions:

```bash
python3.12 -m venv .venv312
.venv312/bin/python -m pip install opencv-python aiohttp numpy mediapipe py-feat
brew install libomp ffmpeg
OPENCV_AVFOUNDATION_SKIP_AUTH=1 .venv312/bin/python scripts/local_vision_dashboard.py
```

The local dashboard defaults to `--emotion-backend pyfeat`, which uses py-feat `Detectorv2` and publishes py-feat `valence` as `v_user_raw` in `[-1, 1]`. Use `--emotion-backend ferplus` to return to the lightweight FER+ classifier.

For smoother local preview on macOS, run the dashboard with decoupled stream and inference rates:

```bash
OPENCV_AVFOUNDATION_SKIP_AUTH=1 .venv312/bin/python scripts/local_vision_dashboard.py \
  --host 127.0.0.1 \
  --port 8080 \
  --camera-index 0 \
  --emotion-backend pyfeat \
  --stream-fps 20 \
  --inference-fps 5 \
  --jpeg-quality 76 \
  --pyfeat-interval 1.2 \
  --camera-width 960 \
  --camera-height 540 \
  --yolo-size 640 \
  --object-interval 0.5
```

In ROS2 launch mode, `advanced_perception:=true` starts `scripts/advanced_perception_node.py`. It subscribes to `/vision/annotated_image`, publishes py-feat valence/arousal to `/vision/advanced_emotion`, publishes MediaPipe hand events to `/vision/gesture_events_advanced`, and the C++ node fuses those into `/vision/signals`.

By default, py-feat mode does not silently downgrade to FER+. If py-feat cannot load, emotion is marked invalid. This avoids hidden accuracy changes. To explicitly use the lighter built-in FER+ path, run the local dashboard with `--emotion-backend ferplus`, or launch ROS2 with `advanced_perception:=false use_builtin_emotion:=true`.

## Build

From a ROS2 workspace:

```bash
cd ~/ros2_ws/src
git clone <this-repo-url> vision-detection
cd ..
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select vision_detection
source install/setup.bash
```

OpenCV must include the `FaceDetectorYN` and `FaceRecognizerSF` APIs, available in modern OpenCV 4 builds.

## Run with local Mac camera

```bash
ros2 launch vision_detection vision_detection.launch.py source_type:=camera camera_index:=0
```

Open the realtime dashboard at:

```text
http://localhost:8080
```

## Run from a ROS2 image topic

```bash
ros2 launch vision_detection vision_detection.launch.py source_type:=ros_topic image_topic:=/camera/image_raw
```

### RealSense D435i

先由 `realsense-ros` 发布对齐 RGB-D，再启动视觉节点：

```bash
ros2 launch vision_detection vision_detection.launch.py \
  source_type:=ros_topic \
  image_topic:=/camera/camera/color/image_raw \
  depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  depth_source:=realsense_d435i
```

### RealSense D405

D405 是近距离深度相机，官方理想范围约 7cm-50cm，适合近距离 Person/proxemic 测试，不适合替代 D435i 做 3-8m 远距离测试。[D405 官方规格](https://www.realsenseai.com/products/stereo-depth-camera-d405/)

```bash
python3 -m pip install pyrealsense2
python3 scripts/local_vision_dashboard.py --source-type realsense --source-id "" \
  --host 127.0.0.1 --port 8080
```

### INDEMIND M1

M1 官方 SDK 支持 Linux/ROS，不作为 Mac 原生首测设备。Jetson/Linux 上启动厂商 ROS SDK 后：

```bash
ros2 launch vision_detection vision_detection.launch.py \
  source_type:=ros_topic \
  image_topic:=/indemind/left/image_raw \
  depth_topic:=/indemind/depth/image_raw \
  depth_source:=indemind_m1 \
  camera_horizontal_fov_deg:=120.0 \
  camera_vertical_fov_deg:=75.0
```

M1 SDK 版本间 topic 名可能不同，按实际 `ros2 topic list` remap。下游仍统一输出 `distance_m/distance_confidence/depth_source`，不改变 attention 接口。设备默认值记录在 `config/camera_profiles.json`。

To disable the Python advanced model node and use only the C++ FER+/YOLO path:

```bash
ros2 launch vision_detection vision_detection.launch.py advanced_perception:=false use_builtin_emotion:=true
```

Disable the dashboard if you only need topics:

```bash
ros2 launch vision_detection vision_detection.launch.py dashboard:=false
```

## Published Topics

- `/vision/signals` (`vision_detection/msg/VisionSignals`): aggregate signal message
- `/vision/people` (`vision_detection/msg/PeopleSignals`): per-person observations for `robot-attention-perception`
- `/vision/available_cameras` (`vision_detection/msg/CameraList`): local cameras detected when the service starts
- `/vision/near_human_present` (`std_msgs/msg/Bool`)
- `/vision/v_user_raw` (`std_msgs/msg/Float32`): nearest face emotion score in `[-1, 1]`
- `/vision/novelty_present` (`std_msgs/msg/Bool`)
- `/vision/face_orient` (`vision_detection/msg/FaceOrientation`)
- `/vision/gesture_events` (`vision_detection/msg/GestureEvents`)
- `/vision/annotated_image` (`sensor_msgs/msg/Image`, optional debug image)
- `/vision/advanced_emotion` (`vision_detection/msg/EmotionState`, consumed by the C++ node when `use_advanced_emotion` is true)
- `/vision/gesture_events_advanced` (`vision_detection/msg/GestureEvents`, consumed by the C++ node when `use_advanced_gestures` is true)

ROS4HRI bridge mapping:

- `/vision/people` should map to `/humans/faces/tracked`, `/humans/bodies/tracked`, `/humans/persons/tracked`, `/humans/candidate_matches`, and TF frames such as `face_<faceID>`, `gaze_<faceID>`, and `body_<bodyID>`.
- `face_id` and `body_id` are transient observation IDs. `person_id` is the longer-lived identity when recognition is available, otherwise a local session ID such as `vision_person_0`.
- `voice_id` is left empty by this visual node and should be filled by a ROS4HRI person manager or attention/person association layer after matching visual bearing with acoustic tracks.

Services:

- `/vision/set_input_source` (`vision_detection/srv/SetInputSource`): switch between `camera`, `ros_topic`, and `video_file`
- `/vision/enroll_nearest_face` (`std_srvs/srv/Trigger`): store the current nearest face as the configured user

## Dashboard

The dashboard is served by `scripts/vision_dashboard.py` and does not require rosbridge. It subscribes to the project topics, streams `/vision/annotated_image` as MJPEG, and sends topic values to the browser over WebSocket.

It provides:

- local camera selection from `/vision/available_cameras`
- ROS image topic input
- video file input
- realtime values for `near_human_present`, `v_user_raw`, `novelty`, `face_orient`, identity, and gesture events

Run it separately if the detection node is already running:

```bash
ros2 run vision_detection vision_dashboard.py --host 0.0.0.0 --port 8080
```

Inspect:

```bash
ros2 topic echo /vision/signals
```

## Performance Notes

The local dashboard originally ran camera capture, YOLO, YuNet face detection, MediaPipe hand landmarks, py-feat emotion, and JPEG encoding in one loop. py-feat is the heaviest step and can pause that loop long enough for the video stream to feel choppy.

The optimized local dashboard separates camera streaming from model inference:

- stream thread: reads camera frames, draws the latest known overlays, and publishes MJPEG
- inference thread: runs object, face, hand, and emotion models at a lower rate
- default local output is resized to `960x540` before encoding and inference
- py-feat valence/arousal is throttled by `--pyfeat-interval`
- YOLO still uses the same model and default `640` input size, but runs at a lower cadence through `--object-interval`
- py-feat mode does not load the FER+ ONNX model, so the two emotion models are not resident at the same time

Measured on the current MacBook Pro with `--emotion-backend pyfeat`:

- before optimization: about `12 FPS` MJPEG, about `240%` CPU
- after stream/inference split: about `18 FPS` MJPEG at `960x540`
- state/model updates: about `4-5 Hz`
- Python process memory: roughly `1.1-1.7 GB` RSS depending on py-feat state and cache behavior

If preview smoothness matters more than py-feat emotion quality, use:

```bash
.venv312/bin/python scripts/local_vision_dashboard.py --emotion-backend ferplus
```

The FER+ path is lighter, but less accurate for continuous `[-1, 1]` valence.

## Hardware And Deployment Sizing

Current tested machine:

- MacBook Pro `Mac16,8`
- Apple `M4 Pro`
- CPU: 12 cores
- GPU: 16-core integrated Apple GPU
- memory: `48 GB` unified memory

Apple Silicon does not expose NVIDIA-style dedicated VRAM. CPU and GPU share unified memory, so this Mac should be treated as a `48 GB unified memory` machine, not as a machine with a separate fixed-size graphics card.

Model file sizes in this repo are small:

- YuNet face detector: about `228 KB`
- MediaPipe hand landmarker: about `7.5 MB`
- YOLOv8n ONNX: about `12 MB`
- FER+ ONNX: about `18 MB`
- SFace recognizer: about `37 MB`

Runtime memory is much higher than model-file size because OpenCV, MediaPipe, PyTorch, py-feat, frame buffers, and JPEG buffers are loaded together. The local py-feat dashboard was measured around `1-2 GB` process RSS for one camera.

Recommended deployment tiers:

- lightweight C++ ONNX path only: `2-4 GB` GPU memory is enough for one camera
- py-feat/advanced emotion enabled: use at least `6-8 GB` GPU memory if running on CUDA, or enough CPU RAM if staying CPU-only
- multi-camera, TensorRT, or larger emotion/gesture models: prefer `12-16 GB` GPU memory
- edge robot deployment: Jetson Orin NX `16 GB` is a practical target; convert models to TensorRT for latency and thermals
- desktop/server deployment: RTX 4060 `8 GB` is workable for one stream, RTX 4060 Ti `16 GB` or RTX 4070-class `12 GB+` is the safer baseline

The current macOS local dashboard is not using NVIDIA CUDA and is mostly CPU-bound. Moving to an NVIDIA machine only helps substantially after switching the heavy paths to CUDA/TensorRT or GPU-backed runtimes.

## Identity Store

Known people are stored as local SFace embeddings in `config/identities.yml`.

By default, unknown faces publish:

- `nearest_user_id: stranger`
- `nearest_user_role: stranger`
- gesture event: `stranger_detected`

To enroll the current nearest face, set the parameters first and call the service:

```bash
ros2 param set /vision_detection enroll_user_id owner
ros2 param set /vision_detection enroll_user_role owner
ros2 service call /vision/enroll_nearest_face std_srvs/srv/Trigger
```

The identity store is local and simple by design. For production systems, encrypt it or move it to a dedicated private database.
