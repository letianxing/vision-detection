#!/usr/bin/env python3

import argparse
import asyncio
import json
import mimetypes
import threading
import time
from pathlib import Path

from aiohttp import WSMsgType, web
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from vision_detection.msg import CameraList, PeopleSignals, VisionSignals
from vision_detection.srv import SetInputSource

try:
    import cv2
    from cv_bridge import CvBridge
except Exception:  # pragma: no cover - depends on ROS/OpenCV runtime
    cv2 = None
    CvBridge = None


def stamp_to_float(stamp):
    return float(stamp.sec) + (float(stamp.nanosec) / 1_000_000_000.0)


def signals_to_dict(msg):
    return {
        "stamp": stamp_to_float(msg.header.stamp),
        "near_human_present": bool(msg.near_human_present),
        "nearest_user_id": msg.nearest_user_id,
        "nearest_user_role": msg.nearest_user_role,
        "v_user_raw": float(msg.v_user_raw),
        "v_user_raw_raw": float(msg.v_user_raw_raw),
        "emotion_label": msg.emotion_label,
        "emotion_confidence": float(msg.emotion_confidence),
        "emotion_arousal": float(msg.emotion_arousal),
        "emotion_valid": bool(msg.emotion_valid),
        "emotion_backend": msg.emotion_backend,
        "face_orient": {
            "valid": bool(msg.face_orient.valid),
            "yaw_deg": float(msg.face_orient.yaw_deg),
            "pitch_deg": float(msg.face_orient.pitch_deg),
            "roll_deg": float(msg.face_orient.roll_deg),
        },
        "novelty": bool(msg.novelty),
        "novelty_labels": list(msg.novelty_labels),
        "novelty_scores": [float(value) for value in msg.novelty_scores],
        "gestures": [
            {
                "user_id": gesture.user_id,
                "user_role": gesture.user_role,
                "gesture": gesture.gesture,
                "score": float(gesture.score),
            }
            for gesture in msg.gestures
        ],
        "person_count": int(msg.person_count),
        "nearest_person_score": float(msg.nearest_person_score),
        "nearest_person_bbox_area_ratio": float(msg.nearest_person_bbox_area_ratio),
    }


def people_to_list(msg):
    return [
        {
            "person_id": person.person_id,
            "role": person.role,
            "face_id": person.face_id,
            "body_id": person.body_id,
            "voice_id": person.voice_id,
            "has_azimuth": bool(person.has_azimuth),
            "azimuth_deg": float(person.azimuth_deg),
            "has_elevation": bool(person.has_elevation),
            "elevation_deg": float(person.elevation_deg),
            "has_distance": bool(person.has_distance),
            "distance_m": float(person.distance_m),
            "face_visible": bool(person.face_visible),
            "face_confidence": float(person.face_confidence),
            "gaze_score": float(person.gaze_score),
            "body_facing_score": float(person.body_facing_score),
            "bbox_area_ratio": float(person.bbox_area_ratio),
            "engagement_status": person.engagement_status,
            "proxemic_space": person.proxemic_space,
            "identity_confidence": float(person.identity_confidence),
            "emotion_valence": float(person.emotion_valence),
            "emotion_arousal": float(person.emotion_arousal),
            "emotion_valid": bool(person.emotion_valid),
            "emotion_label": person.emotion_label,
            "gesture": person.gesture,
            "gesture_score": float(person.gesture_score),
        }
        for person in msg.people
    ]


def camera_list_to_dict(msg):
    return {
        "stamp": stamp_to_float(msg.header.stamp),
        "cameras": [
            {
                "index": int(camera.index),
                "name": camera.name,
                "source_type": camera.source_type,
                "available": bool(camera.available),
            }
            for camera in msg.cameras
        ],
    }


class DashboardBridge(Node):
    def __init__(self):
        super().__init__("vision_dashboard_bridge")
        self._lock = threading.Lock()
        self._loop = None
        self._websockets = set()
        self._latest_state = None
        self._latest_people = []
        self._latest_cameras = {"stamp": 0.0, "cameras": []}
        self._latest_jpeg = None
        self._image_seq = 0
        self._bridge = CvBridge() if CvBridge is not None else None

        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(VisionSignals, "/vision/signals", self._on_signals, 10)
        self.create_subscription(PeopleSignals, "/vision/people", self._on_people, 10)
        self.create_subscription(CameraList, "/vision/available_cameras", self._on_cameras, camera_qos)
        self.create_subscription(Image, "/vision/annotated_image", self._on_image, 10)
        self._set_input_client = self.create_client(SetInputSource, "/vision/set_input_source")

    def attach_loop(self, loop):
        self._loop = loop

    def snapshot_state(self):
        with self._lock:
            if self._latest_state is None:
                return None
            state = dict(self._latest_state)
            state["people"] = list(self._latest_people)
            return state

    def snapshot_cameras(self):
        with self._lock:
            return self._latest_cameras

    def snapshot_jpeg(self):
        with self._lock:
            return self._image_seq, self._latest_jpeg

    async def add_ws(self, ws):
        self._websockets.add(ws)
        await ws.send_json(
            {
                "type": "snapshot",
                "state": self.snapshot_state(),
                "cameras": self.snapshot_cameras(),
            }
        )

    def remove_ws(self, ws):
        self._websockets.discard(ws)

    def call_set_input(self, data):
        if not self._set_input_client.wait_for_service(timeout_sec=2.0):
            return {"success": False, "message": "service /vision/set_input_source is unavailable"}

        request = SetInputSource.Request()
        request.source_type = str(data.get("source_type", "camera"))
        request.camera_index = int(data.get("camera_index", 0))
        request.image_topic = str(data.get("image_topic", "/camera/image_raw"))
        request.video_path = str(data.get("video_path", ""))

        future = self._set_input_client.call_async(request)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if future.done():
                response = future.result()
                return {"success": bool(response.success), "message": response.message}
            time.sleep(0.02)
        return {"success": False, "message": "timed out waiting for input switch"}

    def _on_signals(self, msg):
        payload = signals_to_dict(msg)
        with self._lock:
            payload["people"] = list(self._latest_people)
            self._latest_state = payload
        self._broadcast({"type": "state", "state": payload})

    def _on_people(self, msg):
        payload = people_to_list(msg)
        with self._lock:
            self._latest_people = payload
            if self._latest_state is not None:
                self._latest_state["people"] = payload
                state = dict(self._latest_state)
            else:
                state = {"stamp": stamp_to_float(msg.header.stamp), "people": payload}
        self._broadcast({"type": "state", "state": state})

    def _on_cameras(self, msg):
        payload = camera_list_to_dict(msg)
        with self._lock:
            self._latest_cameras = payload
        self._broadcast({"type": "cameras", "cameras": payload})

    def _on_image(self, msg):
        if self._bridge is None or cv2 is None:
            return
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                return
        except Exception as exc:  # pragma: no cover - depends on camera/runtime
            self.get_logger().warn(f"image conversion failed: {exc}")
            return
        with self._lock:
            self._latest_jpeg = bytes(encoded)
            self._image_seq += 1

    def _broadcast(self, payload):
        if self._loop is None or not self._websockets:
            return
        self._loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._broadcast_async(payload))
        )

    async def _broadcast_async(self, payload):
        stale = []
        for ws in self._websockets:
            if ws.closed:
                stale.append(ws)
                continue
            try:
                await ws.send_json(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.remove_ws(ws)


def static_response(path):
    content_type, _ = mimetypes.guess_type(path.name)
    return web.FileResponse(path, headers={"Cache-Control": "no-store"}, status=200, reason=None)


def make_app(bridge, web_dir):
    routes = web.RouteTableDef()

    @routes.get("/")
    async def index(_request):
        return static_response(web_dir / "index.html")

    @routes.get("/{name:index.js|styles.css}")
    async def asset(request):
        return static_response(web_dir / request.match_info["name"])

    @routes.get("/api/state")
    async def api_state(_request):
        return web.json_response({"state": bridge.snapshot_state()})

    @routes.get("/api/cameras")
    async def api_cameras(_request):
        return web.json_response(bridge.snapshot_cameras())

    @routes.post("/api/connect")
    async def api_connect(request):
        data = await request.json()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, bridge.call_set_input, data)
        return web.json_response(result)

    @routes.get("/ws")
    async def websocket(request):
        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(request)
        await bridge.add_ws(ws)
        async for message in ws:
            if message.type == WSMsgType.ERROR:
                break
        bridge.remove_ws(ws)
        return ws

    @routes.get("/stream.mjpg")
    async def stream(_request):
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-store",
            },
        )
        await response.prepare(_request)
        last_seq = -1
        while True:
            seq, jpeg = bridge.snapshot_jpeg()
            if jpeg is not None and seq != last_seq:
                last_seq = seq
                await response.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(0.02)

    app = web.Application()
    app.add_routes(routes)
    return app


async def run_dashboard(args, bridge):
    loop = asyncio.get_running_loop()
    bridge.attach_loop(loop)
    web_dir = Path(get_package_share_directory("vision_detection")) / "web"
    app = make_app(bridge, web_dir)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    bridge.get_logger().info(f"dashboard listening on http://{args.host}:{args.port}")
    await asyncio.Event().wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    rclpy.init()
    bridge = DashboardBridge()
    spin_thread = threading.Thread(target=rclpy.spin, args=(bridge,), daemon=True)
    spin_thread.start()
    try:
      asyncio.run(run_dashboard(args, bridge))
    finally:
      bridge.destroy_node()
      rclpy.shutdown()
      spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
