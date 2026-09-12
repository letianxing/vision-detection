#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--people-topic", default="/vision/people")
    parser.add_argument("--people-json-topic", default="/vision/people_json")
    parser.add_argument("--queue-size", type=int, default=10)
    args = parser.parse_args()

    try:
        import rclpy
        from hri_msgs.msg import IdsList, IdsMatch
        from std_msgs.msg import String
        from vision_detection.msg import PeopleSignals
    except Exception as exc:
        raise SystemExit(
            "ROS4HRI vision bridge requires a sourced ROS2 workspace with rclpy, "
            f"hri_msgs, std_msgs and vision_detection messages: {exc}"
        )

    rclpy.init()
    node = rclpy.create_node("vision_detection_ros4hri_bridge")
    faces_pub = node.create_publisher(IdsList, "/humans/faces/tracked", args.queue_size)
    bodies_pub = node.create_publisher(IdsList, "/humans/bodies/tracked", args.queue_size)
    persons_pub = node.create_publisher(IdsList, "/humans/persons/tracked", args.queue_size)
    matches_pub = node.create_publisher(IdsMatch, "/humans/candidate_matches", args.queue_size)
    people_json_pub = node.create_publisher(String, args.people_json_topic, args.queue_size)
    person_pubs: dict[str, object] = {}

    def string_pub(topic: str):
        if topic not in person_pubs:
            person_pubs[topic] = node.create_publisher(String, topic, args.queue_size)
        return person_pubs[topic]

    def callback(msg) -> None:
        people = list(msg.people)
        stamp = node.get_clock().now().to_msg()
        publish_ids(faces_pub, stamp, [person.face_id for person in people if person.face_id])
        publish_ids(bodies_pub, stamp, [person.body_id for person in people if person.body_id])
        publish_ids(persons_pub, stamp, [person.person_id for person in people if person.person_id])

        for person in people:
            if not person.person_id:
                continue
            publish_string(string_pub(f"/humans/persons/{person.person_id}/face_id"), person.face_id)
            publish_string(string_pub(f"/humans/persons/{person.person_id}/body_id"), person.body_id)
            publish_string(string_pub(f"/humans/persons/{person.person_id}/voice_id"), person.voice_id)
            publish_string(
                string_pub(f"/humans/persons/{person.person_id}/engagement_status"),
                person.engagement_status,
            )
            publish_string(string_pub(f"/humans/persons/{person.person_id}/proxemic_space"), person.proxemic_space)

            publish_match(matches_pub, stamp, person.person_id, "person", person.face_id, "face", person.face_confidence)
            publish_match(matches_pub, stamp, person.person_id, "person", person.body_id, "body", 0.7)
            publish_match(matches_pub, stamp, person.person_id, "person", person.voice_id, "voice", 0.5)

        payload = String()
        payload.data = json.dumps({"people": [person_to_dict(person) for person in people]}, ensure_ascii=False)
        people_json_pub.publish(payload)

    node.create_subscription(PeopleSignals, args.people_topic, callback, args.queue_size)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def publish_ids(pub, stamp, ids: list[str]) -> None:
    msg = pub.msg_type() if hasattr(pub, "msg_type") else None
    if msg is None:
        from hri_msgs.msg import IdsList

        msg = IdsList()
    msg.header.stamp = stamp
    msg.ids = list(dict.fromkeys(ids))
    pub.publish(msg)


def publish_string(pub, text: str) -> None:
    from std_msgs.msg import String

    msg = String()
    msg.data = text
    pub.publish(msg)


def publish_match(pub, stamp, id1: str, id1_type: str, id2: str, id2_type: str, confidence: float) -> None:
    if not id1 or not id2:
        return
    from hri_msgs.msg import IdsMatch

    match = IdsMatch()
    match.header.stamp = stamp
    for name, value in (
        ("id1", id1),
        ("id1_type", id1_type),
        ("id2", id2),
        ("id2_type", id2_type),
        ("confidence", float(confidence)),
    ):
        if hasattr(match, name):
            setattr(match, name, value)
    pub.publish(match)


def person_to_dict(person) -> dict[str, object]:
    return {
        "person_id": person.person_id,
        "role": person.role,
        "face_id": person.face_id,
        "body_id": person.body_id,
        "voice_id": person.voice_id,
        "azimuth_deg": person.azimuth_deg if person.has_azimuth else None,
        "elevation_deg": person.elevation_deg if person.has_elevation else None,
        "distance_m": person.distance_m if person.has_distance else None,
        "distance_confidence": float(getattr(person, "distance_confidence", 0.0)),
        "depth_source": str(getattr(person, "depth_source", "none")),
        "face_visible": bool(person.face_visible),
        "face_confidence": float(person.face_confidence),
        "mouth_open_ratio": float(getattr(person, "mouth_open_ratio", 0.0)),
        "lip_motion": bool(getattr(person, "lip_motion", False)),
        "mouth_roi_features": list(getattr(person, "mouth_roi_features", [])),
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


if __name__ == "__main__":
    main()
