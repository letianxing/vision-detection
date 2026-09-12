import sys
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from local_vision_dashboard import LocalVisionRuntime


class FaceTrackingTest(unittest.TestCase):
    def runtime(self):
        runtime = LocalVisionRuntime.__new__(LocalVisionRuntime)
        runtime.face_tracks = {}
        runtime.next_face_track_id = 1
        return runtime

    def test_keeps_face_id_when_detection_order_changes(self):
        runtime = self.runtime()
        first = [{"box": [10, 10, 80, 80]}, {"box": [300, 10, 60, 60]}]
        runtime.assign_face_tracks(first)
        left_id = first[0]["track_id"]
        right_id = first[1]["track_id"]

        second = [{"box": [302, 12, 85, 85]}, {"box": [12, 11, 55, 55]}]
        runtime.assign_face_tracks(second)

        self.assertEqual(second[0]["track_id"], right_id)
        self.assertEqual(second[1]["track_id"], left_id)

    def test_far_face_creates_new_track(self):
        runtime = self.runtime()
        first = [{"box": [10, 10, 60, 60]}]
        runtime.assign_face_tracks(first)
        second = [{"box": [500, 300, 60, 60]}]
        runtime.assign_face_tracks(second)

        self.assertNotEqual(first[0]["track_id"], second[0]["track_id"])


if __name__ == "__main__":
    unittest.main()
