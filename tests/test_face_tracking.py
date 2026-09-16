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


class GuestContinuityTest(unittest.TestCase):
    """An unenrolled visitor is still the same visitor after the box is lost.

    Box tracking drops a person whenever they turn away or the detector misses
    two frames, and downstream every drop became a new stranger — thirteen of
    them in two minutes of one field session. The face template survives what the
    box does not.
    """

    def gallery(self):
        from local_vision_dashboard import GuestGallery
        return GuestGallery(threshold=0.45, ttl_s=600)

    def face(self, *values):
        import numpy as np
        return np.asarray(values, dtype=np.float32)

    def test_same_face_after_a_lost_track_keeps_its_id(self):
        gallery = self.gallery()
        first = gallery.resolve(self.face(1, 0, 0, 0), 1000)
        again = gallery.resolve(self.face(0.97, 0.1, 0.05, 0), 20000)

        self.assertEqual(first, again)

    def test_a_different_face_is_a_different_guest(self):
        gallery = self.gallery()
        first = gallery.resolve(self.face(1, 0, 0, 0), 1000)
        other = gallery.resolve(self.face(0, 1, 0, 0), 1100)

        self.assertNotEqual(first, other)

    def test_guests_are_forgotten_when_they_leave(self):
        gallery = self.gallery()
        first = gallery.resolve(self.face(1, 0, 0, 0), 1000)
        later = gallery.resolve(self.face(1, 0, 0, 0), 1000 + 601 * 1000)

        self.assertNotEqual(first, later)

    def test_person_id_prefers_the_name_then_the_guest_then_the_box(self):
        self.assertEqual(LocalVisionRuntime.stable_person_id("天行", "face_track_0007", "guest_0001"), "天行")
        self.assertEqual(LocalVisionRuntime.stable_person_id("stranger", "face_track_0007", "guest_0001"),
                         "vision_guest_0001")
        self.assertEqual(LocalVisionRuntime.stable_person_id("stranger", "face_track_0007", None),
                         "vision_face_track_0007")
