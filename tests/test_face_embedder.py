"""Backend selection must be explicit about what is loaded and what it costs."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import face_embedder  # noqa: E402


def frame():
    return (np.random.rand(480, 640, 3) * 255).astype('uint8')


def yunet_row():
    # box then five landmarks: right eye, left eye, nose, right mouth, left mouth
    return np.array([200, 150, 120, 140, 230, 190, 290, 190, 260, 220, 235, 255, 285, 255, 0.99],
                    dtype=np.float32)


class SelectionTests(unittest.TestCase):
    def test_the_default_is_the_licence_free_backend(self):
        _, info = face_embedder.create(ROOT / 'models')
        self.assertEqual(info['backend'], 'sface')
        self.assertTrue(info['commercial_use'])
        self.assertNotIn('warning', info)

    def test_the_stronger_backend_declares_its_licence_restriction(self):
        _, info = face_embedder.create(ROOT / 'models', 'arcface')
        self.assertFalse(info['commercial_use'])
        self.assertIn('非商业', info['licence'])
        if info['available']:
            self.assertIn('warning', info)

    def test_neither_backend_claims_to_be_calibrated(self):
        for backend in ('sface', 'arcface'):
            _, info = face_embedder.create(ROOT / 'models', backend)
            self.assertFalse(info['calibrated'])
            self.assertIn('未在真人数据上标定', info['note'])

    def test_an_unknown_backend_is_refused_rather_than_substituted(self):
        embedder, info = face_embedder.create(ROOT / 'models', 'magic')
        self.assertIsNone(embedder)
        self.assertFalse(info['available'])

    def test_a_missing_model_is_reported_not_silently_swapped(self):
        embedder, info = face_embedder.create(ROOT / 'nonexistent-models', 'arcface')
        self.assertIsNone(embedder)
        self.assertEqual(info['backend'], 'arcface')
        self.assertIn('缺少模型文件', info['reason'])


class EmbeddingTests(unittest.TestCase):
    def build(self, backend):
        embedder, info = face_embedder.create(ROOT / 'models', backend)
        if embedder is None:
            self.skipTest(f"{backend} unavailable: {info.get('reason')}")
        return embedder, info

    def test_each_backend_returns_a_unit_vector_of_its_declared_size(self):
        for backend in ('sface', 'arcface'):
            embedder, info = self.build(backend)
            vector = embedder.embed(frame(), yunet_row())
            self.assertIsNotNone(vector, backend)
            self.assertEqual(len(vector), info['dimension'], backend)
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=4)

    def test_alignment_falls_back_to_the_box_when_landmarks_are_missing(self):
        embedder, _ = self.build('arcface')
        short = np.array([200, 150, 120, 140], dtype=np.float32)
        vector = embedder.embed(frame(), short)
        self.assertIsNotNone(vector)
        self.assertEqual(len(vector), 512)

    def test_an_off_screen_box_returns_nothing_instead_of_raising(self):
        embedder, _ = self.build('arcface')
        self.assertIsNone(embedder.align(frame(), np.array([900, 900, 50, 50], dtype=np.float32)))


class QualityTests(unittest.TestCase):
    def good_row(self):
        return yunet_row()

    def test_a_good_frontal_face_passes_and_scores_high(self):
        result = face_embedder.face_quality(frame(), self.good_row(), {'yaw': 5, 'pitch': 3})
        self.assertTrue(result['usable'])
        self.assertEqual(result['reasons'], [])
        self.assertGreater(result['quality'], 0.5)

    def test_small_turned_and_weakly_detected_faces_are_refused_with_a_reason(self):
        small = np.array([200, 150, 30, 30] + [0] * 10 + [0.99], dtype=np.float32)
        self.assertEqual(face_embedder.face_quality(frame(), small)['reasons'], ['face_too_small'])
        turned = face_embedder.face_quality(frame(), self.good_row(), {'yaw': 70, 'pitch': 3})
        self.assertIn('not_frontal_enough', turned['reasons'])
        self.assertEqual(turned['quality'], 0.0)
        weak = self.good_row().copy()
        weak[14] = 0.5
        self.assertIn('weak_detection', face_embedder.face_quality(frame(), weak)['reasons'])

    def test_a_flat_crop_is_reported_as_blurred(self):
        flat = np.zeros((480, 640, 3), dtype='uint8')
        self.assertIn('blurred', face_embedder.face_quality(flat, self.good_row())['reasons'])


class TemplateTests(unittest.TestCase):
    def test_the_template_is_the_mean_of_recent_good_frames(self):
        tracker = face_embedder.TemplateTracker(frames=5)
        row = yunet_row()
        base = np.zeros(128, dtype=np.float32)
        base[0] = 1.0
        np.random.seed(7)
        template, single = None, None
        for index in range(6):
            noisy = base + np.random.randn(128).astype('float32') * 0.05
            if single is None:
                single = noisy / np.linalg.norm(noisy)
            template = tracker.update(row, noisy, 1000 + index * 50)
        self.assertEqual(tracker.depth(row, 1250), 5)
        self.assertAlmostEqual(float(np.linalg.norm(template)), 1.0, places=4)
        # The whole point: averaging is closer to the true direction than any one
        # frame, which is what buys accuracy back from a weaker backend.
        self.assertGreater(float(template @ base), float(single @ base))

    def test_a_rejected_frame_does_not_change_the_template(self):
        tracker = face_embedder.TemplateTracker()
        row = yunet_row()
        base = np.zeros(128, dtype=np.float32)
        base[0] = 1.0
        tracker.update(row, base, 1000)
        before = tracker.update(row, None, 1050)
        self.assertEqual(tracker.depth(row, 1050), 1)
        self.assertAlmostEqual(float(before @ base), 1.0, places=4)

    def test_nothing_yet_means_no_template_rather_than_a_guess(self):
        tracker = face_embedder.TemplateTracker()
        self.assertIsNone(tracker.update(yunet_row(), None, 1000))

    def test_a_face_that_left_long_ago_is_forgotten(self):
        tracker = face_embedder.TemplateTracker()
        row = yunet_row()
        tracker.update(row, np.ones(128, dtype=np.float32), 1000)
        self.assertEqual(tracker.depth(row, 9000), 0)
        self.assertIsNone(tracker.update(row, None, 9000))


if __name__ == '__main__':
    unittest.main()
