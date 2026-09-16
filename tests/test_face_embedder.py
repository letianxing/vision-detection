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


if __name__ == '__main__':
    unittest.main()
