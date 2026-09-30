import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from asd_capture import FaceWindows


class FaceWindowTests(unittest.TestCase):
    def test_resampling_and_track_reset(self):
        buffer = FaceWindows()
        frame = np.full((160,160,3),127,np.uint8)
        for stamp in range(1000,2101,50):
            buffer.append(stamp,frame,[('a','face1',[40,40,80,80])],stamp)
        window = buffer.window(2150,'a')
        self.assertTrue(window['available'])
        self.assertEqual(window['frame_count'],25)
        self.assertGreaterEqual(window['unique_frames'],15)
        self.assertLessEqual(window['max_resample_error_ms'],50)
        buffer.append(2150,frame,[('a','face2',[40,40,80,80])],2150)
        self.assertFalse(buffer.window(2200,'a')['available'])
        buffer.clear()
        self.assertEqual(buffer.window(2200)['people'],[])

    def test_missing_frames_old_detection_and_stale_camera(self):
        buffer = FaceWindows()
        frame = np.zeros((160,160,3),np.uint8)
        for stamp in range(1000,2101,200):
            buffer.append(stamp,frame,[('a','f',[40,40,80,80])],stamp)
        self.assertFalse(buffer.window(2150,'a')['available'])
        buffer.clear()
        buffer.append(2000,frame,[('a','f',[40,40,80,80])],1000)
        self.assertEqual(buffer.window(2000)['people'],[])
