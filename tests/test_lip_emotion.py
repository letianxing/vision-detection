import sys
from pathlib import Path
import unittest
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lip_landmarks import mouth_ratio, MouthHistory
from local_vision_dashboard import LocalVisionRuntime


class LipEmotionTests(unittest.TestCase):
    def test_geometry_invariant_to_translation_scale_and_roll(self):
        points=np.zeros((478,2));points[61]=[10,10];points[291]=[50,10];points[13]=[30,8];points[14]=[30,12]
        angle=.5;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        self.assertAlmostEqual(mouth_ratio(points),.1)
        self.assertAlmostEqual(mouth_ratio(points@rotation*2+50),.1)

    def test_static_open_mouth_and_noise_do_not_count_as_motion(self):
        history=MouthHistory()
        for i in range(10):
            moving,_,_=history.update('a',i*50,.3+(i%2)*.005)
            self.assertFalse(moving)
        self.assertFalse(history.update('a',500,.4)[0])
        self.assertFalse(history.update('a',900,.2)[0])
        self.assertFalse(history.update('other',1000,.4)[0])
        self.assertFalse(history.update('a',1000,None)[2])

    def test_repeated_open_close_is_motion_after_warmup(self):
        history=MouthHistory()
        self.assertFalse(history.update('a',0,.05)[0])
        self.assertFalse(history.update('a',50,.15)[0])
        self.assertTrue(history.update('a',100,.04)[0])

    def test_ferplus_uses_pixel_scale_and_rejects_ambiguous_logits(self):
        runtime=LocalVisionRuntime.__new__(LocalVisionRuntime)
        runtime.pyfeat_ready=False;runtime.emotion_backend='ferplus';runtime.emotion_net=Mock()
        runtime.emotion_net.forward.return_value=np.array([[0.,5.,0.,0.,0.,0.,0.,0.]])
        result=runtime.emotion_result(np.full((80,80,3),128,np.uint8),[0,0,80,80])
        blob=runtime.emotion_net.setInput.call_args.args[0]
        self.assertEqual(blob.shape,(1,1,64,64));self.assertEqual(float(blob.mean()),128.)
        self.assertTrue(result['valid']);self.assertEqual(result['label'],'happiness')
        runtime.emotion_net.forward.return_value=np.zeros((1,8))
        self.assertFalse(runtime.emotion_result(np.full((80,80,3),128,np.uint8),[0,0,80,80])['valid'])
