import sys,threading,time,unittest
from collections import deque
from pathlib import Path
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from local_vision_dashboard import LocalVisionRuntime

class RegistrationConsistencyTests(unittest.TestCase):
    def test_single_frame_cannot_register_and_inconsistent_faces_are_rejected(self):
        r=LocalVisionRuntime.__new__(LocalVisionRuntime);r.lock=threading.RLock()
        r.latest_frame=np.zeros((100,100,3),np.uint8);r.latest_faces=[{'track_id':'face1'}];r.identities={};r.save_identities=Mock();r.face_backend={'backend':'sface','threshold':0.38}
        stamp=int(time.time()*1000)
        r.registration_faces=deque([(stamp-100,'face1',[1.,0.],True)])
        self.assertFalse(r.enroll_nearest_face('owner',require_single=True,dry_run=True,started_ms=stamp-1000,ended_ms=stamp)['success'])
        r.registration_faces=deque([(stamp-i*50,'face1',[1.,0.] if i<3 else [-1.,0.],True) for i in range(6)])
        self.assertFalse(r.enroll_nearest_face('owner',require_single=True,dry_run=True,started_ms=stamp-1000,ended_ms=stamp)['success'])
        r.registration_faces=deque([(stamp-i*50,'face1',[1.,0.],True) for i in range(6)])
        self.assertTrue(r.enroll_nearest_face('owner',require_single=True,dry_run=True,started_ms=stamp-1000,ended_ms=stamp)['success'])
        self.assertEqual(r.identities,{})
        r.save_identities.assert_not_called()
