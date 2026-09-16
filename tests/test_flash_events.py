import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from flash_events import FlashDetector

class FlashTests(unittest.TestCase):
    def test_short_flash_requires_return_to_baseline(self):
        d=FlashDetector();frame=lambda level:np.full((80,80,3),level,np.uint8)
        self.assertEqual(d.update(frame(20),1000),[])
        self.assertEqual(d.update(frame(200),1050),[])
        result=d.update(frame(20),1100)
        self.assertEqual(len(result),1);self.assertEqual(result[0]['stamp_ms'],1050)
        d.update(frame(200),1200)
        self.assertEqual(d.update(frame(200),1600),[])
