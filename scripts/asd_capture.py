"""Bounded RAM-only tracked face windows. Never opens a camera or writes media."""
import base64
from collections import deque
import threading
import cv2
import numpy as np


class FaceWindows:
    def __init__(self):
        self.lock = threading.Lock()
        self.rows = {}

    def clear(self):
        with self.lock:
            self.rows.clear()

    def append(self, stamp, frame, faces, detection_stamp):
        if not 0 <= stamp-detection_stamp <= 300:
            return
        crops = []
        for person, track, box in faces[:4]:
            x,y,w,h = map(float, box)
            if not np.isfinite([x,y,w,h]).all() or min(w,h)<32:
                continue
            # Padded 2x face square; upstream then takes the central 112 pixels.
            size = int(2*max(w,h))
            if size > 2*max(frame.shape[:2]):
                continue
            left,top = int(x+w/2-size/2),int(y+h/2-size/2)
            x0,y0,x1,y1 = max(0,left),max(0,top),min(frame.shape[1],left+size),min(frame.shape[0],top+size)
            if x1<=x0 or y1<=y0:
                continue
            crop = np.zeros((size,size,3),np.uint8)
            crop[y0-top:y1-top,x0-left:x1-left] = frame[y0:y1,x0:x1]
            gray = cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(224,224))
            crops.append((person,track,gray))
        with self.lock:
            self.rows = {k:v for k,v in self.rows.items() if stamp-v[-1][0]<2000}
            for person,track,crop in crops:
                if person not in self.rows and len(self.rows)>=8:
                    continue
                rows = self.rows.setdefault(person,deque(maxlen=75))
                if rows and (rows[-1][1]!=track or stamp<=rows[-1][0]):
                    rows.clear()
                rows.append((stamp,track,crop))

    def window(self, now, person=''):
        with self.lock:
            available = sorted(k for k,v in self.rows.items() if 0<=now-v[-1][0]<500)
            if not person:
                return {'people': available}
            rows = list(self.rows.get(person,()))
        if not rows or not 0<=now-rows[-1][0]<500:
            return {'available':False,'reason':'no_fresh_face'}
        end = rows[-1][0]
        stamps = np.array([r[0] for r in rows])
        targets = np.arange(end-1000,end,40)
        indices = [int(np.argmin(abs(stamps-t))) for t in targets]
        errors = [abs(int(stamps[i])-int(t)) for i,t in zip(indices,targets)]
        if max(errors)>50 or len(set(indices))<15:
            return {'available':False,'reason':'insufficient_video_coverage'}
        frames = np.stack([rows[i][2] for i in indices])
        return {'available':True,'person_id':person,'visual_track_id':rows[-1][1],
                'ended_ms':end,'started_ms':end-1000,'fps':25,'frame_count':25,
                'unique_frames':len(set(indices)),'max_resample_error_ms':max(errors),
                'timestamp_source':'host_frame_receive',
                'gray224_base64':base64.b64encode(frames.tobytes()).decode()}
