"""Model-based mouth geometry; motion is evidence, not a speech classifier."""
from collections import deque
import time
import cv2
import numpy as np


def mouth_ratio(points):
    points=np.asarray(points,dtype=np.float64)
    width=np.linalg.norm(points[61,:2]-points[291,:2])
    if not np.isfinite(points).all() or width<5:
        return None
    return float(np.linalg.norm(points[13,:2]-points[14,:2])/width)


class MouthHistory:
    def __init__(self):
        self.samples={}

    def update(self,key,stamp,ratio):
        self.samples={k:v for k,v in self.samples.items() if stamp-v[-1][0]<1500}
        if ratio is None:
            self.samples.pop(key,None)
            return False,0.,False
        history=self.samples.setdefault(key,deque(maxlen=20))
        if history and (stamp-history[-1][0]>250 or stamp<=history[-1][0]):
            history.clear()
        history.append((stamp,ratio))
        while history and stamp-history[0][0]>450:history.popleft()
        ready=len(history)>=3 and stamp-history[0][0]>=100
        values=np.array([v for _,v in history])
        changes=np.abs(np.diff(values))
        score=float(np.ptp(values))
        moving=ready and score>=.045 and np.count_nonzero(changes>=.012)>=2 and changes[-1]>=.008
        return bool(moving),score,ready


class LipLandmarker:
    def __init__(self,path):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
        self.mp=mp
        self.model=FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),running_mode=RunningMode.VIDEO,
            num_faces=4,min_face_detection_confidence=.65,min_face_presence_confidence=.7,min_tracking_confidence=.7))
        self.history=MouthHistory();self.last_stamp=0

    def detect(self,frame,faces):
        stamp=max(self.last_stamp+1,int(time.monotonic()*1000));self.last_stamp=stamp
        result=self.model.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),stamp)
        height,width=frame.shape[:2]
        meshes=[np.array([[p.x*width,p.y*height] for p in mesh]) for mesh in result.face_landmarks]
        used=set()
        for face in faces:
            x,y,w,h=face['box'];orient=face['orientation'];points=None
            matches=[i for i,m in enumerate(meshes) if i not in used and x<=m[1,0]<=x+w and y<=m[1,1]<=y+h]
            valid=(len(matches)==1 and face['score']>=.85 and min(w,h)>=64 and orient.get('valid')
                   and abs(orient.get('yaw_deg',90))<=30 and abs(orient.get('pitch_deg',90))<=25)
            if valid:
                index=matches[0];used.add(index);points=meshes[index]
            ratio=mouth_ratio(points) if points is not None else None
            moving,score,ready=self.history.update(str(face['track_id']),stamp,ratio)
            face.update(mouth_open_ratio=ratio or 0.,lip_motion=moving,lip_motion_valid=bool(valid and ratio is not None and ready),
                        mouth_motion_score=score,mouth_roi_features=[],lip_backend='mediapipe_face_landmarker',
                        lip_reason='tracked' if valid and ready else 'warming_up' if valid else 'face_landmarks_unreliable')
        return faces

    def close(self):
        self.model.close()
