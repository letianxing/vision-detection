"""Brief brightness pulses measured in captured images; raw evidence is preserved."""
import cv2
import numpy as np

class FlashDetector:
    def __init__(self):
        self.previous=None;self.pending=None;self.sequence=0
    def update(self,frame,stamp):
        cells=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(8,8)).astype(float)
        result=[]
        if self.previous is not None:
            delta=cells-self.previous
            if self.pending is None and np.mean(delta>45)>=.25:
                self.pending=(stamp,self.previous.copy())
            elif self.pending:
                start,baseline=self.pending
                if 20<=stamp-start<=250 and np.mean(abs(cells-baseline)<25)>=.75:
                    self.sequence+=1;result=[{'id':f'flash:{start}:{self.sequence}','kind':'flash_pulse','stamp_ms':start,'ended_ms':stamp,'evidence':'brightness_rise_and_return'}];self.pending=None
                elif stamp-start>250:self.pending=None
        self.previous=cells
        return result
