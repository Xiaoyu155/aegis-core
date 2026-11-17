"""CPU geometry oracle for a renderer; no assertion about GPU pixels."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Viewport:
    width: float
    height: float
    tile_width: float=64
    tile_height: float=32

    def __post_init__(self):
        if not all(math.isfinite(v) and v>0 for v in (self.width,self.height,self.tile_width,self.tile_height)):
            raise ValueError('positive finite viewport required')

    def project(self,x,y):
        return (self.width/2+(x-y)*self.tile_width/2,
                self.height/2+(x+y)*self.tile_height/2)

    def unproject(self,x,y):
        a=(x-self.width/2)/(self.tile_width/2)
        b=(y-self.height/2)/(self.tile_height/2)
        return (a+b)/2,(b-a)/2

    def visible(self,x,y,margin=0):
        return -margin<=x<=self.width+margin and -margin<=y<=self.height+margin

class FrameMetrics:
    """Constant-space online center-region variance, suitable for blank-frame gates."""
    def __init__(self):
        self.count=0
        self.mean=0.
        self.m2=0.

    def sample(self,value):
        if not math.isfinite(value): raise ValueError('nonfinite luminance')
        self.count+=1
        delta=value-self.mean
        self.mean+=delta/self.count
        self.m2+=delta*(value-self.mean)

    def healthy(self,threshold=18.5):
        return self.count>1 and self.m2/self.count>=threshold
