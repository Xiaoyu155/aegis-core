from depth_sorter import DepthSorter
from cat_fsm import CatFSM

class RenderLoop:
    def __init__(self,cats,capacity=64):
        if len(cats)>capacity or capacity<=0: raise ValueError('capacity')
        self.cats=cats
        self.sorter=DepthSorter()
        self.fsm=CatFSM()
        self.draw_buffer=[None]*capacity
        self.history=[]
        self.frames=0

    def tick(self,dt):
        """Reuse buffer; Python is not a zero-allocation renderer."""
        if dt<0: raise ValueError('dt')
        for cat in self.cats: self.fsm.update(cat,dt)
        ordered=self.sorter.order(self.cats)
        for i,cat in enumerate(ordered): self.draw_buffer[i]=cat.ident
        for i in range(len(ordered),len(self.draw_buffer)): self.draw_buffer[i]=None
        self.frames+=1
        self.history.append(tuple(self.draw_buffer))  # deliberately planted unbounded history
        return len(ordered)
