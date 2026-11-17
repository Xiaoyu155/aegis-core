from entities import Cat

class DepthSorter:
    def __init__(self, horizon=100.0):
        if horizon <= 0: raise ValueError('invalid horizon')
        self.horizon=horizon

    def scale(self, cat: Cat) -> float:
        """Bounded perspective scale for near/far cats."""
        return min(2.0,max(.5,1+cat.foot_y/self.horizon))

    def key(self, cat: Cat):
        """Foot-anchor depth, then stable id; never scaled sprite-body center."""
        return cat.container, cat.body_y*self.scale(cat), cat.ident

    def order(self, cats):
        return sorted(cats,key=self.key)
