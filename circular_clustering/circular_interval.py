import numpy as np
import matplotlib


class CircularInterval:
    a = None
    b = None
    sign = 1.


    def __init__(self, a, b, gamma=None, sign=1.):
        self.a = a
        self.b = b
        if gamma:
            self.sign = self.get_sign(a,gamma)
        else:
            self.sign = sign

    def get_sign(self, a,b):
        za = [np.cos(a), np.sin(a)]
        zb = [np.cos(b), np.sin(b)] 

        return np.sign(np.cross(za,zb))
    
    def intervals_intersect(self, circularInterval):
        a = circularInterval.a
        b = circularInterval.b
        return self.point_in_circular_interval(a) or self.point_in_circular_interval(b)

    
    def point_in_circular_interval(self, alpha):
        sign_a = self.get_sign(self.a, alpha)
        sign_b = self.get_sign(self.b, alpha)
        if self.sign == sign_a and self.sign != sign_b:
            return True
        else:
            return False
        
    def draw(self,ax):
        wedge = matplotlib.patches.Wedge((0,0), 1, 360*(1/(2*np.pi))*self.a,  360*(1/(2*np.pi))*self.b, alpha=0.5)
        ax.add_artist(wedge)
    
   



