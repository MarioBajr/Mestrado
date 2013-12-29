__author__ = 'Mario'


import sympy as sy
import numpy as np
import math

class Rectangle(object):

    def __init__(self, p1, p2):
        p1 = Point(p1.x, p1.y)
        p2 = Point(p2.x, p2.y)
        if (p1.x > p2.x) is True:
            self.p1, self.p2 = p2, p1
        elif (p1.x == p2.x) is (p1.y > p2.y) is True:
            self.p1, self.p2 = p2, p1
        else:
            self.p1, self.p2 = p1, p2

    def is_inside(self, r):
        return self.p1.x <= r.p1.x and self.p2.x >= r.p2.x and self.p1.y <= r.p1.y and self.p2.y >= r.p2.y

    def collide(self, r):
        ax1 = self.p1.x <= r.p1.x <= self.p2.x
        ax2 = self.p1.x <= r.p2.x <= self.p2.x
        ay1 = self.p1.y <= r.p1.y <= self.p2.y
        ay2 = self.p1.y <= r.p2.y <= self.p2.y

        bx1 = r.p1.x <= self.p1.x <= r.p2.x
        bx2 = r.p1.x <= self.p2.x <= r.p2.x
        by1 = r.p1.y <= self.p1.y <= r.p2.y
        by2 = r.p1.y <= self.p2.y <= r.p2.y

        return (ax1 or ax2 or bx1 or bx2) and (ay1 or ay2 or by1 or by2)

    def inflate(self, d):
        self.p1.x -= d
        self.p2.y -= d

        #sy.Point(self.p1.x-d, self.p1.y-d)
        self.p2.x += d
        self.p2.y += d

        #sy.Point(self.p2.x+d, self.p2.y+d)

    @staticmethod
    def from_segment(s):
        return Rectangle(s.p1, s.p2)


class Point(object):

    # def __init__(cls, *args, **kws):
    #     cls.segments = []
    #     return sy.Point.__init__(cls, *args, **kws)

    # def __new__(cls, *args, **kwargs):
    #     return sy.Point.__new__(cls, *args, **kwargs)

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.segments = []

    def add_segment(self, segment):
        self.segments.append(segment)

    # def remove_segment(self, segment):
    #     self.segments.remove(segment)

    def is_joint(self):
        return len(self.segments) > 1

    @staticmethod
    def min(*args, **kwargs):
        def min_tuple(a, b):
            if a.x < b.y:
                return a
            elif a.x == b.x and a.y < b.y:
                return a
            else:
                return b

        best_p = args[0]
        for p in args:
            best_p = min_tuple(p, best_p)
        return best_p

    @staticmethod
    def max(*args, **kwargs):
        def max_tuple(a, b):
            if a.x < b.x:
                return b
            elif a.x == b.x and a.y < b.y:
                return b
            else:
                return a

        best_p = args[0]
        for p in args:
            best_p = max_tuple(p, best_p)
        return best_p

    def dot(self, p2):
        x1, y1 = self.x, self.y
        x2, y2 = p2.x, p2.y
        return x1*x2 + y1*y2

    @staticmethod
    def distance(p1, p2):
        x = p1.x - p2.x
        y = p1.y - p2.y
        return math.sqrt(x*x + y*y)

    def __add__(self, other):
        return Point(self.x+other.x, self.y+other.y)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, factor):
        return Point(self.x*factor, self.y*factor)

    def __div__(self, divisor):
        return Point(self.x/divisor, self.y/divisor)

    __truediv__ = __div__

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __abs__(self):
        origin = Point(0, 0)
        return Point.distance(origin, self)

    def __str__(self):
        return type(self).__name__ + "(" + `self.x` + ", " + `self.y` + ")"


class Segment(object):

    # def __init__(cls, *args, **kws):
    #     cls.pt1 = args[0]
    #     cls.pt2 = args[1]
    #     return sy.Point.__init__(cls, *args, **kws)
    #
    # def __new__(cls, p1, p2, *args, **kwargs):
    #     s = sy.Segment.__new__(cls, p1, p2, **kwargs)
    #     cls.pt1 = p1
    #     cls.pt2 = p2
    #     p1.add_segment(s)
    #     p2.add_segment(s)
    #     return s

    def __init__(self, p1, p2):
        self._p1 = p1
        self._p2 = p2
        p1.segments.append(self)
        p2.segments.append(self)

    def __del__(self):
        self._p1.segments.remove(self)
        self._p2.segments.remove(self)

    @property
    def p1(self):
        return self._p1

    @p1.setter
    def p1(self, value):
        self._p1.segments.remove(self)
        value.segments.append(self)
        self._p1 = value

    @property
    def p2(self):
        return self._p2

    @p2.setter
    def p2(self, value):
        self._p2 = value

    def is_connected_with(self, s):
        c1 = self.connections
        c2 = self.connections
        return s in self.connections or len(c1.intersection(c2)) > 1

    def distance(self, p):
        x1, y1 = self.p1.x, self.p1.y
        x2, y2 = self.p2.x, self.p2.y
        x3, y3 = p.x, p.y

        px = x2-x1
        py = y2-y1

        d = px*px + py*py

        u = ((x3 - x1) * px + (y3 - y1) * py) / float(d)

        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        x = x1 + u * px
        y = y1 + u * py

        dx = x - x3
        dy = y - y3

        dist = math.sqrt(dx*dx + dy*dy)
        return dist

    def intersection(self, s):
        def perp(a):
            return Point(-a.y, a.x)

        a1 = self.p1
        a2 = self.p2
        b1 = self.p1
        b2 = self.p2
        da = a2-a1
        db = b2-b1
        dp = a1-b1
        dap = perp(da)
        denom = dap.dot(db)#np.dot( dap, db)
        num = dap.dot(dp)#np.dot( dap, dp )
        r = (num / denom)*db + b1
        return r

    @property
    def bounds(self):
        return Rectangle.from_segment(self)

    @property
    def connections(self):
        all_segments = set(self.p1.segments+self.p2.segments)
        return all_segments

    # @property
    # def pt1(self):
    #     self.pt1


    # @property
    # def pt2(self):
    #     self.pt2

    @staticmethod
    def union(s1, s2):
        p1 = Point.min(s1.p1, s1.p2, s2.p1, s2.p2)
        p2 = Point.max(s1.p1, s1.p2, s2.p1, s2.p2)
        print "UNION MIN", s1.p1, s1.p2, s2.p1, s2.p2, ">", p1
        print "UNION MAX", s1.p1, s1.p2, s2.p1, s2.p2, ">", p2
        return Segment(p1, p2)

    @staticmethod
    def angle_between(s1, s2):
        v1 = s1.p2 - s1.p1
        v2 = s2.p2 - s2.p1
        return math.acos(v1.dot(v2)/(abs(v1)*abs(v2)))

    def __str__(self):
        return type(self).__name__ + "(" + self.p1.__str__() + ", " + self.p2.__str__() + ")"

# class Point(object):
#
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#         self.connected = False
#
#     def to_numpy(self):
#         return np.array([self.x, self.y])
#
#     def from_numpy(self, a):
#         return Point(a[0], a[1])
#
#     @staticmethod
#     def zero():
#         return Point(0, 0)
#
#     def distance(a, b):
#         x = a.x-b.x
#         y = a.y-b.y
#         return math.sqrt(x*x+y*y)
#
# class Segment(object):
#
#     def __init__(self, p1, p2):
#         self.p1 = p1
#         self.p2 = p2
#
#     def cartesian_representation(self):
#         a = math.atan2(self.p2.y-self.p1.y, self.p2.x-self.p1.x)
#         b = (self.p2.x*self.p1.y-self.p1.x*self.p2.y)/(self.p2.x-self.p1.x)
#         return a, b
#
#     @staticmethod
#     def __ccw(a, b, c):
#         return (c.y-a.y) * (b.x-a.x) > (b.y-a.y) * (c.x-a.x)
#
#     def intersect(l1, l2):
#         a = l1.p1, b = l1.p2, c = l2.p1, d = l2.p2
#         return Segment.__ccw(a, c, d) != Segment.__ccw(b, c, d) and Segment.__ccw(a, b, c) != Segment.__ccw(a, b, d)
#
#     def intersection(a, b):
#         def perp(a) :
#             b = np.empty_like(a)
#             b[0] = -a[1]
#             b[1] = a[0]
#             return b
#
#         a1 = a.p1.to_numpy()
#         a2 = a.p2.to_numpy()
#         b1 = b.p1.to_numpy()
#         b2 = b.p2.to_numpy()
#         da = a2-a1
#         db = b2-b1
#         dp = a1-b1
#         dap = perp(da)
#         denom = np.dot( dap, db)
#         num = np.dot( dap, dp )
#         r = (num / denom)*db + b1
#         return Point.from_numpy(r)