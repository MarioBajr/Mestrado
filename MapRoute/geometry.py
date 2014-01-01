__author__ = 'Mario'

import numpy as np
import math


class Rectangle(object):

    def __init__(self, p1, p2):
        self.p1 = Point(min(p1.x, p2.x), min(p1.y, p2.y))
        self.p2 = Point(max(p1.x, p2.x), max(p1.y, p2.y))

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
        d = int(d)
        self.p1.x -= d
        self.p1.y -= d
        self.p2.x += d
        self.p2.y += d

    @staticmethod
    def from_segment(s):
        return Rectangle(s.p1, s.p2)

    def __str__(self):
        return type(self).__name__ + "(" + self.p1.__str__() + ", " + self.p2.__str__() + ")"


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.segments = set([])

    def is_joint(self):
        return len(self.segments) > 1

    def to_tuple(self):
        return int(self.x), int(self.y)

    @staticmethod
    def interpolate(p1, p2, f):
        x = p1.x + (p2.x - p1.x)*f
        y = p1.y + (p2.y - p1.y)*f
        return Point(x, y)

    @staticmethod
    def min(*args, **kwargs):
        def min_tuple(a, b):
            if a.x < b.x:
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
    def squared_distance(p1, p2):
        x = p1.x - p2.x
        y = p1.y - p2.y
        return x*x + y*y

    @staticmethod
    def distance(p1, p2):
        x = p1.x - p2.x
        y = p1.y - p2.y
        return math.sqrt(x*x + y*y)

    @staticmethod
    def is_collinear(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        a = math.acos(v1.dot(v2)/(abs(v1)*abs(v2)))
        return a == 0

    @staticmethod
    def zero():
        return Point(0, 0)

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

    def __init__(self, p1, p2):
        self._p1 = p1
        self._p2 = p2
        self._update()

    def unlink(self):
        self._p1.segments.remove(self)
        self._p2.segments.remove(self)

    @property
    def squared_length(self):
        return Point.squared_distance(self._p1, self._p2)


    @property
    def length(self):
        return Point.distance(self._p1, self._p2)

    @property
    def center(self):
        return (self.p2+self.p1)/2

    @property
    def p1(self):
        return self._p1

    @p1.setter
    def p1(self, value):
        self._p1.segments.remove(self)
        value.segments.add(self)
        self._p1 = value
        self._update()

    @property
    def p2(self):
        return self._p2

    @p2.setter
    def p2(self, value):
        self._p2.segments.remove(self)
        value.segments.add(self)
        self._p2 = value
        self._update()

    def _update(self):
        p1, p2 = self._p1, self._p2
        self._p1 = Point.min(p1, p2)
        self._p2 = Point.max(p1, p2)
        self._p1.segments.add(self)
        self._p2.segments.add(self)

    def is_connected_with(self, s):
        c1 = s.connections
        c2 = self.connections
        return s in self.connections or len(c1.intersection(c2)) > 1

    def distance(self, p):
        return math.sqrt(self.squared_distance(p))

    def squared_distance(self, p):
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

        return dx*dx + dy*dy

    def intersection(self, s):

        def inter_func(s1, s2):
            a = s1.p1
            b = s1.p2
            c = s2.p1
            d = s2.p2
            f_ab = (d.x - c.x) * (a.y - c.y) - (d.y - c.y) * (a.x - c.x)

            if f_ab == 0:
                return float('inf')

            f_cd = (b.x - a.x) * (a.y - c.y) - (b.y - a.y) * (a.x - c.x)
            f_d = (d.y - c.y) * (b.x - a.x) - (d.x - c.x) * (b.y - a.y)

            if f_d == 0:
                return float('inf')

            f1 = f_ab/float(f_d)
            f2 = f_cd/float(f_d)

            if (f1 <= 0 or f1 >= 1):
                return float('inf')
            if (f2 <= 0 or f2 >= 1):
                return float('inf')
            return f1;

        f = inter_func(self, s)

        if f == float('inf') or f <= 0 or f >= 1:
            return None

        return self.interpolate(f)

    def perpendicular_segment(self, p):
        d = self.p1 - self.p2
        d1, d2 = d.x, d.y
        if d2 == 0:  # If a horizontal line
            if p.y == self.p1.y:  # if p is on this linear entity
                return Segment(p, p + Point(0, 1))
            else:
                p2 = Point(p.x, self.p1.y)
                return Segment(p, p2)
        else:
            p2 = Point(p.x - d2, p.y + 1)
            s = Segment(p, p2)
            return Segment(p, p2)

    @property
    def bounds(self):
        return Rectangle.from_segment(self)

    @property
    def connections(self):
        all_segments = self.p1.segments.union(self.p2.segments)
        return all_segments

    def interpolate(self, f):
        return Point.interpolate(self.p1, self.p2, f)

    @staticmethod
    def union(s1, s2):
        def compare(a, b):
            return int(a[2]-b[2])

        l = [(s1.p1, s2.p1), (s1.p1, s2.p2), (s1.p2, s2.p1), (s1.p2, s2.p2), (s1.p1, s1.p2), (s2.p1, s2.p2)]#TODO
        l = [(p1, p2, Point.distance(p1, p2)) for p1, p2 in l]
        l = sorted(l, cmp=compare)
        p1, p2, d = l[-1]

        return Segment(p1, p2)

    @staticmethod
    def angle_between(s1, s2):
        v1 = s1.p2 - s1.p1
        v2 = s2.p2 - s2.p1
        a = v1.dot(v2)/(abs(v1)*abs(v2))
        a = max(min(a,  1), -1)
        return math.acos(a)

    def __str__(self):
        return type(self).__name__ + "(" + self.p1.__str__() + ", " + self.p2.__str__() + ")"
