
__author__ = 'Mario'

import cv2 as cv
import numpy as np
from skimage import morphology
# import sympy as sy
import os
import math
from geometry import *

IMAGE_PATH = 'Resources/1.bmp'
SKELETON_PATH = 'Resources/sk.png'

LINK_GAP = 10
ATTRACT_GAP = 20
INFLATE_GAP = 1.4*ATTRACT_GAP #sqrt(2)*ATTRACT_GAP

NEAR_MIN_DIST = 5
NEAR_MIN_ANG = np.pi/60

def process_skeleton():
    # Extract Skeleton
    if not os.path.isfile(SKELETON_PATH):
        print 'Generating Skeleton'
        im = cv.imread(IMAGE_PATH)
        im_bin = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
        im_bin[im_bin != 236] = 0
        im_bin[im_bin == 236] = 1

        out = morphology.skeletonize(im_bin)

        out.dtype = np.uint8
        out *= 255

        cv.imwrite(SKELETON_PATH, out)
        skel = out
    else:
        print 'Reading Skeleton'
        skel = cv.imread(SKELETON_PATH)
        skel = cv.cvtColor(skel, cv.COLOR_RGB2GRAY)
    return skel


def clean_segments(segments):
    visited = []
    list = segments[:]

    have_changes = True
    while have_changes:
        print "clean loop", len(visited), len(list)
        have_changes = False
        list1 = list[:]
        list2 = list[:]
        for seg1 in list1:
            list2.remove(seg1)

            r1 = seg1.bounds
            r1.inflate(INFLATE_GAP)

            for seg2 in list2:

                r2 = seg2.bounds
                r2.inflate(INFLATE_GAP)

                if (not seg1.is_connected_with(seg2)) and r1.collide(r2):
                    dist_s1 = seg1.distance(seg2.p1)
                    dist_s2 = seg1.distance(seg2.p2)

                    dist_s3 = seg2.distance(seg1.p1)
                    dist_s4 = seg2.distance(seg1.p2)

                    min_dist = min(dist_s1+dist_s2, dist_s3+dist_s4)
                    angle = Segment.angle_between(seg1, seg2)

                    if angle <= NEAR_MIN_ANG and min_dist <= NEAR_MIN_DIST:
                        have_changes = True
                        list.remove(seg1)
                        list.remove(seg2)

                        s = Segment.union(seg1, seg2)
                        list.append(s)

                if have_changes:
                    break
            if have_changes:
                break
            else:
                visited.append(seg1)
                list.remove(seg1)
    return visited

def connect_segments(segments):

    print "processing"

    visited = []
    list = segments[:]

    def add_seg(seg):
        if isinstance(seg, Segment):
            list.append(seg)
            print "add -----> ", seg

    have_changes = True
    while have_changes:
        print "loop"
        print len(list)

        have_changes = False
        list1 = list[:]
        list2 = list[:]
        for seg1 in list1:
            list2.remove(seg1)

            r1 = seg1.bounds
            r1.inflate(INFLATE_GAP)

            print "SEG1", seg1

            for seg2 in list2:

                print " SEG2", seg2

                r2 = seg2.bounds
                r2.inflate(INFLATE_GAP)

                print " >", r1.p1, r1.p2, " <> ", r2.p1, r2.p2, " #> ", (not seg1.is_connected_with(seg2)), r1.collide(r2)

                if (not seg1.is_connected_with(seg2)) and r1.collide(r2):

                    inters = seg1.intersection(seg2)

                    if len(inters) > 0:
                        have_changes = True
                        print 'inter', seg1, seg2, inters
                        p = inters[0]

                        p = Point(int(p.x), int(p.y))
                        list.remove(seg1)
                        list.remove(seg2)
                        print "remove", seg1, seg2

                        add_seg(Segment(seg1.p1, p))
                        add_seg(Segment(seg1.p2, p))
                        add_seg(Segment(seg2.p1, p))
                        add_seg(Segment(seg2.p2, p))

                    if have_changes:
                        break

                    dist_a = seg1.p1.distance(seg2.p1)
                    dist_b = seg1.p1.distance(seg2.p2)
                    dist_c = seg1.p2.distance(seg2.p1)
                    dist_d = seg1.p2.distance(seg2.p2)

                    min_dist = min(dist_a, dist_b, dist_c, dist_d)

                    if min_dist < LINK_GAP:
                        print 'min', int(min_dist), "|", dist_a, dist_b, dist_c, dist_d
                        have_changes = True

                        create_segment = min_dist > ATTRACT_GAP

                        # if not create_segment:
                        #     list.remove(seg2)
                        #     print "remove", seg2

                        if dist_a == min_dist:
                            print 'A', seg1.p1, seg1.p2, seg2.p1, seg2.p2
                            if create_segment:
                                add_seg(Segment(seg1.p1, seg2.p1))
                                # add_seg(Segment(seg1.p1, seg2.p2))
                            else:
                                seg2.p1 = seg1.p1

                        elif dist_b == min_dist:
                            print 'B', seg1, seg2
                            if create_segment:
                                add_seg(Segment(seg1.p1, seg2.p2))
                            else:
                                seg2.p2 = seg1.p1
                                # add_seg(Segment(seg2.p1, seg1.p1))
                        elif dist_c == min_dist:
                            print 'C', seg1, seg2
                            if create_segment:
                                add_seg(Segment(seg1.p2, seg2.p1))
                            else:
                                seg2.p1 = seg1.p2
                                # add_seg(Segment(seg1.p2, seg2.p2))
                        elif dist_d == min_dist:
                            print 'D', seg1, seg2
                            if create_segment:
                                add_seg(Segment(seg1.p2, seg2.p2))
                            else:
                                seg2.p2 = seg1.p2
                                # add_seg(Segment(seg2.p1, seg1.p2))

                    if have_changes:
                        break

                    dist_s1 = seg1.distance(seg2.p1)
                    dist_s2 = seg1.distance(seg2.p2)
                    min_dist = min(dist_s1, dist_s2)

                    s1 = None

                    if min_dist < LINK_GAP:
                        s1 = seg1
                        s2 = seg2

                    dist_s1 = seg2.distance(seg1.p1)
                    dist_s2 = seg2.distance(seg1.p2)
                    min_dist = min(dist_s1, dist_s2)

                    if min_dist < LINK_GAP:
                        s1 = seg2
                        s2 = seg1

                    print "min", seg1, seg2, " <> ",  int(dist_s1), int(dist_s2)
                    if s1 is not None:
                        have_changes = True

                        p = [s2.p1, s2.p2][min_dist == dist_s2]
                        s = s1.perpendicular_segment(p)
                        p0 = [s.p1, s.p2][sy.Point.is_collinear(s1.p1, s1.p2, s.p1)]
                        p0 = Point(p0.x, p0.y)

                        list.remove(s1)
                        add_seg(Segment(s1.p1, p0))
                        add_seg(Segment(s1.p2, p0))
                        add_seg(Segment(p, p0))

                if have_changes:
                    break
            if have_changes:
                break
            else:
                visited.append(seg1)
                list.remove(seg1)

    return visited

def draw_segments(im, segments):
    for segment in segments:
        x1 = int(segment.p1.x)
        y1 = int(segment.p1.y)
        x2 = int(segment.p2.x)
        y2 = int(segment.p2.y)

        r = np.random.random_sample()*255
        g = np.random.random_sample()*255
        b = np.random.random_sample()*255

        color1 = {0:(0, 255, 0), 1:(0, 0, 255)}[segment.p1.is_joint()]
        color2 = {0:(0, 255, 0), 1:(0, 0, 255)}[segment.p2.is_joint()]

        cv.circle(im, (x1, y1), 5, color1, 1)
        cv.circle(im, (x2, y2), 5, color2, 1)
        cv.line(im, (x1, y1), (x2, y2), (r, g, b), 3)

if __name__ == '__main__':

    s1 = Segment(Point(100, 100), Point(300, 300))
    s2 = Segment(Point(200, 100), Point(200, 400))

    print s1.intersection(s2)

    skeleton = process_skeleton()

    segments = []
    lines = cv.HoughLinesP(skeleton, rho=1, theta=np.pi/180, threshold=5, lines=10, minLineLength=5, maxLineGap=5)
    for x1, y1, x2, y2 in lines[0]:
        segments.append(Segment(Point(x1, y1), Point(x2, y2)))

    # segments = segments[105:107] + segments[10:15]
    # segments = segments[10:30] + segments[100:110]

    # segments = [
    #     Segment(Point(100, 100), Point(100, 200)),
    #     Segment(Point(0, 150),   Point(400, 150)),
    #     Segment(Point(300, 0),   Point(300, 149)),
    #     Segment(Point(401, 150), Point(401, 200)),
    #     Segment(Point(350, 152), Point(450, 152)),
    #     Segment(Point(350, 150), Point(400, 200)),
    #     Segment(Point(400, 200), Point(500, 300)),
    # ]
    # skeleton = np.zeros((500, 500), dtype=np.uint8)


    im_vec = cv.cvtColor(skeleton, cv.COLOR_GRAY2RGB)
    draw_segments(im_vec,  segments)
    cv.imwrite('Resources/vec.png', im_vec)

    segments = clean_segments(segments)
    segments = connect_segments(segments)

    im_seg = cv.cvtColor(skeleton, cv.COLOR_GRAY2RGB)
    draw_segments(im_seg, segments)
    cv.imwrite('Resources/seg.png', im_seg)