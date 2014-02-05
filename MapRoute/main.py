
__author__ = 'Mario'

import cv2 as cv
import numpy as np
from skimage import morphology
import os
from geometry import *


IMAGE_PATH = 'Resources/1.png'
SKELETON_PATH = 'Resources/sk.png'

NEW_SEG_GAP = 12
LINK_SEG_GAP = 10
INFLATE_GAP = NEW_SEG_GAP

NEAR_MIN_DIST = 10
NEAR_MIN_ANG = np.pi/36

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

    def filter_small_segments(s):
        return s.squared_length > 50

    list = filter(filter_small_segments, segments)
    print len(segments), len(list)
    # list = segments[:]


    #work only with squared values to improve performance
    SQUARED_NEAR_MIN_DIST = NEAR_MIN_DIST*NEAR_MIN_DIST

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

            # print "SEG1", seg1

            for seg2 in list2:

                r2 = seg2.bounds
                r2.inflate(INFLATE_GAP)

                # print " SEG2", seg2

                if r1.collide(r2):
                    dist_s1 = seg1.squared_distance(seg2.p1)
                    dist_s2 = seg1.squared_distance(seg2.p2)

                    dist_s3 = seg2.squared_distance(seg1.p1)
                    dist_s4 = seg2.squared_distance(seg1.p2)

                    min_dist = min(dist_s1, dist_s2, dist_s3, dist_s4)
                    angle = Segment.angle_between(seg1, seg2)
                    angle = abs(min(angle, np.pi-angle))

                    # print " COLLIDE", min_dist, angle, SQUARED_NEAR_MIN_DIST

                    if angle <= NEAR_MIN_ANG and min_dist <= SQUARED_NEAR_MIN_DIST:
                        have_changes = True
                        list.remove(seg1)
                        list.remove(seg2)
                        seg1.unlink()
                        seg2.unlink()

                        s = Segment.union(seg1, seg2)
                        list.append(s)

                        print "remove", seg1, seg2, " ===> ", s

                if have_changes:
                    break
            if have_changes:
                break
            else:
                visited.append(seg1)
                list.remove(seg1)

    print "finish clean", len(visited)
    for i in visited:
        print i
    return visited

def connect_segments(segments):

    print "\n\n\nprocessing"

    visited = []
    list = segments[:]

    def sorter_func(a, b):
        da = Point.squared_distance(a.center, Point.zero())
        db = Point.squared_distance(b.center, Point.zero())
        return da - db

    list = sorted(list, cmp=sorter_func)

    def add_seg(seg):
        if isinstance(seg, Segment):
            list.append(seg)

    def rm_seg(seg):
        seg.unlink()
        list.remove(seg)

    have_changes = True
    while have_changes:
        print "loop ----------------------", len(list)

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

                # if seg2.squared_length <= 1:
                #     rm_seg(seg2)
                #     continue

                r2 = seg2.bounds
                r2.inflate(INFLATE_GAP)

                print " HIT", (not seg1.is_connected_with(seg2)), r1.collide(r2)

                if (not seg1.is_connected_with(seg2)) and r1.collide(r2):

                    p = seg1.intersection(seg2)

                    if p is not None:
                        print "--\n<Cross>"
                        have_changes = True
                        rm_seg(seg1)
                        rm_seg(seg2)

                        add_seg(Segment(seg1.p1, p))
                        add_seg(Segment(seg1.p2, p))
                        add_seg(Segment(seg2.p1, p))
                        add_seg(Segment(seg2.p2, p))

                    if have_changes:
                        break

                    dist_a = Point.distance(seg1.p1, seg2.p1)
                    dist_b = Point.distance(seg1.p1, seg2.p2)
                    dist_c = Point.distance(seg1.p2, seg2.p1)
                    dist_d = Point.distance(seg1.p2, seg2.p2)

                    min_dist = min(dist_a, dist_b, dist_c, dist_d)

                    if min_dist < NEW_SEG_GAP:
                        have_changes = True

                        create_segment = min_dist > LINK_SEG_GAP

                        if create_segment:
                            print "--\n<New Seg>", min_dist
                        else:
                            print "--\n<Link>", min_dist

                        # if not create_segment:
                        #     list.remove(seg2)
                        #     print "remove", seg2
                        #TODO
                        if dist_a == min_dist:
                            print 'A', seg1, seg2
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

                    gap = LINK_SEG_GAP*LINK_SEG_GAP

                    dist, s, p = seg1.nearest_distance_from_segment(seg2)

                    if dist < gap:
                        print "--\n<Next>"
                        have_changes = True

                        s0 = [seg1, seg2][s == seg1]
                        p1 = [s0.p1, s0.p2][p == s0.p1]
                        st = Segment(Segment.interpolate(s0, -1), Segment.interpolate(s0, 2))
                        p0 = s.intersection(st)

                        rm_seg(s0)
                        rm_seg(s)
                        add_seg(Segment(s.p1, p0))
                        add_seg(Segment(s.p2, p0))
                        add_seg(Segment(p1, p0))

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
        rec = Rectangle.from_segment(segment)
        rec.inflate(INFLATE_GAP)
        cv.rectangle(im, rec.p1.to_tuple(), rec.p2.to_tuple(), (100, 0, 0, 0), 1)

    for segment in segments:
        x1 = int(segment.p1.x)
        y1 = int(segment.p1.y)
        x2 = int(segment.p2.x)
        y2 = int(segment.p2.y)

        color1 = [(0, 255, 0, 255), (0, 0, 255, 255)][segment.p1.is_joint()]
        color2 = [(0, 255, 0, 255), (0, 0, 255, 255)][segment.p2.is_joint()]

        cv.circle(im, (x1, y1), NEW_SEG_GAP, color1, 1)
        cv.circle(im, (x2, y2), NEW_SEG_GAP, color2, 1)
        cv.line(im, (x1, y1), (x2, y2), random_color(), 3)


def write_segments(im, filename, segments):
    im_out = np.copy(im)
    draw_segments(im_out, segments)
    cv.imwrite(filename, im_out)


def filter_segment_at_rect(im, segments, rect):
    cv.rectangle(skeleton, rect.p1.to_tuple(), rect.p2.to_tuple(), 100, -1)

    def filter_segment(s):
        r2 = Rectangle.from_segment(s)
        return rect.collide(r2)
    return filter(filter_segment, segments)


def random_color():
    r = int(np.random.random_sample()*255)
    g = int(np.random.random_sample()*255)
    b = int(np.random.random_sample()*255)
    return r, g, b, 255


def approach1():
    skeleton = process_skeleton()

    segments = []
    lines = cv.HoughLinesP(skeleton, rho=1, theta=np.pi/180, threshold=5, lines=10, minLineLength=5, maxLineGap=5)
    for x1, y1, x2, y2 in lines[0]:
        segments.append(Segment(Point(x1, y1), Point(x2, y2)))

    # Filter Segments for DEBUG
    # segments = filter_segment_at_rect(skeleton, segments, Rectangle(Point(1730, 800), Point(1850, 850)))

    template = cv.cvtColor(skeleton, cv.COLOR_GRAY2RGBA)

    #Print Base
    # base = cv.imread('Resources/1.png')
    # base = cv.cvtColor(base, cv.COLOR_RGB2RGBA)
    # template = base+template


    write_segments(template, 'Resources/vec.png', segments)

    segments = clean_segments(segments)

    write_segments(template, 'Resources/clean.png', segments)

    segments = connect_segments(segments)

    write_segments(template, 'Resources/seg.png', segments)


if __name__ == '__main__':

    skeleton = process_skeleton()

    contour, hier = cv.findContours(skeleton, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)

    out = cv.cvtColor(skeleton, cv.COLOR_GRAY2RGB)

    segments = []

    for cnt in contour:

        cnt2 = cv.approxPolyDP(cnt, 3, True)

        cv.drawContours(out, [cnt2], 0, random_color(), 2)

        l = cnt2.shape[0]
        last_p = Point(cnt2[-1][0][0], cnt2[-1][0][1])
        for i in range(l):
            x = cnt2[i][0][0]
            y = cnt2[i][0][1]
            r = 3 + int(np.random.random_sample()*4)
            cv.circle(out, (x, y), r, random_color(), -1)

            curr_p = Point(x, y)
            s = Segment(last_p, curr_p)
            segments.append(s)
            last_p = curr_p

    # segments = filter_segment_at_rect(skeleton, segments, Rectangle(Point(500, 700), Point(600, 900)))

    template = cv.cvtColor(skeleton, cv.COLOR_GRAY2RGBA)
    #Print Base
    base = cv.imread('Resources/1.png')
    base = cv.cvtColor(base, cv.COLOR_RGB2RGBA)
    template = base+template


    write_segments(template, 'Resources/vec.png', segments)

    segments = clean_segments(segments)
    write_segments(template, 'Resources/clean.png', segments)

    segments = connect_segments(segments)
    write_segments(template, 'Resources/seg.png', segments)