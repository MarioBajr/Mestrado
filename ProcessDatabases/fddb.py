__author__ = 'Mario'

import cv2 as cv
import glob
import os
import numpy as np

from sklearn.ensemble import AdaBoostClassifier

def process_fddb():
    extract_faces()

    # print "will save"
    # np_pos = np.array(pos)
    # np.save('../Databases/Temp/fddb_faces/pos', np_pos)
    #
    # np_neg = np.array(neg)
    # np.save('../Databases/Temp/fddb_faces/neg', np_neg)
    #
    # print ">>>>", np_pos.shape, np_neg.shape

def extract_faces():

    #'../Databases/FDDB/FDDB-folds/*.pgm'
    pics_path = '../Databases/FDDB/originalPics/'
    pattern = '../Databases/FDDB/FDDB-folds/FDDB-fold*ellipseList.txt'
    files = glob.glob(pattern)

    for fname in files:

        type = 0
        im_count = 0
        im_path = None
        im = None

        im_rects = None

        f = open(fname)
        for idx, line in enumerate(f):
            # if idx > 20:
            #     return

            if type == 0:
                type = 1
                im_path = line.split()[0] + '.jpg'
                path = pics_path + im_path
                im = cv.imread(path)
                im_rects = []
            elif type == 1:
                type = 2
                im_count = int(line)
            else:
                w, h, a, x, y, i = line.split()
                x = int(float(x))
                y = int(float(y))
                a = float(a)
                w = int(float(w))
                h = int(float(h))

                mx = max(w, h)
                mi = min(w, h)

                # cv.ellipse(im, (x, y), (w, h), a*(180/np.pi), 0, 360, (255, 0, 0))
                # cv.rectangle(im, (x-mi, y-mi), (x+mi, y+mi), (0, 255, 0))
                # cv.circle(im, (x, y), mi, (0, 0, 255))

                print 'ellipse', im_count

                x0 = max(0, x-mi)
                x1 = x+mi
                y0 = max(0, y-mi)
                y1 = y+mi
                print idx, "=>>", x, y, mi,  x0, x1, y0, y1, ">>", x1-x0, y1-y0
                print im.shape

                im_crop = im[y0:y1, x0:x1, :]
                im_crop = cv.resize(im_crop, (32, 32))
                im_rects.append(Rectangle(Point(x0, y0), Point(x1, y1)))

                # cv.imwrite('../Databases/Temp/fddb_faces/pos/'+str(idx)+'.jpg', im_crop)
                process_fddb.pos.append(im_crop)

                im_count -= 1
                if im_count == 0:
                    type = 0
                    # path = '../Databases/Temp/fddb/'+str(idx)+'.jpg'
                    # cv.imwrite(path, im)
                    extract_negatives(im, im_rects)

        print "writing file"
        np_pos = np.array(process_fddb.pos)
        np.save('../Databases/Temp/fddb_faces/pos_%s' % fname.split('/')[-1], np_pos)
        process_fddb.pos = []
        del np_pos

        np_neg = np.array(process_fddb.neg)
        np.save('../Databases/Temp/fddb_faces/neg_%s' % fname.split('/')[-1], np_neg)
        process_fddb.neg = []
        del np_neg
        print "done .."



def extract_negatives(im, rects, window_edge=32, step_trans=32, step_scale=1.2):
    ih, iw, ic = im.shape

    attributes = []
    patterns = []

    s = window_edge

    def collide(r):
        for f in rects:
            if r.collide(f):
                return True
        return False

    while s < min(ih, iw):
        for i in range(0, ih, step_trans):
            for j in range(0, iw, step_trans):
                if i+s <= ih and j+s <= iw:
                    r = Rectangle(Point(j, i), Point(j+s, i+s))
                    if not collide(r):
                        crop = im[i:i+s, j:j+s, :]
                        window = cv.resize(crop, (window_edge, window_edge))
                        # cv.imwrite('../Databases/Temp/fddb_faces/neg/'+str(extract_negatives.counter)+'.jpg', window)
                        process_fddb.neg.append(window)

                        extract_negatives.counter += 1

                        # patterns.append(window)
                        # attributes.append([(j, i), (j+s, i+s)])
        print "neg", extract_negatives.counter
        s = int(s*step_scale)

    return attributes, patterns
extract_negatives.counter = 0
process_fddb.pos = []
process_fddb.neg = []

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


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