__author__ = 'Mario'


import cv2 as cv
import numpy as np
from skimage import morphology
import os
from geometry import *


IMAGE_PATH = 'Resources/1.png'
SKELETON_PATH = 'Resources/sk.png'


def random_color():
    r = int(np.random.random_sample()*255)
    g = int(np.random.random_sample()*255)
    b = int(np.random.random_sample()*255)
    return r, g, b

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

def process_graph():

    skeleton = process_skeleton()

    contour, hier = cv.findContours(skeleton, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
    print len(contour)

    out = cv.cvtColor(skeleton, cv.COLOR_GRAY2RGB)

    for cnt in contour:

        cv.drawContours(out, [cnt], 0, random_color(), 7)

        cnt2 = cv.approxPolyDP(cnt, 3, True)

        cv.drawContours(out, [cnt2], 0, random_color(), 2)

        l = cnt2.shape[0]
        for i in range(l):
            x = cnt2[i][0][0]
            y = cnt2[i][0][1]
            cv.circle(out, (x, y), 3, (255, 0, 0), -1)

        print cnt.shape, cnt2.shape
        print hier

    # out = morphology.skeletonize(im_bin)

    # out.dtype = np.uint8
    # out *= 255

    cv.imwrite('Resources/___1.png', out)


if __name__ == '__main__':

    process_graph()