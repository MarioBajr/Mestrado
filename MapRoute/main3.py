__author__ = 'Mario'


import cv2 as cv
import numpy as np
from skimage import morphology
import os
from geometry import *

IMAGE_PATH = 'Resources/1.png'
SKELETON_PATH = 'Resources/sk.png'
im_source = cv.imread(IMAGE_PATH)

def get_walking_area():
    im_bin = cv.cvtColor(im_source , cv.COLOR_RGB2GRAY)
    im_bin[im_bin != 236] = 0
    im_bin[im_bin == 236] = 255
    return im_bin

def get_skeleton():
    # Extract Skeleton
    if not os.path.isfile(SKELETON_PATH):
        print 'Generating Skeleton'
        im = im_source
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


if __name__ == '__main__':

    skeleton = get_skeleton()
    cv.imwrite('Resources/t1.png', skeleton)

    im_walking_area = get_walking_area()
    cv.imwrite('Resources/t2.png', im_walking_area)

    im_mask = np.copy(im_walking_area)
    im_mask[im_mask == 255] = 1

    k = 51
    s = 100
    im_blur = cv.GaussianBlur(im_walking_area, (k, k), sigmaX=s)
     # im_blur = cv.blur(im_walking_area, (k, k))
    cv.imwrite('Resources/t3.png', im_blur)

    # print skeleton.shape, im_blur.shape, skeleton.max(), im_blur.max()

    im_skel = cv.dilate(skeleton, np.ones((3, 3)), iterations=2)

    im_blur = np.maximum(im_blur, im_skel)
    # print im_blur.shape, im_blur.dtype
    # im_blur = im_blur.astype(np.uint16, copy=False)
    # print im_blur.shape

    # im_blur *= 1.2
    # im_blur = np.maximum(im_blur, im_skel)
    # im_blur = np.minimum(im_blur, np.ones(im_blur.shape)*255)

    cv.imwrite('Resources/t4.png', im_blur)

    im_out = (im_blur * im_mask)
    cv.imwrite('Resources/t5.png', im_out)



    # base = cv.cvtColor(skeleton, cv.COLOR_RGB2RGBA)

    # write_segments(template, 'Resources/t1.png', segments)
