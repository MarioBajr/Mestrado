__author__ = 'Mario'

import numpy as np
import cv2 as cv
import scipy.signal as sgn


def stage1(im):

    def get_template(name):
        template = cv.imread(name)
        template = cv.cvtColor(template, cv.COLOR_RGB2GRAY)
        template = template.astype(np.float32, copy=False)
        return template

    template1 = get_template('template1.png')
    template2 = get_template('template2.png')

    im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    im = im.astype(np.float32, copy=False)

    c1 = mminmax(im)
    c2 = mminmax(template1)
    c3 = mminmax(template2)

    c_1 = sgn.convolve2d(c1, c2, mode='same')
    c_2 = sgn.convolve2d(c1, c3, mode='same')

    re1, thresh1 = cv.threshold(c_1, 65, 255, cv.THRESH_BINARY)
    re2, thresh2 = cv.threshold(c_2, 65, 255, cv.THRESH_BINARY)
    c_3 = thresh1 + thresh2

    cv.imwrite('../Databases/Temp/test/output111.png', c_1)
    cv.imwrite('../Databases/Temp/test/output222.png', c_2)
    cv.imwrite('../Databases/Temp/test/output333.png', c_3)
    # cv.imshow('Stage 1', c_1)
    # cv.waitKey(0)

# Auxiliar

def mminmax(input):
    max = np.max(input)
    min = np.min(input)

    return ((input - min)/(max - min) - 0.5) * 2