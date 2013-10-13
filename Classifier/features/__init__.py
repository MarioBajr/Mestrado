__author__ = 'Mario'

import numpy as np
import cv2 as cv
import math
import gabor_filter

from skimage import feature

def enum(**enums):
    return type('Enum', (), enums)

Features = enum(HOG='hog',
                LBP='LBP',
                HARRIS='harris',
                DAISY='daisy',
                SHI_TOMASI='corner_shi_tomasi',
                GABOR='gabor',
                #OpenCV Simple Features
                FAST='FAST',
                STAR='STAR',
                SIFT='SIFT',
                SURF='SURF',
                ORB='ORB',
                MSER='MSER',
                GFTT='GFTT',
                #HARRIS="HARRIS",
                #OpenCV Grid Features
                GridFAST='GridFAST',
                GridSTAR='GridSTAR',
                GridSIFT='GridSIFT',
                GridSURF='GridSURF',
                GridORB='GridORB',
                GridMSER='GridMSER',
                GridGFTT='GridGFTT',
                GridHARRIS="GridHARRIS",
                #OpenCV Pyramid Features
                PyramidFAST='PyramidFAST',
                PyramidSTAR='PyramidSTAR',
                PyramidSIFT='PyramidSIFT',
                PyramidSURF='PyramidSURF',
                PyramidORB='PyramidORB',
                PyramidMSER='PyramidMSER',
                PyramidGFTT='PyramidGFTT',
                PyramidHARRIS="PyramidHARRIS",
                )

# Auxiliar

def _reduce_image(im, s):
    h = int(im.shape[0]*s)
    w = int(im.shape[1]*s)
    return cv.resize(im, (h, w))

# Output 1D

def _hog_feature(im):
    src = im
    if len(src.shape) == 3:
        src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)

    return feature.hog(src, visualise=False, normalise=False)

def _opencv_feature(feature, im):

    src = im
    if len(src.shape) == 3:
        src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)

    #Normilizing Image
    min = np.min(src)
    max = np.max(src)

    src= ((src- min) / (max - min)) * 255
    src = src.astype(np.uint8)

    FeatureDetector = cv.FeatureDetector_create(feature)
    # DescriptorExtractor = cv.DescriptorExtractor_create(feature)

    keyPoints = FeatureDetector.detect(src)
    # keyPoints, kpDescriptors = DescriptorExtractor.compute(im_gray,keyPoints)

    del FeatureDetector

    n = im.shape[0]
    out = np.zeros(im.shape)
    for k in keyPoints:
        a, b = k.pt
        r = int(k.size)

        for i in range(r):
            y,x = np.ogrid[-a:n-a, -b:n-b]
            mask = x*x + y*y <= i*i
            out[mask] += 1

    return out

# Output 2D

def _lbp_feature(im):
    out = im
    if len(im.shape) == 3:
        out = cv.cvtColor(out, cv.COLOR_RGB2GRAY)
    return feature.local_binary_pattern(out, 8, 3)

def _harris_feature(im):
    return feature.corner_harris(im)

def _daisy_feature(im):
    out = im
    if len(im.shape) == 3:
        out = cv.cvtColor(out, cv.COLOR_RGB2GRAY)
    return feature.daisy(out)

def _corner_shi_tomasi_feature(im):
    return feature.corner_shi_tomasi(im)

def _gabor_feature(im):
    src = im
    if len(src.shape) == 3:
        src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    angles = 9
    lst = []
    for i, angle in [(a, math.degrees(a*math.pi/angles)) for a in range(angles)]:
        out = gabor_filter.Process(src, 5, 50, angle, 90)
        out *= 255
        lst.append(np.array(out))
    out = np.vstack([np.hstack(lst[0:3]), np.hstack(lst[3:6]), np.hstack(lst[6:9])])
    return out
# Features Extractors

def compose_features(im, features, scale):
    output = _reduce_image(im, scale)
    for feature in features:
        method_name = '_%s_feature' % feature.lower()
        if method_name in globals():
        # if hasattr(globals(), method_name):
            output = globals()[method_name](output)
        else:
            output = _opencv_feature(feature, output)

    return np.reshape(output, -1)