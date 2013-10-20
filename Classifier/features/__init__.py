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


# Output 1D
def _hog_feature(im):
    return feature.hog(im, visualise=False, normalise=False)

def _opencv_feature(feature, im):

    #Normilizing Image
    min = np.min(im)
    max = np.max(im)

    src= ((im- min) / (max - min)) * 255
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
    return feature.local_binary_pattern(im, 8, 3)


def _harris_feature(im):
    return feature.corner_harris(im)


def _daisy_feature(im):
    return feature.daisy(im)


def _corner_shi_tomasi_feature(im):
    return feature.corner_shi_tomasi(im)


def _gabor_feature(im):
    angles = 9
    lst = []
    for i, angle in [(a, math.degrees(a*math.pi/angles)) for a in range(angles)]:
        out = gabor_filter.Process(im, 5, 50, angle, 90)
        out *= 255
        lst.append(np.array(out))
    out = np.vstack([np.hstack(lst[0:3]), np.hstack(lst[3:6]), np.hstack(lst[6:9])])
    return out

def _receptive_fields(im):
    a = _field_window(im, (4, 4))
    b = _field_window(im, (8, 8))
    c = _field_window(im, (8, 1))
    return a+b+c


def _field_window(im, d):
    dy, dx = d
    h = im.shape[0]/dy
    w = im.shape[1]/dx

    output = []
    for i in range(dy):
        y = i*h
        for j in range(dx):
            x = j*w
            out = im[y:y+h, x:x+w]
            output.append(out.mean())
    return output


def compose_features(im, features):
    output = im
    if len(output.shape) == 3:
        output = cv.cvtColor(output, cv.COLOR_RGB2GRAY)

    cv.equalizeHist(output, output)

    for f in features:
        method_name = '_%s_feature' % f.lower()
        if method_name in globals():
            output = globals()[method_name](output)
        else:
            output = _opencv_feature(f, output)

    #output = cv.resize(output, (32, 32))
    #return np.array(_receptive_fields(output))
    return np.reshape(output, -1)