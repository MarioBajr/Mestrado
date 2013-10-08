__author__ = 'Mario'

import cv2 as cv
import matplotlib.pyplot as plt

def extract_feature(window, feature, im):

    im_gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    FeatureDetector = cv.FeatureDetector_create(feature)
    keyPoints = FeatureDetector.detect(im_gray)

    plt.figure(window)
    for k in keyPoints:
        x,y = k.pt
        plt.plot(x,-y,'ro')
    plt.axis('equal')

if __name__ == '__main__':

    im = cv.imread('../Databases/Temp/tank.jpg')

    h,w,d = im.shape

    dh = h-50
    dw = w-50

    out = im[10:dh, 10:dw]
    extract_feature('10x10', 'SIFT', out)

    out = im[20:dh, 10:dw]
    extract_feature('20x10', 'SIFT', out)

    out = im[0:dh, 0:dw]
    extract_feature('0x0', 'SIFT', out)

    out = im[50:dh, 10:dw]
    extract_feature('50x10', 'SIFT', out)

    plt.show()