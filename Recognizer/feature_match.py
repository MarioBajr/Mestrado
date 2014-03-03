__author__ = 'Mario'

import numpy as np
import cv2 as cv
import math
import itertools
import helper

FEATURE = "SIFT"

def match(img1, img2, dist=200, qtd=5):

    kp1, kp2, dsc1, dsc2 = findKeyPoints(img1, img2, dist)

    min_d = (min(img2.shape[:2])/2.0) ** 2

    rkp1 = []
    rkp2 = []
    rdsc1 = []
    rdsc2 = []

    num = min(len(kp1), len(kp2), qtd)
    for i in range(num):
        x1, y1 = int(kp1[i].pt[0]), int(kp1[i].pt[1])
        x2, y2 = int(kp2[i].pt[0]), int(kp2[i].pt[1])
        d = (x2-x1)**2 + (y2-y1)**2

        if d < min_d:
            rkp1.append(kp1[i])
            rkp2.append(kp2[i])
            rdsc1.append(dsc1[i])
            rdsc2.append(dsc2[i])

        if len(rkp1) >= qtd:
            break

    newimg = drawKeyPoints(img1, img2, rkp1, rkp2, num=num)
    helper.showImage(newimg)

    return rkp1, rkp2, rdsc1, rdsc2

def findKeyPoints(img1, img2, distance):
    detector = cv.FeatureDetector_create(FEATURE)
    descriptor = cv.DescriptorExtractor_create(FEATURE)

    skp = detector.detect(img1)
    skp, sd = descriptor.compute(img1, skp)

    tkp = detector.detect(img2)
    tkp, td = descriptor.compute(img2, tkp)

    skp_final, sdsc_final = flann_key_points(skp, sd, td, distance)
    tkp_final, tdsc_final = flann_key_points(tkp, td, sd, distance)

    return skp_final, tkp_final, sdsc_final, tdsc_final

def flann_key_points(kp1, desc1, desc2, distance):
    flann_params = dict(algorithm=1, trees=4)
    flann = cv.flann_Index(desc1, flann_params)
    idx, dist = flann.knnSearch(desc2, 1, params={})
    del flann

    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    key_points_final = []
    description_final = []
    for i, dis in itertools.izip(idx, dist):
        if dis < distance:
            key_points_final.append(kp1[i])
            description_final.append(desc1[i])
        else:
            break;

    return key_points_final, description_final

def drawKeyPoints(img1, img2, kp1, kp2, num=-1):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    nWidth = w1+w2
    nHeight = max(h1, h2)
    hdif = (h1-h2)/2
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[hdif:hdif+h2, :w2] = img2
    newimg[:h1, w2:w1+w2] = img1

    maxlen = min(len(kp1), len(kp2))
    if num < 0 or num > maxlen:
        num = maxlen
    for i in range(num):
        pt_a = (int(kp2[i].pt[0]), int(kp2[i].pt[1]+hdif))
        pt_b = (int(kp1[i].pt[0]+w2), int(kp1[i].pt[1]))

        #TEMP begin
        dy = pt_a[1] - pt_b[1]

        if abs(dy) < h1/5.0:
            cv.line(newimg, pt_a, pt_b, (255, 0, 0))
        else:
            cv.line(newimg, pt_a, pt_b, (0, 0, 255))

        #TEMP end

        # cv.line(newimg, pt_a, pt_b, (255, 0, 0))
    return newimg