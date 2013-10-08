__author__ = 'Mario'

# vec<Rect> candidates
# vec<int> levels
# vec<double> wights
# bool outputRejectLevels
# def detect_single_scale(im, stripCount, stripSize, rectSize, yStep, factor, classifier):
#
#     if not classifier(im):
#         return False;

import numpy as np
import cv2 as cv

def pyramidal_scale(im, window_edge=64):
    MOVE_WINDOW = 20
    SCALE_STEP = .2
    s_max = 1/(min(im.shape[:1]) / float(window_edge))

    # s_step = int(s_max/SCALE_STEP)

    step = int((1 - s_max)/SCALE_STEP)

    IH, IW, ID = im.shape

    attributes = []
    patterns = []

    # feature = feature_extractor(im)
    # H, W = feature.shape
    # cv.resize(im, (h, w))

    # out = cv.resize)

    print min(im.shape), s_max, step

    for k in range(step):

        s = 1 - (k * SCALE_STEP)

        h = int(IH * float(s))
        w = int(IW * float(s))

        out = cv.resize(im, (w, h))
        d = int(window_edge/float(s))

        for i in range(0, h-window_edge, MOVE_WINDOW):
            for j in range(0, w-window_edge, MOVE_WINDOW):
                if i+window_edge <= h and j+window_edge <= w:
                    window = out[i:i+window_edge, j:j+window_edge]
                    patterns.append(window)

                    x = int(j/float(s))
                    y = int(i/float(s))
                    attributes.append([(x, y), (x+d, y+d)])
            # break

    return (attributes, patterns)