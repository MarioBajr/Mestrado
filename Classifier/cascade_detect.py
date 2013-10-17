__author__ = 'Mario'


import cv2 as cv
import math

def pyramidal_scale(im, window_edge=64, step_trans=16, step_scale=.3):
    s_max = 1/(min(im.shape[:1]) / float(window_edge))

    step = int(math.ceil((1 - s_max)/step_scale))

    ih, iw, id = im.shape

    attributes = []
    patterns = []

    for k in range(step):

        s = 1 - (k * step_scale)

        h = int(ih * float(s))
        w = int(iw * float(s))

        out = cv.resize(im, (w, h))
        d = int(window_edge/float(s))

        for i in range(0, h, step_trans):
            for j in range(0, w, step_trans):
                if i+window_edge <= h and j+window_edge <= w:
                    window = out[i:i+window_edge, j:j+window_edge, :]
                    patterns.append(window)

                    x = int(j/float(s))
                    y = int(i/float(s))
                    attributes.append([(x, y), (x+d, y+d)])
            # break

    return attributes, patterns