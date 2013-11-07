__author__ = 'Mario'


import cv2 as cv


def pyramidal_scale(im, window_edge=64, step_trans=2, step_scale=1.2):
    ih, iw, id = im.shape

    attributes = []
    patterns = []

    s = window_edge

    while s < min(ih, iw):
        for i in range(0, ih, step_trans):
            for j in range(0, iw, step_trans):
                if i+s <= ih and j+s <= iw:
                    crop = im[i:i+s, j:j+s, :]
                    window = cv.resize(crop, (window_edge, window_edge))
                    patterns.append(window)
                    attributes.append([(j, i), (j+s, i+s)])
        s = int(s*step_scale)

    return attributes, patterns