__author__ = 'Mario'


import cv2 as cv


def pyramidal_scale(im, rect=(64,64), step_trans=2, step_scale=1.2):
    ih, iw, id = im.shape

    attributes = []
    patterns = []

    sw = rect[0]
    sh = rect[1]

    while sh < ih or sw < iw:
        for i in range(0, ih, step_trans):
            for j in range(0, iw, step_trans):
                if i+sh <= ih and j+sw <= iw:
                    crop = im[i:i+sh, j:j+sw, :]
                    window = cv.resize(crop, rect)
                    patterns.append(window)
                    attributes.append([(j, i), (j+sh, i+sw)])
        sw = int(sw*step_scale)
        sh = int(sh*step_scale)

    return attributes, patterns