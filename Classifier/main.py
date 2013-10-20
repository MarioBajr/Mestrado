# -*- coding: utf-8 -*-

__author__ = 'Mario'

import cv2 as cv
import features as ft
from mixture import *

if __name__ == '__main__':

    classifiers = [
        #LVQClassifier(ClassifierType.LVQ, [ft.Features.DAISY], .5),
        #LVQClassifier(ClassifierType.LVQ, [ft.Features.LBP, ft.Features.HOG], .5),
        LVQClassifier(ClassifierType.LVQ, [ft.Features.LBP, ft.Features.GABOR], .25),
    ]

    mixture = Mixture(classifiers)

    im = cv.imread('../Databases/Temp/test/Mosaic.png')
    out = mixture.run(im)
    cv.imwrite('../Databases/Temp/test/output.png', out)