# -*- coding: utf-8 -*-

__author__ = 'Mario'

import cv2 as cv
import neural_network as nn
import features as ft
from mixture import *
import filter as fr

import image_scan as ims

if __name__ == '__main__':

    classifiers = [
       #LVQClassifier([ft.Features.DAISY], .5),
       #ELMClassifier(nn.ELM_Type.ELM_sin, [ft.Features.DAISY], .5),
       #LVQClassifier([ft.Features.LBP, ft.Features.HOG], .5),
       LVQClassifier([ft.Features.GABOR], 1),
       LVQClassifier([ft.Features.LBP, ft.Features.GABOR], 1),
       # AdaBoostClassifier([ft.Features.LBP, ft.Features.GABOR], .25)
    ]

    mixture = Mixture(classifiers)

    im = cv.imread('../Databases/Temp/test/im1.jpg')
    out = mixture.run(im)
    cv.imwrite('../Databases/Temp/test/output.png', out)


    #-------------------------------------------------------

    # im = cv.imread('../Databases/Temp/test/im3.jpg')
    # ims.stage1(im)