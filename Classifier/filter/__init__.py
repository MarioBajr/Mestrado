__author__ = 'Mario'

import math
import cv2 as cv
import numpy as np
from numpy.core.umath import log10
from scipy.ndimage.filters import *
from scipy.signal import *

import matplotlib.pyplot as plt


def show_colormap(im):
    imgplot = plt.imshow(im)
    imgplot.set_cmap('spectral')
    plt.colorbar()
    plt.show()

def log_opp(x):
    return 105*log10(x+1)


def skin_filter(im):

    h, w, c = im.shape

    red = im[:, :, 0]
    green = im[:, :, 1]
    blue = im[:, :, 2]

    #Accout for zero response of camera. Subtract the smallest
    #pixel values > 10 pixels from any edge from the rgb matrices
    zero_resp = im[10:h-10, 10:w-10, :].min()

    red = red - zero_resp
    green = green - zero_resp
    blue = blue - zero_resp

    red = red.astype(np.float)
    green = green.astype(np.float)
    blue = blue.astype(np.float)

    #Transform the RGB values into log-opponent values I, Rg, Ry
    I = (log_opp(red)+log_opp(green)+log_opp(blue))/3.0
    Rg = log_opp(red)-I
    By = log_opp(blue)-(I+log_opp(red))/2.0

    scale = (h+w)/320.0
    scale = int(round(scale))
    scale = max(1, scale)

    #Rg = median_filter(Rg, (4*scale, 4*scale))
    Rg = medfilt2d(Rg, (4*scale+1, 4*scale+1))
    #Rg = cv.medianBlur(Rg, 4*scale+1)

    #By = median_filter(By, (4*scale, 4*scale))
    By = medfilt2d(By, (4*scale+1, 4*scale+1))
    #cv.medianBlur(By, (4*scale, 4*scale))

    #Compute texture ampliture
    #I_filt = median_filter(I, (8*scale, 8*scale))
    I_filt = medfilt2d(I, (8*scale+1, 8*scale+1))

    MAD = I-I_filt
    MAD = abs(MAD)
    #MAD = median_filter(MAD, (12*scale, 12*scale))
    MAD = medfilt2d(MAD, (12*scale+1, 12*scale+1))


    hue = np.arctan2(By, Rg) * (180 / np.pi)
    saturation = np.sqrt(np.power(Rg, 2) + np.power(By, 2))

    #hue = hue.astype(np.int64)
    max_hue = hue.max()
    min_hue = hue.min()
    hue = ((hue-min_hue)/(max_hue-min_hue))
    hue *= 255
    print max_hue, min_hue, hue.min(), hue.max()
    print MAD.min(), MAD.max()

    saturation = saturation.astype(np.uint8)
    #saturation = cv.equalizeHist(saturation, saturation)

    out = np.zeros((h, w))

    #Detect skin texture regions
    for y in range(h):
        for x in range(w):
            #if MAD[y, x] < 4.5 and 120 < hue[y, x] < 160 and 10 < saturation[y, x] < 60:
            if 120 < hue[y, x] < 240 and 10 < saturation[y, x] < 60:
                out[y, x] = 1

            #if MAD[y, x] < 4.5 and 150 < hue[y, x] < 180 and 20 < saturation[y, x] < 80:
            if 150 < hue[y, x] < 180 and 20 < saturation[y, x] < 80:
                out[y, x] = 1

    #Expand skin regions
    out = cv.dilate(out, (5, 5))

    ##Shrink regions
    #for y in range(h):
    #    for x in range(w):
    #        #if out[y, x] == 1 and 110 <= hue[y, x] <= 180 and 0 <= saturation[y, x] <= 130:
    #        if out[y, x] == 1 and 110 <= hue[y, x] <= 240 and 0 <= saturation[y, x] <= 130:
    #            out[y, x] = 1
    #        else:
    #            out[y, x] = 0

    show_colormap(MAD)
    show_colormap(hue)
    show_colormap(saturation)
    show_colormap(out)

    out *= 255

    return out, I, Rg, By, hue, saturation, MAD