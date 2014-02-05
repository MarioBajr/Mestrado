__author__ = 'Mario'

import cv2 as cv

SHOW_WINDOWS = 1
WINDOW_X = 0
WINDOW_Y = 100#1000

windowIndex = 0

def showImage(im):
    global windowIndex

    if not SHOW_WINDOWS:
        return


    x = WINDOW_X + im.shape[1] * (windowIndex % 10)
    y = WINDOW_Y + im.shape[0] * (windowIndex / 10)


    cv.namedWindow('Image_Window-' + str(windowIndex))
    cv.imshow('Image_Window-' + str(windowIndex), im)
    cv.moveWindow('Image_Window-' + str(windowIndex), x, y)

    windowIndex +=1
