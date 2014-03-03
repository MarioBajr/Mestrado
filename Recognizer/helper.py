__author__ = 'Mario'

import cv2 as cv

SHOW_WINDOWS = 1
WINDOW_X = 0
WINDOW_Y = -500
GAPX = 10
GAPY = 20

windowIndex = 0

def showImage(im):
    global windowIndex

    if not SHOW_WINDOWS:
        return


    x = WINDOW_X + (im.shape[1]+GAPX) * (windowIndex % 8)
    y = WINDOW_Y + (im.shape[0]+GAPY) * (windowIndex / 8)


    cv.namedWindow('Image_Window-' + str(windowIndex))
    cv.imshow('Image_Window-' + str(windowIndex), im)
    cv.moveWindow('Image_Window-' + str(windowIndex), x, y)

    windowIndex +=1
