__author__ = 'Mario'

import cv2 as cv
import numpy as np


def mkKernel(ks, sig, th , lm, ps):
    if not ks % 2:
        exit(1)
    hks = ks/2
    theta = th * np.pi/180.
    psi = ps * np.pi/180.
    xs=np.linspace(-1.,1.,ks)
    ys=np.linspace(-1.,1.,ks)
    lmbd = np.float(lm)
    x,y = np.meshgrid(xs,ys)
    sigma = np.float(sig)/ks
    x_theta = x*np.cos(theta)+y*np.sin(theta)
    y_theta = -x*np.sin(theta)+y*np.cos(theta)
    return np.array(np.exp(-0.5*(x_theta**2+y_theta**2)/sigma**2)*np.cos(2.*np.pi*x_theta/lmbd + psi),dtype=np.float32)

# sigma    5 21
# lambda  50 100
# phase    0 180
# psi     90 360


def Process(img, sig, lm, pha, psi):
    kernel_size = 21

    src = np.array(img, dtype=np.float32)
    src /= 255
    if not kernel_size % 2:
        kernel_size +=1

    lm = 0.5+lm/100.
    kernel = mkKernel(kernel_size, sig, pha, lm, psi)
    return cv.filter2D(src, cv.CV_32F,kernel)