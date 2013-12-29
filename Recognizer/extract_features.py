__author__ = 'Mario'

import glob
import png
import numpy as np
import re
import cv2 as cv


def read_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                        dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                        count=int(width)*int(height),
                        offset=len(header)).reshape((int(height), int(width)))


def load_db():

    pattern = '../Databases/orl_faces/*/*.pgm'
    files = glob.glob(pattern)
    files = files[:10]

    persons = []

    for file in files:
        pieces = files.split("/")

    # patterns = []
    #         for square in squares:
    #             sqr = ps.scale_image(square, scale)
    #             patterns.append(ft.compose_features(sqr, features))