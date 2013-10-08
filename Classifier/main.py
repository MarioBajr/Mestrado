# -*- coding: utf-8 -*-

__author__ = 'Mario'

import process_samples as ps
import neural_network as nn
import features as ft
import cascade_detect as cd

import cv2 as cv
import numpy as np
import os

s = .5

configurations = [
    # [[ft.Features.GABOR, ft.Features.HOG], .25],
    [[ft.Features.LBP, ft.Features.GABOR], 1],#95
    # [[ft.Features.LBP, ft.Features.HOG], .5],
    # [[ft.Features.HARRIS, ft.Features.HOG], .5],
    # [[ft.Features.DAISY], .5],
    # [[ft.Features.SHI_TOMASI, ft.Features.HOG], .5]

    # [[ft.Features.SHI_TOMASI, ft.Features.FAST], s],
    # [[ft.Features.SHI_TOMASI, ft.Features.STAR], s],
    # [[ft.Features.SHI_TOMASI, ft.Features.SIFT], s],
    # [[ft.Features.SHI_TOMASI, ft.Features.SURF], s],
    # [[ft.Features.SHI_TOMASI, ft.Features.ORB], s],
    # [[ft.Features.SHI_TOMASI, ft.Features.MSER], s],
    # [[ft.Features.SHI_TOMASI, ft.Features.GFTT], s],
    #
    # [[ft.Features.GABOR, ft.Features.GridFAST], s],
    # [[ft.Features.GABOR, ft.Features.GridSTAR], s],
    # [[ft.Features.GABOR, ft.Features.GridSIFT], s],
    # [[ft.Features.GABOR, ft.Features.GridSURF], s],
    # [[ft.Features.GABOR, ft.Features.GridORB], s],
    # [[ft.Features.GABOR, ft.Features.GridMSER], s],
    # [[ft.Features.GABOR, ft.Features.GridGFTT], s],
    # [[ft.Features.GABOR, ft.Features.GridHARRIS], s],
    #
    # [[ft.Features.GABOR, ft.Features.PyramidFAST], s],
    # [[ft.Features.GABOR, ft.Features.PyramidSTAR], s],
    # [[ft.Features.GABOR, ft.Features.PyramidSIFT], s],
    # [[ft.Features.GABOR, ft.Features.PyramidSURF], s],
    # [[ft.Features.GABOR, ft.Features.PyramidORB], s],
    # [[ft.Features.GABOR, ft.Features.PyramidMSER], s],
    # [[ft.Features.GABOR, ft.Features.PyramidGFTT], s],
    # [[ft.Features.GABOR, ft.Features.PyramidHARRIS], s]
]


def run_classifiers(features, scale):
    cache_enabled = True

    print "Processing Samples"
    (train, test) = process_samples(features, scale, cache_enabled)

    print "Split Samples"
    (train_target, train_input) = ps.split_target_input(train)
    (test_target, test_input) = ps.split_target_input(test)

    print "LVQ Training"
    net = nn.run_lvq(train_input, train_target, test_input, test_target)
    run_fddb(features=features, scale=scale, detector=net.sim)

    # print "ELM Training"
    # nn.run_elm(train_input, train_target, test_input, test_target)

    # print "SVM Training"
    # nn.run_svm(train_input, train_target, test_input, test_target)


def process_samples(features, scale, cache_enabled=True):
    train_file_path = '../Databases/Temp/train_%s' % '_'.join(features)
    test_file_path = '../Databases/Temp/test_%s' % '_'.join(features)

    if cache_enabled and os.path.isfile(train_file_path+'.npy') and os.path.isfile(test_file_path+'.npy'):
        print "Loading Cache"
        train = np.load(train_file_path+'.npy')
        test = np.load(test_file_path+'.npy')
    else:
        (pos, neg) = ps.process_samples(features=features, scale=scale)

        print "Split Samples"
        (train, test) = ps.split_classes(pos, neg, .7)
        np.save(train_file_path, train)
        np.save(test_file_path, test)
        del pos
        del neg

    return train, test


def run_fddb(features, scale, detector):

    im = cv.imread('../Databases/Temp/out.jpg')
    attrs, squares = cd.pyramidal_scale(im)

    pattrns = []
    for square in squares:
        pattrns.append(ft.compose_features(square, features, scale))

    pattrns = np.array(pattrns)

    print pattrns.shape
    results = detector(pattrns)
    print results[:,0]
    results = nn.target_2d_to_1d(results)

    dst = im.copy()

    for i, result in enumerate(results):
        if result == 1:
            cv.rectangle(dst, attrs[i][0], attrs[i][1], (255, 0, 0), 1)
            cv.imwrite('../Databases/Temp/output/%s.jpg' % i, squares[i])

    cv.imwrite('../Databases/Temp/realout.jpg', dst)


def test_filter():
    im = cv.imread('../Databases/lfwcrop_color/faces/Aaron_Sorkin_0002.png')
    cv.imwrite('out.jpg', im)

    ftr = ft.surf_feature(im)

    cv.imwrite('out2.jpg', ftr)


if __name__ == '__main__':

    # test_filter()

    for configuration in configurations:
        print configuration
        run_classifiers(features=configuration[0], scale=configuration[1])