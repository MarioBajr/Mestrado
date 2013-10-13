# -*- coding: utf-8 -*-

__author__ = 'Mario'

import process_samples as ps
import neural_network as nn
import features as ft
import cascade_detect as cd

import cv2 as cv
import numpy as np
import os
import math

s = .5

configurations = [
    # [[ft.Features.GABOR, ft.Features.HOG], .25],
    [[ft.Features.LBP, ft.Features.GABOR], .25],
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
        print "train:", train.shape, "test:", test.shape
    else:
        (pos, neg) = ps.process_samples(features=features, scale=scale)

        print "Split Samples"
        (train, test) = ps.split_classes(pos, neg, .7)
        np.save(train_file_path, train)
        np.save(test_file_path, test)

    return train, test


def run_fddb(features, scale, detector):

    print "Pyramidal"
    im = cv.imread('../Databases/Temp/test/tank.jpg')
    attrs, squares = cd.pyramidal_scale(im)

    ps.remove_folder_content('../Databases/Temp/debug/tank/')

    pattrns = []
    for square in squares:
        pattrns.append(ft.compose_features(square, features, scale))

        #out = pattrns[-1]
        #d = int(math.sqrt(out.shape[0]))
        #out = np.reshape(out, (d, -1))
        #cv.imwrite('../Databases/Temp/debug/tank/tank_%s.png' % len(pattrns), square)

    pattrns = np.array(pattrns)

    results = detector(pattrns)
    results = nn.target_2d_to_1d(results)

    print results.sum()

    dst = im.copy()

    for i, result in enumerate(results):
        #if result == 1:
        import random
        if random.random() > 0.9:
            cv.rectangle(dst, attrs[i][0], attrs[i][1], (255, 0, 0), 1)
            #cv.imwrite('../Databases/Temp/output/%s.jpg' % i, squares[i])

    cv.imwrite('../Databases/Temp/test/realout.jpg', dst)


def test_1(features, scale, detector):

    path = '../Databases/Temp/output/'
    pattrns = ps.features_from_images(path, features, scale)
    results = detector(pattrns)
    print results
    results = nn.target_2d_to_1d(results)

    files = os.listdir(path)
    files = ps.remove_invalid_images(files)

    for i, result in enumerate(results):
        if result == 1:
            #f = files[i]
            #im = cv.imread('%s/%s' % (path, f))

            im = pattrns[i]
            d = math.sqrt(im.shape[0])
            im = np.reshape(im, (d, -1))
            cv.imwrite('../Databases/Temp/output2/%s.jpg' % i, im)


def test_2(features, scale):

    pos_path = '../Databases/lfwcrop_color/faces'
    neg_path = '../Databases/INRIA/negatives'

    def extract_features(path, folder):
        pattrns = ps.features_from_images(path, features, scale)

        files = os.listdir(path)
        files = ps.remove_invalid_images(files)
        files = files[:100]

        m = len(files)

        for i, f in enumerate(files):
            im = pattrns[i]
            d = int(math.sqrt(im.shape[0]))
            im1 = np.reshape(im, (d, -1))
            im1 = cv.cvtColor(im1, cv.COLOR_GRAY2RGB)

            im2 = cv.imread('%s/%s' % (path, f))
            im2 = cv.resize(im2, (d, d))

            im3 = np.concatenate((im1, im2), axis=1)

            cv.imwrite('../Databases/Temp/debug/%s/%s.jpg' % (folder, i), im3)

            print i, '/', m, (i/float(m))

    extract_features(pos_path, "pos")
    extract_features(neg_path, "neg")


def test_3(im, pattern):
    d = int(math.sqrt(pattern.shape[0]))
    im1 = np.reshape(pattern, (d, -1))
    im1 = cv.cvtColor(im1, cv.COLOR_GRAY2RGB)

    im2 = im
    im2 = cv.resize(im2, (d, d))

    im3 = np.concatenate((im1, im2), axis=1)

    print d, im.shape, pattern.shape
    print im1.shape, im2.shape, im3.shape

    return im3


def test_filter():
    im = cv.imread('../Databases/lfwcrop_color/faces/Aaron_Sorkin_0002.png')
    cv.imwrite('out.jpg', im)

    ftr = ft.surf_feature(im)

    cv.imwrite('out2.jpg', ftr)


if __name__ == '__main__':

    # test_filter()

    for configuration in configurations:
        print "Configuration:", configuration
        run_classifiers(features=configuration[0], scale=configuration[1])

        #test_2(features=configuration[0], scale=configuration[1])