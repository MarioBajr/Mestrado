# -*- coding: utf-8 -*-

__author__ = 'Mario'

import process_samples as ps
import neural_network as nn

import Image
import numpy
from skimage import data, io, filter, feature
import cv2 as cv


from numpy.random import shuffle

if __name__ == '__main__':

    print "Processing Samples"
    #extract_lbp_hog
    (pos, neg) = ps.process_samples(cache_enabled=True, feature_extractor=ps.extract_hog_gabor)

    print "Split Samples"
    (train, test) = ps.split_classes(pos, neg, .7)
    (train_target, train_input) = ps.split_target_input(train)
    (test_target, test_input) = ps.split_target_input(test)

    # print "LVQ Training"
    # net = nn.run_lvq(train_input, train_target)
    #
    # print "LVQ Testing"
    # output = net.sim(test_input)
    # output = nn.target_2d_to_1d(output)
    # nn.confusion_matrix(output, test_target)

    # print "ELM Training"
    # nn.run_elm(train_input, train_target, test_input, test_target)

    print "SVM Training"
    nn.run_svm(train_input, train_target, test_input, test_target)