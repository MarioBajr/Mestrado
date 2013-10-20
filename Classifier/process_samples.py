# -*- coding: utf-8 -*-

__author__ = 'Mario'

import cv2 as cv
import numpy as np
import os

import features as ft


def process_samples(features, scale):

    pos_path = '../Databases/lfwcrop_color/faces'
    neg_path = '../Databases/INRIA/negatives'

    pos_features = features_from_images(pos_path, features, scale)
    neg_features = features_from_images(neg_path, features, scale)
    print 'processed:', pos_features.shape, neg_features.shape

    return pos_features, neg_features


def reduce_image(im, s):
    h = int(im.shape[0]*s)
    w = int(im.shape[1]*s)
    return cv.resize(im, (h, w))


def features_from_images(images_folder, features, scale):
    patterns = []

    files = os.listdir(images_folder)
    files = remove_invalid_images(files)
    files = files[:3000]

    for f in files:
        im = cv.imread('%s/%s' % (images_folder, f))

        # Extract Feature

        im = reduce_image(im, scale)
        pattern = ft.compose_features(im, features)
        patterns.append(pattern)

    return np.array(patterns)


def remove_folder_content(folder):
    files = os.listdir(folder)
    for f in files:
        file_path = os.path.join(folder, f)
        os.remove(file_path)


def remove_invalid_images(files):
    not_image = set(['.DS_Store', '.png'])  # the subset of A
    return [im for im in files if im not in not_image]


def split_classes(pos, neg, pc):

    pos = np.random.permutation(pos)
    neg = np.random.permutation(neg)

    (train_pos, test_pos) = split_samples(pos, pc)
    (train_neg, test_neg) = split_samples(neg, pc)

    train_pos = np.concatenate((np.ones((train_pos.shape[0], 1)), train_pos), axis=1)
    test_pos = np.concatenate((np.ones((test_pos.shape[0], 1)), test_pos), axis=1)

    train_neg = np.concatenate((np.zeros((train_neg.shape[0], 1)), train_neg), axis=1)
    test_neg = np.concatenate((np.zeros((test_neg.shape[0], 1)), test_neg), axis=1)

    train = np.concatenate((train_pos, train_neg), axis=0)
    train = np.random.permutation(train)

    test = np.concatenate((test_pos, test_neg), axis=0)
    test = np.random.permutation(test)

    return train, test


def split_samples(samples, pc):
    random = np.random.permutation(samples)
    train_size = int(random.shape[0] * pc)

    train = random[:train_size, :]
    test = random[train_size:random.shape[0], :]

    return train, test


def split_target_input(samples):
    return samples[:, 0], samples[:, 1:]


def process_network_inputs(features, scale):

    print features
    features_name = "_".join(features)
    train_file_path = '../Databases/Temp/train_%s_%s' % (features_name, scale)
    test_file_path = '../Databases/Temp/test_%s_%s' % (features_name, scale)

    if os.path.isfile(train_file_path+'.npy') and os.path.isfile(test_file_path+'.npy'):
        print "Loading Cache"
        train = np.load(train_file_path+'.npy')
        test = np.load(test_file_path+'.npy')
    else:
        print "Processing Samples"
        (pos, neg) = process_samples(features=features, scale=scale)

        print "Split Samples"
        (train, test) = split_classes(pos, neg, .7)

        print "Storing Cache"
        np.save(train_file_path, train)
        np.save(test_file_path, test)

    (train_target, train_input) = split_target_input(train)
    (test_target, test_input) = split_target_input(test)

    print "train:", train.shape, "test:", test.shape
    return train_target, train_input, test_target, test_input