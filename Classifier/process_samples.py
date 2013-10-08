# -*- coding: utf-8 -*-

__author__ = 'Mario'

import cv2 as cv
import numpy as np
import os

import features as ft


def process_samples(features, scale):

    print features
    #pos_path = '../Databases/lfwcrop_color/faces'
    #neg_path = '../Databases/INRIA/negatives'

    pos_path = '../Databases/Old/Positives'
    neg_path = '../Databases/Old/Negatives'

    pos_features = features_from_images(pos_path, features, scale)
    neg_features = features_from_images(neg_path, features, scale)
    print 'processed:', pos_features.shape, neg_features.shape

    return pos_features, neg_features


def features_from_images(images_folder, features, scale):

    patterns = []

    files = os.listdir(images_folder)
    files = remove_invalid_images(files)

    for f in files:
        im = cv.imread('%s/%s' % (images_folder, f))

        # Extract Feature
        pattern = ft.compose_features(im, features, scale)
        patterns.append(pattern);

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

# def pyramidal_scale(im):
#     D = 10
#
#     IH, IW, ID = im.shape
#     print IH, IW, ID
#
#     attributes = []
#     patterns = []
#
#     list = convert_sample(im)
#     for s in range(1, 2):
#         h = IH/float(H)
#         w = IW/float(W)
#         for i in range(0, IH-H, D):
#             for j in range(0, IW-W, D):
#                 pattern = np.array([])
#                 for out in list:
#                     window = out[i:i+H, j:j+W]
#                     pattern = np.append(pattern, window.reshape(-1))
#                 patterns.append(pattern)
#                 attributes.append([s, (j, i), (j+W, i+H)])
#             break
#
#     return (attributes, patterns)