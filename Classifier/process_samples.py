# -*- coding: utf-8 -*-

__author__ = 'Mario'

import cv2 as cv
import gabor_filter
import numpy as np
import os
import math

from skimage import data, io, filter, feature, color

def extract_hog_gabor(im):
    s = .25
    w = int(im.shape[0]*s)
    h = int(im.shape[1]*s)

    im_rsz = cv.resize(im, (h, w))
    im_gabor = gabor_filter(im_rsz)
    im_hog = feature.hog(im_gabor, visualise=False, normalise=False);
    return im_hog#np.reshape(im_gabor, -1)

def extract_lbp_hog(im):

    s = .5
    w = int(im.shape[0]*s)
    h = int(im.shape[1]*s)

    im_rsz = cv.resize(im, (h, w))

    im_gray = cv.cvtColor(im_rsz, cv.COLOR_RGBA2GRAY)
    im_lbp = feature.local_binary_pattern(im_gray, 8, 3)
    im_hog = feature.hog(im_lbp, visualise=False, normalise=False);

    return im_hog

def extract_lbp_gabor(im):
    s = .25
    w = int(im.shape[0]*s)
    h = int(im.shape[1]*s)

    im_rsz = cv.resize(im, (h, w))
    im_gabor = gabor_filter(im_rsz)
    im_lbp = feature.local_binary_pattern(im_gabor, 8, 3)
    return np.reshape(im_lbp, -1)

def gabor_filter(im):
    angles = 9
    lst = []
    for i, angle in [(a, math.degrees(a*math.pi/angles)) for a in range(angles)]:
        out = gabor_filter.Process(im, 5, 50, angle, 90)
        out *= 255
        lst.append(np.array(out))

    return np.vstack([np.hstack(lst[0:3]), np.hstack(lst[3:6]), np.hstack(lst[6:9])]);

def process_samples(cache_enabled=True, feature_extractor=extract_hog_gabor):

    pos_file_path = '../Databases/Temp/pos_%s' % feature_extractor.__name__
    neg_file_path = '../Databases/Temp/neg_%s' % feature_extractor.__name__

    if cache_enabled and os.path.isfile(pos_file_path+'.npy') and os.path.isfile(pos_file_path+'.npy'):
        pos_features = np.load(pos_file_path+'.npy')
        neg_features = np.load(neg_file_path+'.npy')

        print 'cache:', pos_features.shape, neg_features.shape

        return pos_features, neg_features

    pos_path = '../Databases/lfwcrop_color/faces'
    neg_path = '../Databases/INRIA/negatives'

    pos_features = features_from_images(pos_path, feature_extractor);
    neg_features = features_from_images(neg_path, feature_extractor);

    # cache results
    np.save(pos_file_path, pos_features)
    np.save(neg_file_path, neg_features)

    print 'processed:', pos_features.shape, neg_features.shape

    return pos_features, neg_features

def features_from_images(images_folder, feature_extractor):

    features = []

    files = os.listdir(images_folder)
    files = remove_invalid_images(files)
    # files = files[:2000]

    for file in files:
        im = cv.imread('%s/%s' % (images_folder, file))

        # Extract Feature
        feature = feature_extractor(im)
        features.append(feature);

    return np.array(features)

def remove_folder_content(folder):
    files = os.listdir(folder)
    for file in files:
        file_path = os.path.join(folder, file)
        os.remove(file_path)

def remove_invalid_images(files):
    not_image = set(['.DS_Store', '.png']) # the subset of A
    return [im for im in files if im not in not_image]

def split_classes(pos, neg, pc):

    (train_pos, test_pos) = split_samples(pos, pc)
    (train_neg, test_neg) = split_samples(neg, pc)

    print train_pos.shape, train_neg.shape
    print test_pos.shape, test_neg.shape

    train_pos = np.concatenate((np.ones((train_pos.shape[0], 1)), train_pos), axis=1)
    test_pos = np.concatenate((np.ones((test_pos.shape[0], 1)), test_pos), axis=1)

    train_neg = np.concatenate((np.zeros((train_neg.shape[0], 1)), train_neg), axis=1)
    test_neg = np.concatenate((np.zeros((test_neg.shape[0], 1)), test_neg), axis=1)

    print train_pos.shape, train_neg.shape
    print test_pos.shape, test_neg.shape

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
    # targets = samples[:, 0]
    # inputs = samples[:, 1:]
    #
    # list = []
    # for i in target_1d:
    #     if i == 0:
    #         list.append([1, 0])
    #     else:
    #         list.append([0, 1])
    # target = np.array(list)
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