# -*- coding: utf-8 -*-

__author__ = 'Mario'

import os
import cv2 as cv
import numpy as np
import features as ft
import random
import math

from sklearn.externals import joblib
from sklearn.decomposition import RandomizedPCA

cache_file = '../Databases/Temp/net/'


def process_samples(features, scale):

    pos_path = '../Databases/lfwcrop_color/faces'
    neg_path = '../Databases/INRIA/negatives'

    pos_features = features_from_images(pos_path, features, scale, variations=5)
    neg_features = features_from_images(neg_path, features, scale)
    print 'processed:', pos_features.shape, neg_features.shape

    return pos_features, neg_features


def scale_image(im, s):
    h = int(im.shape[0]*s)
    w = int(im.shape[1]*s)
    return cv.resize(im, (h, w))


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[:2])/2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)

    s = 0.95
    p = image.shape[0]*s
    result = cv.warpAffine(image, rot_mat, (int(p), int(p)), flags=cv.INTER_LINEAR)

    d = int((image.shape[0]-p)/1)

    result = result[d:d+p, d:d+p]

    return result


def degree_to_radians(ang):
    return ang * (math.pi/180)


def radians_to_degree(ang):
    return ang * (180/math.pi)


def random_between(min, max):
    r = random.random()
    return (max-min)*r + min


def generate_variations(im, q):
    results = [im]
    for i in range(q-1):
        ang = random_between(-10, 10)
        out = rotate_image(im, ang)
        scale = random_between(.9, 1.1)
        out = scale_image(out, scale)
        if random.random() > .5:
            out = cv.flip(out, flipCode=1)
        results.append(out)
    return results


def features_from_images(images_folder, features, scale, variations=1):
    patterns = []

    files = os.listdir(images_folder)
    files = remove_invalid_images(files)
    files = files[:2000/variations]

    for f in files:
        im = cv.imread('%s/%s' % (images_folder, f))
        items = generate_variations(im, variations)
        for item in items:
            # Extract Feature
            var = cv.resize(item, (int(im.shape[0]*scale), int(im.shape[1]*scale)))
            pattern = ft.compose_features(var, features)
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


def get_pca_path(features, scale):
    features_name = "_".join(features)
    return '%spca_%s_%s' % (cache_file, features_name, scale)


def create_pca(pca_file_path, pos, neg):
    n_components = 150
    all = np.concatenate((pos, neg), axis=0)
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(all)
    joblib.dump(pca, pca_file_path)

    return pca


def process_network_inputs(features, scale):
    print features
    features_name = "_".join(features)
    train_file_path = '%strain_%s_%s' % (cache_file, features_name, scale)
    test_file_path = '%stest_%s_%s' % (cache_file, features_name, scale)

    if os.path.isfile(train_file_path+'.npy') and os.path.isfile(test_file_path+'.npy'):
        print "Loading Cache"
        train = np.load(train_file_path+'.npy')
        test = np.load(test_file_path+'.npy')
    else:
        print "Processing Samples"
        (pos, neg) = process_samples(features=features, scale=scale)

        print "PCA"
        #pca = create_pca(get_pca_path(features, scale), pos, neg)
        #pos = pca.transform(pos)
        #neg = pca.transform(neg)

        print "Split Samples"
        (train, test) = split_classes(pos, neg, .7)

        print "Storing Cache"
        np.save(train_file_path, train)
        np.save(test_file_path, test)

    (train_target, train_input) = split_target_input(train)
    (test_target, test_input) = split_target_input(test)

    print "train:", train.shape, "test:", test.shape
    return train_target, train_input, test_target, test_input