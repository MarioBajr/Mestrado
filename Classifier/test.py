__author__ = 'Mario'

import numpy as np
import process_samples as ps
from sklearn.externals import joblib
from sklearn.decomposition import RandomizedPCA

cache_db = '../Databases/Temp/db_test/%s'
cache_file = '../Databases/Temp/net/'


def test_samples(features, scale):
    train_pos_features = ps.features_from_images(cache_db % 'train_pos', features, scale, variations=1)
    train_neg_features = ps.features_from_images(cache_db % 'train_neg', features, scale)
    test_pos_features = ps.features_from_images(cache_db % 'test_pos', features, scale, variations=1)
    test_neg_features = ps.features_from_images(cache_db % 'test_neg', features, scale)

    n_components = 150
    all = np.concatenate((train_pos_features, train_neg_features, test_pos_features, test_neg_features), axis=0)
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(all)
    train_pos = pca.transform(train_pos_features)
    train_neg = pca.transform(train_neg_features)
    test_pos = pca.transform(test_pos_features)
    test_neg = pca.transform(test_neg_features)

    features_name = "_".join(features)
    pca_file_path = '%spca_%s_%s' % (cache_file, features_name, scale)
    joblib.dump(pca, pca_file_path)

    train_pos = np.concatenate((np.ones((train_pos.shape[0], 1)), train_pos), axis=1)
    test_pos = np.concatenate((np.ones((test_pos.shape[0], 1)), test_pos), axis=1)

    train_neg = np.concatenate((np.zeros((train_neg.shape[0], 1)), train_neg), axis=1)
    test_neg = np.concatenate((np.zeros((test_neg.shape[0], 1)), test_neg), axis=1)

    train = np.concatenate((train_pos, train_neg), axis=0)
    train = np.random.permutation(train)

    test = np.concatenate((test_pos, test_neg), axis=0)
    test = np.random.permutation(test)

    (train_target, train_input) = ps.split_target_input(train)
    (test_target, test_input) = ps.split_target_input(test)

    return train_target, train_input, test_target, test_input
