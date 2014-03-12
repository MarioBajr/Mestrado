__author__ = 'Mario'

from skimage import feature
from sklearn.decomposition import ProbabilisticPCA

import math
import cv2 as cv
import numpy as np
import helper as hp
from operator import mul

################ NODE ################

class Pair(object):
    def __init__(self, clazz, a, b):
        self.clazz = clazz
        self.a = a
        self.b = b

    @staticmethod
    def score(p1, p2):
        d1 = np.linalg.norm(p1.a-p2.a) + np.linalg.norm(p1.b-p2.b)
        d2 = np.linalg.norm(p1.a-p2.b) + np.linalg.norm(p1.b-p2.a)
        return min(d1, d2)

    def __repr__(self):
        return "<Pair ("+str(self.clazz)+","+str(self.a.shape)+")"

################ Logic ################

PATH_LIST = '../Databases/lfwcrop_color/'
PATH_IMG  = '../Databases/lfw_funneled/'

QDT_DEBUG_LOAD = 100

def extract_features(im):
    return feature.local_binary_pattern(im, 8, 3)


def load_file(fname, clazz):
    f = open(fname, 'r')

    i = 0
    ret = []
    for line in f:
        # data = proccess_images(line.split())
        # if min(data.shape) > 0:
        #     ret.append(data)
        im1, im2 = proccess_images(line.split())
        ret.append(Pair(clazz, im1, im2))

        i+=1
        if i>=QDT_DEBUG_LOAD:
            break
    return ret

def proccess_images(imgs):
    path1 = PATH_IMG+imgs[0][0:-5]+'/'+imgs[0]+'.jpg'
    path2 = PATH_IMG+imgs[1][0:-5]+'/'+imgs[1]+'.jpg'

    im1 = cv.imread(path1)
    im1 = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
    im1 = im1[65:185, 85:165]#120
    pattr1 = extract_features(im1)

    im2 = cv.imread(path2)
    im2 = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)
    im2 = im2[65:185, 85:165]
    pattr2 = extract_features(im2)

    hp.showImage(im1)
    hp.showImage(im2)

    return pattr1, pattr2

def load_db(index):
    train_same = load_file(PATH_LIST+'lists/'+index+'_train_same.txt', 1)
    train_diff = load_file(PATH_LIST+'lists/'+index+'_train_diff.txt', 0)
    test_same = load_file(PATH_LIST+'lists/'+index+'_test_same.txt', 1)
    test_diff = load_file(PATH_LIST+'lists/'+index+'_test_diff.txt', 0)

    return train_same+train_diff, test_same+test_diff

# R: Train
# T: Test
# Time Complexity: R+T

def knn(train, test, k=5):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for t in test:

        score_train = [(Pair.score(p, t), p) for p in train]
        fit_sorted = sorted(score_train, key=lambda tuple: tuple[0])
        best_fit = fit_sorted[:k]
        score = sum(tuple[1].clazz for tuple in best_fit)
        # print score
        clazz = score > math.ceil(k/2.0)

        if clazz == t.clazz:
            if clazz == 1:
                tp += 1
            else:
                tn += 1
        else:
            if clazz == 1:
                fp += 1
            else:
                fn += 1

    print "train:", len(train), "shape:", reduce(mul,train[0].a.shape)
    print "test:", len(test), "shape:", reduce(mul,train[0].a.shape)
    print "tp:", tp, "tn:", tn
    print "fp:", fp, "fn:", fn
    print "score:", (tp + tn) / float(tp+tn+fp+fn)

if __name__ == '__main__':

    print "processing"
    train, test = load_db('01')

    print "classifing"
    knn(train, test, k=5)

    # cv.waitKey(0)