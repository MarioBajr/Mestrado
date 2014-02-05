__author__ = 'Mario'

import os
import cv2 as cv
import test as ts
import numpy as np
import neurolab as nl
import features as ft
import neural_network as nn
import cascade_detect as cd
import process_samples as ps

from sklearn.externals import joblib

cache_prefix = '../Databases/Temp/net/'


class Classifier(object):

    def __init__(self, features, scale):
        self.features = features
        self.scale = scale
        self.net = self.train()
        #pca_path = ps.get_pca_path(features, scale)
        #self.pca = joblib.load(pca_path)

    def process_inputs(self):
        #return ts.test_samples(self.features, self.scale)
        return ps.process_network_inputs(self.features, self.scale)

    def train(self):
        return 0

    def load(self, f):
        pass

    def save(self, f, net):
        pass

    def run(self, x):
        #return self.pca.transform(x)
        return x


class LVQClassifier(Classifier):

    def __init__(self, features, scale):
        super(LVQClassifier, self).__init__(features, scale)

    def __add_distf(self, net):
        def euclidean(a, b):
            return np.sqrt(np.sum(np.square(a-b), axis=1))
        for layer in net.layers:
            layer.distf = euclidean

    def __remove_distf(self, net):
        for layer in net.layers:
            layer.distf = None

    def train(self):
        cache_file = '%slvq_%s_%s.net' % (cache_prefix, "_".join(self.features), self.scale)
        net = self.load(cache_file)

        if not net:
            (train_target, train_input, test_target, test_input) = self.process_inputs()
            net = nn.run_lvq(train_input, train_target, test_input, test_target)
            self.save(cache_file, net)
        return net

    def save(self, file, net):
        self.__remove_distf(net)
        net.save(file)
        self.__add_distf(net)

    def load(self, file):
        net = None
        if os.path.isfile(file):
            net = nl.tool.load(file)
            self.__add_distf(net)
        return net

    def run(self, x):
        x = super(LVQClassifier, self).run(x)
        print x.shape
        results = self.net.sim(x)
        results = nn.target_2d_to_1d(results)
        return results


class ELMClassifier(Classifier):
    def __init__(self, classifier_type, features, scale):
        self.classifier_type = classifier_type
        super(ELMClassifier, self).__init__(features, scale)

    def train(self):
        (train_target, train_input, test_target, test_input) = self.process_inputs()
        return nn.run_elm(self.classifier_type, train_input, train_target, test_input, test_target)

    def run(self, input):
        input = self.pca.transform(input)
        return self.net.predict(input)


class SVMClassifier(Classifier):
    def __init__(self, features, scale):
        super(SVMClassifier, self).__init__(features, scale)

    def train(self):
        cache_file = '%ssvm_%s_%s.net' % (cache_prefix, "_".join(self.features), self.scale)
        net = self.load(cache_file)

        if not net:
            (train_target, train_input, test_target, test_input) = self.process_inputs()
            net = nn.run_svm(train_input, train_target, test_input, test_target)
            self.save(cache_file, net)
        return net

    def save(self, f, net):
        joblib.dump(net, file, )

    def load(self, f):
        net = None
        if os.path.isfile(file):
            net = joblib.load(file)
        return net

    def run(self, x):
        x = super(SVMClassifier, self).run(x)
        return self.net.predict(x)


class AdaBoostClassifier(Classifier):
    def __init__(self, features, scale):
        super(AdaBoostClassifier, self).__init__(features, scale)

    def train(self):
        cache_file = '%sadaboost_%s_%s.net' % (cache_prefix, "_".join(self.features), self.scale)
        net = self.load(cache_file)

        if not net:
            (train_target, train_input, test_target, test_input) = self.process_inputs()
            net = nn.run_adaboost(train_input, train_target, test_input, test_target)
            self.save(cache_file, net)
        return net

    def save(self, file, net):
        joblib.dump(net, file)

    def load(self, file):
        net = None
        if os.path.isfile(file):
            net = joblib.load(file)
        return net

    def run(self, x):
        x = super(AdaBoostClassifier, self).run(x)
        return self.net.predict(x)


class Mixture(object):

    def __init__(self, classifiers):
        self.classifiers = classifiers

    def run(self, im):

        def pyramidal(attrs, squares, features, scale):
            patterns = []
            for square in squares:
                sqr = ps.scale_image(square, scale)
                patterns.append(ft.compose_features(sqr, features))
            return np.array(patterns)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

        attrs, squares = cd.pyramidal_scale(im, rect=(32, 32))

        all_results = [-1] * len(squares)

        dst = im.copy()
        for classifier in self.classifiers:
            print "Mixture Classifier", classifier.__class__
            patterns = pyramidal(attrs, squares, classifier.features, classifier.scale)

            print "Run", patterns.shape
            results = classifier.run(patterns)

            color = colors.pop()

            for i, result in enumerate(results):
                if result == 1:
                    draw_circle(dst, attrs[i][0], attrs[i][1], color, 1)
                    #cv.imwrite('../Databases/Temp/temp2/%s.png' % i, squares[i])
                    #cv.rectangle(dst, attrs[i][0], attrs[i][1], color, 1)
                    #cv.imwrite('../Databases/Temp/temp2/%s.png' % i, squares[i])

                    if all_results[i] == -1:
                        all_results[i] = attrs[i]
                else:
                    all_results[i] = None

        for result in all_results:
            if result:
                draw_circle(dst, result[0], result[1], (255, 255, 0), 2)
                #cv.rectangle(dst, result[0], result[1], (255, 255, 0), 2)
        return dst


def draw_circle(im, p0, p1, color, thickness=1):
    m = (p0[0]+((p1[0]-p0[0])/2.0), p0[1]+((p1[1]-p0[1])/2.0))

    m = (int(m[0]), int(m[1]))
    r = int(p1[0]-m[0])

    cv.circle(im, m, r, color, thickness)
    #cv.rectangle(dst, result[0], result[1], (255, 255, 0), 2)