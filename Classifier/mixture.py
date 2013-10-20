__author__ = 'Mario'

import os
import cv2 as cv
import neurolab as nl
import features as ft
import numpy as np
import neural_network as nn
import cascade_detect as cd
import process_samples as ps

from sklearn.externals import joblib

cache_prefix = '../Databases/Temp/'

class Classifier(object):

    def __init__(self, features, scale):
        self.features = features
        self.scale = scale

    def load(self, file):
        pass

    def save(self, file, net):
        pass

    def run(self, input):
        pass


class LVQClassifier(Classifier):

    def __init__(self, features, scale):
        super(LVQClassifier, self).__init__(features, scale)
        self.net = self.__train()

    def _add_distf(self, net):
        def euclidean(a, b):
            return np.sqrt(np.sum(np.square(a-b), axis=1))
        for layer in net.layers:
            layer.distf = euclidean

    def _remove_distf(self, net):
        for layer in net.layers:
            layer.distf = None

    def __train(self):
        cache_file = '%slvq_%s_%s.net' % (cache_prefix, "_".join(self.features), self.scale)
        net = self.load(cache_file)

        if not net:
            (train_target, train_input, test_target, test_input) = ps.process_network_inputs(self.features, self.scale)
            net = nn.run_lvq(train_input, train_target, test_input, test_target)
            self.save(cache_file, net)
        return net

    def save(self, file, net):
        self._remove_distf(net)
        net.save(file)
        self._add_distf(net)

    def load(self, file):
        net = None
        if os.path.isfile(file):
            net = nl.tool.load(file)
            self.add_distf(net)
        return net

    def run(self, input):
        return self.net.sim(input)


class AdaBoostClassifier(Classifier):
    def __init__(self, features, scale):
        super(AdaBoostClassifier, self).__init__(features, scale)
        self.net = self.__train()

    def __train(self):
        cache_file = '%sadaboost_%s_%s.net' % (cache_prefix, "_".join(self.features), self.scale)
        net = self.load(cache_file)

        if not net:
            (train_target, train_input, test_target, test_input) = ps.process_network_inputs(self.features, self.scale)
            net = nn.run_adaboost(train_input, train_target, test_input, test_target)
            self.save(cache_file, net)
        return net

    def save(self, file, net):
        joblib.dump(net, file, )

    def load(self, file):
        net = None
        if os.path.isfile(file):
            net = joblib.load(file)
        return net

class Mixture(object):

    def __init__(self, classifiers):
        self.classifiers = classifiers

    def run(self, im):

        def pyramidal(attrs, squares, features, scale):
            patterns = []
            for square in squares:
                sqr = ps.reduce_image(square, scale)
                patterns.append(ft.compose_features(sqr, features))
            patterns = np.array(patterns)
            return attrs, squares, patterns

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

        attrs, squares = cd.pyramidal_scale(im, step_trans=64)

        all_results = [-1] * len(squares)

        dst = im.copy()
        for classifier in self.classifiers:
            print "Mixture Classifier", classifier.__class__
            attrs, squares, patterns = pyramidal(attrs, squares, classifier.features, classifier.scale)

            print "Run", patterns.shape
            results = classifier.run(patterns)
            results = nn.target_2d_to_1d(results)

            color = colors.pop()

            for i, result in enumerate(results):
                if result == 1:

                    cv.imwrite('../Databases/Temp/temp2/%s.png' % i, squares[i])
                    #cv.rectangle(dst, attrs[i][0], attrs[i][1], color, 1)

                    if all_results[i] == -1:
                        all_results[i] = attrs[i]
                else:
                    all_results[i] = None

        for result in all_results:
            if result:
                p0 = result[0]
                p1 = result[1]

                m = (p0[0]+((p1[0]-p0[0])/2.0), p0[1]+((p1[1]-p0[1])/2.0))

                m = (int(m[0]), int(m[1]))
                r = int(p1[0]-m[0])

                cv.circle(dst, m, r, (255, 255, 0), 1)
                #cv.rectangle(dst, result[0], result[1], (255, 255, 0), 2)

        return dst