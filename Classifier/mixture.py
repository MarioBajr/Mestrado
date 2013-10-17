__author__ = 'Mario'

import os
import cv2 as cv
import neurolab as nl
import features as ft
import numpy as np
import neural_network as nn
import cascade_detect as cd
import process_samples as ps

cache_prefix = '../Databases/Temp/'

def enum(**enums):
    return type('Enum', (), enums)

ClassifierType = enum(LVQ='lvq', ELM='elm', SVM='svm')


class Classifier(object):

    def __init__(self, classifier_type, features, scale):
        self.classifier_type = classifier_type
        self.features = features
        self.scale = scale

    def run(self):
        pass


class LVQClassifier(Classifier):

    def __init__(self, classifier_type, features, scale):
        super(LVQClassifier, self).__init__(classifier_type, features, scale)
        self.net = self.__train()

    def __train(self):
        def add_distf(net):
            def euclidean(a, b):
                return np.sqrt(np.sum(np.square(a-b), axis=1))
            for layer in net.layers:
                layer.distf = euclidean

        def remove_distf(net):
            for layer in net.layers:
                layer.distf = None

        def load_net(file):
            net = None
            if os.path.isfile(file):
                net = nl.tool.load(file)
                add_distf(net)
            return net

        def save_net(file, net):
            remove_distf(net)
            net.save(file)
            add_distf(net)

        cache_file = '%slvq_%s_%s.net' % (cache_prefix, "_".join(self.features), self.scale)
        net = load_net(cache_file)

        if not net:
            (train_target, train_input, test_target, test_input) = ps.process_network_inputs(self.features, self.scale)
            net = nn.run_lvq(train_input, train_target, test_input, test_target)
            save_net(cache_file, net)
        return net

    def run(self, input):
        return self.net.sim(input)


class Mixture(object):

    def __init__(self, classifiers):
        self.classifiers = classifiers

    def run(self, im):

        def pyramidal(attrs, squares, features, scale):
            patterns = []
            for square in squares:
                patterns.append(ft.compose_features(square, features, scale))
            patterns = np.array(patterns)
            return attrs, squares, patterns

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

        attrs, squares = cd.pyramidal_scale(im, step_trans=32, step_scale=.3)

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
                    #cv.rectangle(dst, attrs[i][0], attrs[i][1], color, 1)

                    if all_results[i] == -1:
                        all_results[i] = attrs[i]
                else:
                    all_results[i] = None

        for result in all_results:
            if result:
                cv.rectangle(dst, result[0], result[1], (255, 255, 0), 2)

        return dst