
# -*- coding: utf-8 -*-

__author__ = 'Mario'

import os
import neurolab as nl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from elm.elm import ELMClassifier
from elm.random_hidden_layer import SimpleRandomHiddenLayer, RBFRandomHiddenLayer


def enum(**enums):
    return type('Enum', (), enums)


ELM_Type = enum(ELM_tan=0,
                ELM_tan_lr=1,
                ELM_sin=2,
                ELM_tribas=3,
                ELM_hardlim=4,
                ELM_rbf=5)

def run_lvq(train_input, train_target, test_input, test_target):

    #Convert Target Dimension
    train_target = target_1d_to_2d(train_target)

    p = train_target[:, 0].sum()/float(train_target.shape[0])

    #Train Network
    net = nl.net.newlvq(nl.tool.minmax(train_input), 30, [p, 1-p])
    error = net.train(train_input, train_target, epochs=100, goal=.001, show=10, adapt=True)

    #Plot Results
#    import  pylab as pl
#    pl.subplot(211)
#    pl.plot(error)
#    pl.xlabel(u'Número de Épocas')
#    pl.ylabel(u'Erro (MSE)')
#    pl.show()

    output = net.sim(test_input)
    output = target_2d_to_1d(output)
    confusion_matrix(output, test_target)

    return net

def run_ff(train_input, train_target):
    #Convert Target Dimension
    train_target = target_1d_to_2d(train_target)

    #Train Network
    net = nl.net.newff(nl.tool.minmax(train_input), [500, 500, 500, 2])
#    net.errorf = nl.error.MSE()
#    net.trainf = nl.train.train_bfgs
    error = net.train(train_input, train_target, epochs=100, goal=-1, show=10)
    return net


def run_er(train_input, train_target):
    #Convert Target Dimension
    train_target = target_1d_to_2d(train_target)

    #Train Network
    net = nl.net.newelm(nl.tool.minmax(train_input), [100, 2], [nl.trans.TanSig(), nl.trans.PureLin()])
    net.train(train_input, train_target, epochs=100, goal=-1, show=10)

    return net


def target_1d_to_2d(targets):
    pos = targets
    neg = np.logical_not(targets)
    return np.transpose(np.vstack((pos,neg)))


def target_2d_to_1d(targets):
    return targets[:, 0]


def confusion_matrix(results, targets):
    results_2d = target_1d_to_2d(results)
    targets_2d = target_1d_to_2d(targets)

    tp = np.logical_and(targets_2d[:, 0], results_2d[:, 0]).sum()
    tn = np.logical_and(targets_2d[:, 1], results_2d[:, 1]).sum()
    fn = np.logical_and(targets_2d[:, 0], results_2d[:, 1]).sum()
    fp = np.logical_and(targets_2d[:, 1], results_2d[:, 0]).sum()
    p = tp + fn
    n = fp + tn

    print 'TP: ', tp,
    print 'TN: ', tn
    print 'FN: ', fn,
    print 'FP: ', fp
    print ''
    print 'Sensitivity', tp/float(tp+fn)
    print 'Specificity', tn/float(fp+tn)
    print 'PPV:', tp/float(tp+fp), 'PPN:', tn/float(tn+fn)
    print ''
    print 'Rate: ', (tp+tn)/float(p+n)
    print 'TP:', tp, 'TN:', tn, 'FP:', fp, 'FN:', fn, '(',  ((tp+tn)/float(p+n)*100), '%)'
    print '---------------------------------'

def roc_curve(results, targets):
    curve = np.array([])
    tp = 0
    fp = 0

    p = targets[targets[:, 0] == 1, 0].shape[0]
    n = targets[targets[:, 0] == 0, 0].shape[0]

    t = results.shape[0]
    for i in range(t):
        if results[i, 0] == 1:
            if targets[i, 0] == 1:
                tp += 1
            else:
                fp += 1
        curve = np.append(curve, [fp/float(n), tp/float(p)])

    curve = np.append(curve, [1, 1])

    print tp, fp, p, n
    return curve.reshape((-1, 2))

# ELM


def make_classifiers():
    nh = 10

    # pass user defined transfer func
    sinsq = (lambda x: np.power(np.sin(x), 2.0))
    srhl_sinsq = SimpleRandomHiddenLayer(n_hidden=nh,
                                         activation_func=sinsq,
                                         random_state=0)

    # use internal transfer funcs
    srhl_tanh = SimpleRandomHiddenLayer(n_hidden=nh,
                                        activation_func='tanh',
                                        random_state=0)

    srhl_tribas = SimpleRandomHiddenLayer(n_hidden=nh,
                                          activation_func='tribas',
                                          random_state=0)

    srhl_hardlim = SimpleRandomHiddenLayer(n_hidden=nh,
                                           activation_func='hardlim',
                                           random_state=0)

    # use gaussian RBF
    srhl_rbf = RBFRandomHiddenLayer(n_hidden=nh*2, gamma=0.1, random_state=0)

    log_reg = LogisticRegression()

    classifiers = [ELMClassifier(srhl_tanh),
                   ELMClassifier(srhl_tanh, regressor=log_reg),
                   ELMClassifier(srhl_sinsq),
                   ELMClassifier(srhl_tribas),
                   ELMClassifier(srhl_hardlim),
                   ELMClassifier(srhl_rbf)]
    return classifiers


def run_elm(classifier_type, train_input, train_target, test_input, test_target):

    classifiers = make_classifiers()
    clf = classifiers[classifier_type]

    clf.fit(train_input, train_target)

    output = clf.predict(test_input)
    confusion_matrix(output, test_target)

    # score = clf.score(test_input, test_target)
    # print score
    return clf


def run_svm(train_input, train_target, test_input, test_target):
    clf = svm.SVC()
    clf.fit(train_input, train_target)

    output = clf.predict(test_input)
    confusion_matrix(output, test_target)
    return clf


def run_adaboost(train_input, train_target, test_input, test_target):
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             algorithm="SAMME",
                             n_estimators=200)
    clf.fit(train_input, train_target)

    output = clf.predict(test_input)
    confusion_matrix(output, test_target)
    return clf