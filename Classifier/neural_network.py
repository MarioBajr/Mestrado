
# -*- coding: utf-8 -*-

__author__ = 'Mario'

import neurolab as nl
import numpy as np
import process_samples as ps

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from elm import ELMClassifier
from random_hidden_layer import SimpleRandomHiddenLayer, RBFRandomHiddenLayer

def run_lvq(train_input, train_target):
    #Convert Target Dimension
    train_target = target_1d_to_2d(train_target)

    #Train Network
    net = nl.net.newlvq(nl.tool.minmax(train_input), 30, [.5, .5])
    error = net.train(train_input, train_target, epochs=100, goal=-1, show=10, adapt=True)

    #Plot Results
#    import  pylab as pl
#    pl.subplot(211)
#    pl.plot(error)
#    pl.xlabel(u'Número de Épocas')
#    pl.ylabel(u'Erro (MSE)')
#    pl.show()

#    nl.tool.save(net, 'net')
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
    return targets[:,0]

def confusion_matrix(results, targets):
    results_2d = target_1d_to_2d(results)
    targets_2d = target_1d_to_2d(targets)

    tp = np.logical_and(targets_2d[:, 0], results_2d[:, 0]).sum()
    fn = np.logical_and(targets_2d[:, 1], results_2d[:, 1]).sum()
    tn = np.logical_and(targets_2d[:, 0], results_2d[:, 1]).sum()
    fp = np.logical_and(targets_2d[:, 1], results_2d[:, 0]).sum()
    p = tp + fn
    n = fp + tn

    print 'True Positives: ', tp
    print 'False Negatives: ', fn
    print 'True Negatives: ', tn
    print 'False Positives: ', fp

    print 'Accuracy', (tp+tn)/float(p+n)
    print 'Specificity', (tp+tn)/float(p+n)
    print 'Precision', tp/float(tp+fp)

    print 'Rate: ', (tp+fn)/float(p+n)

def roc_curve(results, targets):
    curve = np.array([])
    tp = 0
    fp = 0

    p = targets[targets[:, 0]==1, 0].shape[0]
    n = targets[targets[:, 0]==0, 0].shape[0]

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

    names = ["ELM(10,tanh)", "ELM(10,tanh,LR)", "ELM(10,sinsq)",
             "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]

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

def run_elm(train_input, train_target, test_input, test_target):
    for clf in make_classifiers():
        clf.fit(train_input, train_target)

        score = clf.score(test_input, test_target)
        print score

def run_svm(train_input, train_target, test_input, test_target):
    clf = svm.SVC()
    clf.fit(train_input, train_target)

    output = clf.predict(test_input)

    confusion_matrix(output, test_target)

