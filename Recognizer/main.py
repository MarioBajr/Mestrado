__author__ = 'Mario'

from config import *
import cv2 as cv
import numpy as np
import neurolab as nl

import face
import neural_network as nn
import feature_match as fm

PATH_LIST = '../Databases/lfwcrop_color/'
PATH_IMG  = '../Databases/lfw_funneled/'

def load_db(index):
    print "same"
    train_same = load_train_file(PATH_LIST+'lists/'+index+'_train_same.txt')

    print "diff"
    train_diff = load_train_file(PATH_LIST+'lists/'+index+'_train_diff.txt')

    print "same"
    test_same = load_test_file(PATH_LIST+'lists/'+index+'_test_same.txt')

    print "diff"
    test_diff = load_test_file(PATH_LIST+'lists/'+index+'_test_diff.txt')

    print "train - same", train_same.shape, "train - diff", train_diff.shape

    np.set_printoptions(precision=3)
    print train_same
    print "---------"
    print train_diff

    train_same = np.concatenate((np.ones((train_same.shape[0], 1)), train_same), axis=1)
    train_diff = np.concatenate((np.zeros((train_diff.shape[0], 1)), train_diff), axis=1)

    # test_same = np.concatenate((np.ones((test_same.shape[0], 1)), test_same), axis=1)
    # test_diff = np.concatenate((np.zeros((test_diff.shape[0], 1)), test_diff), axis=1)

    train = np.concatenate((train_same, train_diff), axis=0)
    train = np.random.permutation(train)

    # test = np.concatenate((test_same, test_diff), axis=0)

    (train_target, train_input) = split_target_input(train)
    # (test_target, test_input) = split_target_input(test)

    return train_target, train_input, test_same, test_diff

def split_target_input(samples):
    return samples[:, 0], samples[:, 1:]

def load_train_file(fname):
    f = open(fname, 'r')

    i = 0
    ret = []
    for line in f:
        data = proccess_images(line.split())
        if min(data.shape) > 0:
            ret.append(data)
        i+=1
        if i>=QDT_DEBUG_LOAD:
            break

    return np.array(ret)

def load_test_file(fname):
    f = open(fname, 'r')

    i = 0
    ret = []
    for line in f:
        data = proccess_images(line.split())
        if min(data.shape) > 0:
            ret.append(data)
        i+=1
        if i>=QDT_DEBUG_LOAD:
            break

    return np.array(ret)

def proccess_images(imgs):

    path1 = PATH_IMG+imgs[0][0:-5]+'/'+imgs[0]+'.jpg'
    path2 = PATH_IMG+imgs[1][0:-5]+'/'+imgs[1]+'.jpg'

    image1 = cv.imread(path1)
    image1 = image1[65:185, 65:185]#120

    image2 = cv.imread(path2)
    image2 = image2[65:185, 65:185]

    cv.imwrite('~/Desktop/'+imgs[0]+'.png', image1)
    cv.imwrite('~/Desktop/'+imgs[1]+'.png', image2)

    # faceImage1 = face.FaceImage(path1)
    # faceImage2 = face.FaceImage(path2)
    #
    # faceImage1.cropToFace()
    # faceImage2.cropToFace()
    #
    # faceImage1.save('~/Desktop/'+imgs[0]+'.png')
    # faceImage2.save('~/Desktop/'+imgs[1]+'.png')

    # print (faceImage1.log)
    # print (faceImage2.log)

    qtd = QDT_SIFT_FEATURES
    kp1, kp2, ds1, ds2 = fm.match(image1, image2, qtd=qtd)
    # kp1, kp2, ds1, ds2 = fm.match(faceImage1.image, faceImage2.image, qtd=qtd)

    # d1 = np.array(ds1)
    # d2 = np.array(ds2)

    # return np.concatenate((d1, d2), axis=0)
    return extract_features(image1.shape, qtd, kp1, kp2)

def extract_features(shape, qtd, kp1, kp2):

    h = shape[0]/3.0
    d = shape[0]/4.0

    y0 = 0
    y1 = 0
    y2 = 0

    num = min(len(kp1), len(kp2))
    for i in range(num):
        dx = kp1[i].pt[0] - kp2[i].pt[0]
        dy = kp1[i].pt[1] - kp2[i].pt[1]
        ay = (kp1[i].pt[1] + kp2[i].pt[1])/2.0

        if abs(dy) < d:
            if ay < h:
                y0 += 1
            elif ay < 2*h:
                y1 += 1
            else:
                y2 += 1

    # print "In ", num, yt0, yf0, yt1, yf1, yt2, yf2

    q = num/float(qtd)

    a0 = y0/float(y0+y1+y2)
    a1 = y1/float(y0+y1+y2)
    a2 = y2/float(y0+y1+y2)

    # print "Out", q, x0, x1, y0, y1, z0, z1

    return np.array([q, a0, a1, a2])

if __name__ == '__main__':

    # im1 = cv.imread('../Databases/Temp/test/im1.png')
    # im2 = cv.imread('../Databases/Temp/test/im2.png')
    #
    # kp1, kp2, ds1, ds2 = fm.match(im1, im2, qtd=10)
    #
    # cv.waitKey(0)


    train_target, train_input, test_same, test_diff = load_db('01')

    # train_target = train_target.astype(np.int8, copy=False)
    # train_input = train_input.astype(np.int8, copy=False)

    # train_same = test_target.astype(np.int8, copy=False)
    # test_input = test_input.astype(np.int8, copy=False)

    train_target = nn.target_1d_to_2d(train_target)

    net = nl.net.newlvq(nl.tool.minmax(train_input), 30, [.5, .5])
    error = net.train(train_input, train_target, epochs=500, goal=.001, show=10, adapt=True)

    output = net.sim(test_same)
    output = nn.target_2d_to_1d(output)
    same_rate = output.sum()/len(output)
    print "SAME", output, same_rate


    output = net.sim(test_diff)
    output = nn.target_2d_to_1d(output)
    diff_rate = (len(output)-output.sum())/len(output)
    print "DIFF", output, diff_rate

    print "RATE", (same_rate+diff_rate)/2.0


    cv.waitKey(0)
