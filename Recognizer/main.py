__author__ = 'Mario'


import cv2 as cv
import numpy as np
import neurolab as nl

# import neural_network as nn
import feature_match as fm

PATH = '../Databases/lfwcrop_color/'

def load_db(index):
    print "same"
    train_same = load_train_file(PATH+'lists/'+index+'_train_same.txt')

    print "diff"
    train_diff = load_train_file(PATH+'lists/'+index+'_train_diff.txt')

    print "same"
    test_same = load_test_file(PATH+'lists/'+index+'_test_same.txt')

    print "diff"
    test_diff = load_test_file(PATH+'lists/'+index+'_test_diff.txt')

    print "train - same", train_same.shape, "train - diff", train_diff.shape

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
        print data.shape
        if min(data.shape) > 0:
            ret.append(data)
        i+=1
        if i>=100:
            break

    return np.concatenate(ret, axis=0)

def load_test_file(fname):
    f = open(fname, 'r')

    i = 0
    ret = []
    for line in f:
        data = proccess_images(line.split())
        print data.shape
        if min(data.shape) > 0:
            ret.append(data)
        i+=1
        if i>=100:
            break

    return ret

def proccess_images(imgs):
    img1 = cv.imread(PATH+'faces_png/'+imgs[0]+'.png')
    img2 = cv.imread(PATH+'faces_png/'+imgs[1]+'.png')

    kp1, kp2, ds1, ds2 = fm.match(img1, img2, qtd=5)

    d1 = np.array(ds1)
    d2 = np.array(ds2)

    return np.concatenate((d1, d2), axis=0)

if __name__ == '__main__':

    # im1 = cv.imread('../Databases/Temp/test/im1.png')
    # im2 = cv.imread('../Databases/Temp/test/im2.png')
    #
    # kp1, kp2, ds1, ds2 = fm.match(im1, im2, qtd=10)
    #
    # cv.waitKey(0)


    train_target, train_input, test_same, test_diff = load_db('01')

    train_target = train_target.astype(np.int8, copy=False)
    train_input = train_input.astype(np.int8, copy=False)

    # train_same = test_target.astype(np.int8, copy=False)
    # test_input = test_input.astype(np.int8, copy=False)

    # print "train:", train.shape, "test:", test.shape
    print train_target.dtype, train_input.dtype


    net = nl.net.newlvq(nl.tool.minmax(train_input), 30, [.5, .5])
    error = net.train(train_input, train_target, epochs=100, goal=.001, show=10, adapt=True)

    


    # net = nn.run_lvq(train_input, train_target, test_input, test_target)

    cv.waitKey(0)
