from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import input
from math import *
import tensorflow as tf
import numpy as np
from scipy import spatial
from numpy import linalg as LA
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_nrmse as nrmse
from skimage.measure import compare_mse as mse



FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('data_dir','data_dir','Temporary storage')

test_svhn = np.load(FLAGS.data_dir + '/svhn_test_1000.npy')
test_cifar10 = np.load(FLAGS.data_dir + '/cifar10_test_1000.npy')
base_mnist = np.load(FLAGS.data_dir + '/train-images-idx3-ubyte.gz.npy')
base_cifar10 = np.load(FLAGS.data_dir + '/cifar10_train.npy')


def pre_processing_mnist(arr):
    # trim image to minst size 28 x 28, and extract different channels
    return arr[:, 2:30, 2:30, 0:1], arr[:, 2:30, 2:30, 1:2], arr[:, 2:30, 2:30, 2:3]


def average_distance(arr1, arr2, distance, flat = 'array'):
    shapes = arr1.shape
    if flat == 'array':
        base = arr1.reshape((arr1.shape[0], -1)) [::10] # sample one of every ten training sets
        test = arr2.reshape((arr2.shape[0], -1))
    elif flat == 'single':
        base = arr1.reshape(arr1.shape[0], shapes[1], -1)[::50] # sample one of every 100 training sets
        test = arr2.reshape(arr2.shape[0], shapes[1], -1)[::10]

    #print(base.shape, test.shape)
    distance_sum = 0
    for b in  base:
        for t in test:
            d = distance(b,t)
            distance_sum += d 
    return distance_sum/float(len(base)*len(test))

def avg_cosine(arr1,arr2):
    # fast implementation of cached vector length
    base = arr1.reshape((arr1.shape[0], -1))[::10] # sample one of every ten training sets
    test = arr2.reshape((arr2.shape[0], -1))

    distance_sum = 0
    base_length = LA.norm(base, axis = 1)
    test_length = LA.norm(test, axis = 1)

    for i,b in enumerate(base):
        for j,t in enumerate(test):
            d = 1- np.inner(b,t)/(base_length[i] * test_length[j])
            distance_sum += d 
    return distance_sum/float(len(base)*len(test))

test_svhn_0, test_svhn_1, test_svhn_2 = pre_processing_mnist(test_svhn)
test_cifar10_0, test_cifar10_1, test_cifar10_2 = pre_processing_mnist(test_cifar10)
test_mnist = np.load(FLAGS.data_dir + '/mnist_test_1000.npy')


def fast_manhattan(x,y):
    # faster implementation of manhattan using numpy broadcast.
    return np.absolute(x-y).sum()

def euclidean_distance(x,y):
    """ return euclidean distance between two lists """
    return np.square(x-y).sum()

def cosine_distance(x,y):
    return spatial.distance.cosine(x,y)

def test_distance(distance, cos = False, flat = 'array'):
    tests = ['test_mnist', 'test_svhn_0', 'test_svhn_1', 'test_svhn_2', 'test_cifar10_0', 'test_cifar10_1', 'test_cifar10_2']
    for t in tests:
        #print('Base:Minst; Public:' + t +': ', average_distance(base_mnist, eval(t), distance))
        if cos:
            print(avg_cosine(base_mnist, eval(t)))
        else:
            print(average_distance(base_mnist, eval(t), distance, flat))

#test_distance(fast_manhattan)
#test_distance(euclidean_distance)
#test_distance(cosine_distance, cos= True)
print('ssim')
test_distance(ssim, flat = 'single')
print('nrmse')
test_distance(nrmse, flat = 'single')
print('mse')
test_distance(mse, flat = 'single')

