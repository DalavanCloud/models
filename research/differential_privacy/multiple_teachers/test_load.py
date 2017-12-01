#test loading to new data used to train student, move to train student afterwards
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import input
import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('data_dir','data_dir','Temporary storage')

tmp = 'digit_data/all-digits-MNIST-gammaC-sh-asp.data.gz'
train_data = input.extract_mnist_data(tmp, 1000, 28, 1)
#test_data = extract_mnist_data(local_urls[2], 10000, 28, 1)
#test_labels = extract_mnist_labels(local_urls[3], 10000)

print(train_data.shape)
