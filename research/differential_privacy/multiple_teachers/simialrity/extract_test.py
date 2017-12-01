from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import input
import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('data_dir','data_dir','Temporary storage')


test_data, test_labels = input.ld_svhn(test_only=True)
test_data_1000 = test_data[:1000, :, :, :]
np.save(FLAGS.data_dir + '/svhn_test_1000',test_data_1000)

test_data, test_labels = input.ld_cifar10(test_only=True)
test_data_1000 = test_data[:1000, :, :, :]
np.save(FLAGS.data_dir + '/cifar10_test_1000',test_data_1000)

test_data, test_labels = input.ld_mnist(test_only=True)
test_data_1000 = test_data[:1000, :, :, :]
np.save(FLAGS.data_dir + '/mnist_test_1000',test_data_1000)
