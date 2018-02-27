import tensorflow as tf
import numpy as np
import scipy.io


def conv2d_layer(input_, weights, bias):
    """ convolution 2d layer with bias """

    x = tf.nn.conv2d(input_, filter=weights, strides=(1, 1, 1, 1), padding='SAME')
    x = tf.nn.bias_add(x, bias)

    return x


def pool2d_layer(input_, pool='avg'):
    if pool == 'avg':
        x = tf.nn.avg_pool(input_, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
    else:
        x = tf.nn.max_pool(input_, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

    return x


class VGG19:

    def __init__(self):
        pass
