import tensorflow as tf
import numpy as np
import scipy.io

import utils


vgg19_file_name = 'imagenet-vgg-verydepp-19.mat'


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


class VGG19(object):

    def __init__(self, input_image):
        utils.vgg19_download()  # download vgg19 pre-trained model

        self.vgg19_layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )

        self.mean_pixels = np.array([123.68, 116.779, 103.939])

        self.weights = scipy.io.loadmat(vgg19_file_name)['layers'][0]

        self.input_img = input_image
        self.vgg19 = self.build(self.input_img)

    def _get_weight(self, idx, layer_name):
        weight = self.weights[idx][0][0][2][0][0]
        bias = self.weights[idx][0][0][2][0][1]

        assert layer_name == self.weights[idx][0][0][0][0]

        weight = np.transpose(weight, (1, 0, 2, 3))  # (h, w, in_c, out_c) to (h, w, in_c, out_c)
        bias = bias.reshape(-1)

        return weight, bias

    def build(self, img):
        x = {}  # network
        net = img

        for idx, name in enumerate(self.vgg19_layers):
            layer_name = name[:4]

            if layer_name == 'conv':
                weight, bias = self._get_weight(idx, name)

                net = conv2d_layer(net, weight, bias)
            elif layer_name == 'relu':
                net = tf.nn.relu(net)
            elif layer_name == 'pool':
                net = pool2d_layer(net)

            x[name] = net

        return x
