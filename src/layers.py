import numpy as np
import tensorflow as tf
from IPython import embed

activations = {"relu": tf.nn.relu,
               "leaky_relu": tf.nn.leaky_relu,
               "tanh": tf.nn.tanh,
               "sigmoid": tf.nn.sigmoid,
               "identity": tf.identity}
normalizers = {"batch_norm": tf.contrib.layers.batch_norm,
               "none": None}

def full_connection(x, param, phase, scope, reuse):
    with tf.variable_scope(scope):
        y = tf.contrib.layers.fully_connected(x,
                                              param["num_outputs"],
                                              activation_fn=activations[param["activation"]],
                                              normalizer_fn=normalizers[param["normalizer"]],
                                              normalizer_params={'is_training':phase},
                                              reuse=reuse,
                                              scope="fc")
    return y
    
def convolution2d(x, param, phase, scope, reuse):
    with tf.variable_scope(scope):
        y = tf.contrib.layers.conv2d(x,
                                     param["num_outputs"],
                                     param["kernel"],
                                     param["stride"],
                                     padding='SAME',
                                     activation_fn=activations[param["activation"]],
                                     normalizer_fn=normalizers[param["normalizer"]],
                                     normalizer_params={'is_training':phase},
                                     reuse=reuse,
                                     scope="conv2d")
    return y

def transposed_convolution2d(x, param, phase, scope, reuse):
    with tf.variable_scope(scope):
        y = tf.contrib.layers.conv2d_transpose(x,
                                               param["num_outputs"],
                                               param["kernel"],
                                               param["stride"],
                                               padding='SAME',
                                               activation_fn=activations[param["activation"]],
                                               normalizer_fn=normalizers[param["normalizer"]],
                                               normalizer_params={'is_training':phase},
                                               reuse=reuse,
                                               scope="deconv2d")
    return y

def reshape(x, param, phase, scope, reuse):
    shape = [-1] + param["shape"]
    print shape
    return tf.reshape(x, shape)
