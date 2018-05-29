import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tfcl
from IPython import embed

from layers import *
functions = {"fc": full_connection,
             "conv": convolution2d,
             "deconv": transposed_convolution2d,
             "reshape": reshape}


class UnitModel():
    def __init__(self, layers, scope="unitmodel"):
        self.scope = scope
        self.layers = layers
    
    def __call__(self, h, phase=True, reuse=False):
        return self.forward(h, phase, reuse)
    
    def forward(self, h, phase, reuse):
        with tf.variable_scope(self.scope):
            for scope, layer in self.layers.iteritems():
                h = functions[layer["type"]](h, layer["param"], phase, str(scope), reuse)
            return h

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.scope in var.name]  

class OutputHead():
    def __init__(self, n_class, is_wgan=False, scope="example"):
        self.scope = scope
        self.n_class = n_class
        self.is_wgan = is_wgan

    def __call__(self, h, phase, reuse=False):
        return self.forward(h, phase, reuse)

    def forward(self, h, phase, reuse):
        with tf.variable_scope(self.scope):
            if self.is_wgan:
                activation_fn = tf.identity
            else:
                activation_fn = tf.nn.sigmoid
            dr = tfcl.fully_connected(h, 1,
                                      activation_fn=activation_fn,
                                      scope="art",
                                      reuse=reuse)
            dc = tfcl.fully_connected(h, self.n_class,
                                      activation_fn=tf.nn.softmax,
                                      scope="class",
                                      reuse=reuse)
        return dr, dc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.scope in var.name]
    
class SiameseCats():
    def __init__(self, g_layers, d_layers, batchsize, z_size, n_class, is_wgan=False):
        self.generator = UnitModel(g_layers, "generator")
        self.discriminator = UnitModel(d_layers, "discriminator")
        self.discriminator_head = OutputHead(n_class, is_wgan, "output_head")
        self.batchsize = batchsize
        self.z_size = z_size
            
    def sample(self, phase=False):
        z = tf.random_normal((self.batchsize, self.z_size), 0, 1)
        gz = self.generator(z, phase, False)
        dgz = self.discriminator(gz, phase, False)
        drgz, dcgz = self.discriminator_head(dgz, phase, False)
        return gz, drgz, dcgz
    
    def forward(self, x, phase):
        z = tf.random_normal((self.batchsize, self.z_size), 0, 1)
        gz = self.generator(z, phase, False)
        dx = self.discriminator(x, phase, False)
        dgz = self.discriminator(gz, phase, True)
        drx, dcx  = self.discriminator_head(dx, phase, False)
        drgz, dcgz = self.discriminator_head(dgz, phase, True)
        return drx, drgz, dcx, dcgz, gz
    
    def __call__(self, real, phase):
        return self.forward(real, phase)
