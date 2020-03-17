import tensorflow as tf
import numpy as np

'''
    Some utilities and functions which make our life easy while constructing the tensorflow graph defined here.
'''

def Xavier(inp,outp,const=1.0):
    '''
        Generate distribution of Xavier initalization: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        Inputs: inp: no. of input neurons - int
                outp: no. of output neurons - int
                const: multipier for fine-tuning.
        Returns: range limit for the normal distribution in Xavier- float.
    '''
    number = const*np.sqrt(4.0/(inp+outp))
    return number
def GetFCVariable(shape, is_bias = True):
    '''
        Generates a variable from a Fully-connected layer, with Xavier initialization.
        Inputs: shape: a list of ints, first int gives no. of inputs and second int no. of outputs.
                is_bias: a boolean; True if bias variable needs to be generated, else False.
        Returns: list of tensors; [weight,bias] if is_bias = True, else returns [weight].
    '''
    xav = Xavier(shape[0],shape[1])
    if is_bias is True:
        return [tf.Variable(tf.random_uniform([shape[0],shape[1]],-xav,xav)),tf.Variable(tf.random_uniform([shape[1]],-xav,xav))]
    else:
        return [tf.Variable(tf.random_uniform([shape[0],shape[1]],-xav,xav))]
def GetConvVar(shape):
    '''
        Generates a variable for performing convolution with Xavier initialization.
        Inputs: shape: a list of ints, put up like [kernelsize,kernelsize,channel_in,channel_out]
        Returns: list of [weight,bias] of corresponding shape.
    '''
    xav = Xavier(shape[0]*shape[1]*shape[2],shape[3])
    return[tf.Variable(tf.random_uniform([shape[0],shape[1],shape[2],shape[3]],-xav,xav)),tf.Variable(tf.random_uniform([shape[3]],-xav,xav))]
def conv2d(x,W,b,strides = 2, activation  = tf.nn.relu):
    '''
        Performs convolution, adds bias term, and then performs the activation. Padding used is "SAME".
        Inputs: x: tf tensor of shape [batch_size,size,size,n_channels]
                W: convolution kernel - a tf variable of shape [ksize,ksize,n_channel,n_channel_out]
                b: convolution kernel bias - a tf variable of shape [n_channel_out]
                strides: stride size for convolution - int
                activation: activation function to be performed on the convolution - valid tensorflow activation function.
        Returns: tensor of shape [batch_size,size2,size2,n_channel_out], depending on stride size.
    '''
    x=tf.nn.conv2d(x,W,strides = [1,strides,strides,1],padding = 'SAME')
    x=tf.nn.bias_add(x,b)
    return activation(x)
def maxpool2d(x,k = 2, s = 2):
    '''
        Function to perform max-pooling (uniform over an image). Padding used is "SAME".
        Inputs: x: tf tensor of shape [batch_size,size,size,n_channel].
                k: pooling kernel size- int.
                s: stride size - int.
        Returns: tf tensor of shape [batch_size,size2,size2,n_channel] , depending on s.
    '''
    return tf.nn.max_pool(x,ksize = [1,k,k,1],strides = [1,s,s,1],padding = 'SAME')

def Saliency(outp,inp,sess):
    '''
        Come up with a function which calculates gradients of output w.r.t input.
    '''
    with sess.as_default():
        grads = tf.stack(tf.gradients(outp,inp))
        return grads
