import os
import time
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.io as sio
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import PIL
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
import tensorflow as tf
layers = tf.keras.layers

from scipy.ndimage.filters import convolve, median_filter
from scipy.ndimage.filters import gaussian_filter
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils, base
from tensorflow.python.ops import array_ops, nn, init_ops, nn_ops
from tensorflow.python.keras import activations
from tensorflow.keras.layers import InputSpec

import config as c

def partitioning(unpartitioned, p_size):
    up_size = unpartitioned.shape[0]
    parts = np.zeros((int((up_size/p_size)**2), p_size, p_size, 1))
    n=0
    for u in range(int(up_size/p_size)):
        for v in range(int(up_size/p_size)):
            parts[n,:,:,0] = unpartitioned[p_size*u:p_size*u+p_size , p_size*v:p_size*v+p_size]
            n = n + 1
    return parts

def tiling(partitions, rec_size):
    p_size = partitions.shape[1]
    reconstructed_img = np.zeros((rec_size,rec_size))
    n=0
    for u in range(int(rec_size/p_size)):
        for v in range(int(rec_size/p_size)):
            reconstructed_img[p_size*u:p_size*u+p_size , p_size*v:p_size*v+p_size] = partitions[n,:,:,0]
            n = n + 1
    return reconstructed_img

def W_tiling(partitions, rec_size, num_bitplane):
    p_size = partitions.shape[1]
    reconstructed_W = np.zeros((rec_size, rec_size, num_bitplane))
    n=0
    for u in range(int(rec_size/p_size)):
        for v in range(int(rec_size/p_size)):
            reconstructed_W[p_size*u:p_size*u+p_size , p_size*v:p_size*v+p_size, :] = partitions[n,:,:,:]
            n = n + 1
    return reconstructed_W


class Conv2D_circular(tf.keras.layers.Layer):
  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               rank=2,
               **kwargs):
    super(Conv2D_circular, self).__init__(trainable=trainable, name=name,
                                activity_regularizer=activity_regularizer,
                                **kwargs)
    self.rank = 2
    self.filters = filters
    self.kernel_size = utils.normalize_tuple(kernel_size, rank, 'kernel_size')
    self.strides = utils.normalize_tuple(strides, rank, 'strides')
    self.padding = utils.normalize_padding(padding)
    self.data_format = utils.normalize_data_format(data_format)
    self.dilation_rate = utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint
    self.input_spec = InputSpec(ndim=self.rank + 2)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    self.kernel = self.add_variable(name='kernel',
                                    shape=kernel_shape,
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    trainable=True,
                                    dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_variable(name='bias',
                                    shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    trainable=True,
                                    dtype=self.dtype)
    else:
      self.bias = None
    self.input_spec = InputSpec(ndim=self.rank + 2,
                                     axes={channel_axis: input_dim})
    self._convolution_op = nn_ops.Convolution(
        input_shape,
        filter_shape = self.kernel.shape,
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=utils.convert_data_format(self.data_format,
                                              self.rank + 2))
    self.built = True

  def call(self, inputs):
      
    input_padded = tf.concat((inputs,inputs[:,:,0:1,:]),axis=2)
    input_padded2 = tf.concat((input_padded,input_padded[:,0:1,:,:]),axis=1)
    outputs_padded = self._convolution_op(input_padded2, self.kernel)
    outputs = outputs_padded[:,0:-1,0:-1,:]

    if self.use_bias:
      if self.data_format == 'channels_first':
        if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          outputs += bias
        if self.rank == 2:
          outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
        if self.rank == 3:
          # As of Mar 2017, direct addition is significantly slower than
          # bias_add when computing gradients. To use bias_add, we collapse Z
          # and Y into a single dimension to obtain a 4D input tensor.
          outputs_shape = outputs.shape.as_list()
          if outputs_shape[0] is None:
            outputs_shape[0] = -1
          outputs_4d = array_ops.reshape(outputs,
                                         [outputs_shape[0], outputs_shape[1],
                                          outputs_shape[2] * outputs_shape[3],
                                          outputs_shape[4]])
          outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
          outputs = array_ops.reshape(outputs_4d, outputs_shape)
      else:
        outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                      [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                      new_space)



# Utilities
def multiply_255(x):
    return x*255.0   

def divide_255(x):
    return x/255.0  

def scalar_output_shape(input_shape):
    return input_shape

def multiply_scalar(x, scalar):
    return x * tf.convert_to_tensor(scalar, tf.float32)

def buildModel(model_path, patch_rows=32, patch_cols=32, channels=1, block_size=8, use_circular=True):
    
    conv2d_layer = layers.Conv2D if use_circular == False else Conv2D_circular
    
    w_rows = int((patch_rows) / block_size)
    w_cols = int((patch_cols) / block_size)
    
    input_img = layers.Input(shape=(patch_rows, patch_cols, 1), name='input_img')
    input_strenght_alpha = layers.Input(shape=(1,), name='strenght_factor_alpha')
    input_watermark = layers.Input(shape=(w_rows, w_cols, 1), name='input_watermark')
    
    # Rearrange input 
    rearranged_img = l1 = layers.Lambda(tf.nn.space_to_depth, arguments={'block_size':block_size}, name='rearrange_img')(input_img)
    
    dct_layer = layers.Conv2D(64, (1, 1), activation='linear', padding='same', use_bias=False, trainable=False, name='dct1')
    dct_layer2 = layers.Conv2D(64, (1, 1), activation='linear', padding='same', use_bias=False, trainable=False, name='dct2')
    idct_layer = layers.Conv2D(64, (1, 1), activation='linear', padding='same', use_bias=False, trainable=False, name='idct')
    dct_layer_img = dct_layer(rearranged_img)
    
    # Concatenating The Image's dct coefs and watermark
    encoder_input = layers.Concatenate(axis=-1, name='encoder_input')([dct_layer_img, input_watermark])
    
    # Encoder
    encoder_model = layers.Conv2D(64, (1, 1), dilation_rate=1, activation='elu', padding='same', name='enc_conv1')(encoder_input)
    encoder_model = conv2d_layer(64, (2, 2), dilation_rate=1, activation='elu', padding='same', name='enc_conv2')(encoder_model)
    encoder_model = conv2d_layer(64, (2, 2), dilation_rate=1, activation='elu', padding='same', name='enc_conv3')(encoder_model)
    encoder_model = conv2d_layer(64, (2, 2), dilation_rate=1, activation='elu', padding='same', name='enc_conv4')(encoder_model)
    encoder_model = conv2d_layer(64, (2, 2), dilation_rate=1, activation='elu', padding='same', name='enc_conv5')(encoder_model)
    encoder_model = idct_layer(encoder_model)
    
    # Strength
    encoder_model = tf.math.multiply(encoder_model, input_strenght_alpha, name="strenght_factor")
    # encoder_model = layers.Lambda(multiply_scalar, arguments={'scalar':input_strenght_alpha}, output_shape=scalar_output_shape, name='strenght_factor')(encoder_model)
    
    encoder_model = layers.Add(name='residual_add')([encoder_model, l1])
    encoder_model = x = layers.Lambda(tf.nn.depth_to_space, arguments={'block_size':block_size}, name='enc_output_depth2space')(encoder_model)
    
    # Attack (The attacks occure in test phase)
    
    # Watermark decoder
    input_attacked_img = layers.Input(shape=(patch_rows, patch_cols, 1), name='input_attacked_img')
    decoder_model = layers.Lambda(tf.nn.space_to_depth, arguments={'block_size':block_size}, name='dec_input_space2depth')(input_attacked_img)
    decoder_model = dct_layer2(decoder_model)
    decoder_model = layers.Conv2D(64, (1, 1), dilation_rate=1, activation='elu', padding='same', name='dec_conv1')(decoder_model)
    decoder_model = conv2d_layer(64, (2, 2), dilation_rate=1, activation='elu', padding='same', name='dec_conv2')(decoder_model)
    decoder_model = conv2d_layer(64, (2, 2), dilation_rate=1, activation='elu', padding='same', name='dec_conv3')(decoder_model)
    decoder_model = conv2d_layer(64, (2, 2), dilation_rate=1, activation='elu', padding='same', name='dec_conv4')(decoder_model)
    decoder_model = layers.Conv2D(1, (1, 1), dilation_rate=1, activation='sigmoid', padding='same', name='dec_output_depth2space')(decoder_model)
    
    # Whole model
    embedding_net = tf.keras.models.Model(inputs=[input_img, input_watermark, input_strenght_alpha], outputs=[x])
    extractor_net = tf.keras.models.Model(inputs=[input_attacked_img], outputs=[decoder_model])
    
    # Set weights
    DCT_MTX = sio.loadmat('./transforms/DCT_coef.mat')['DCT_coef']
    dct_mtx = np.reshape(DCT_MTX, [1,1,64,64])
    embedding_net.get_layer('dct1').set_weights(np.array([dct_mtx]))
    extractor_net.get_layer('dct2').set_weights(np.array([dct_mtx]))
    
    IDCT_MTX = sio.loadmat('./transforms/IDCT_coef.mat')['IDCT_coef']
    idct_mtx = np.reshape(IDCT_MTX, [1,1,64,64])
    embedding_net.get_layer('idct').set_weights(np.array([idct_mtx]))
    
    embedding_net.load_weights(model_path,by_name = True)
    extractor_net.load_weights(model_path,by_name = True)
    return embedding_net, extractor_net





def embed_watermark(embedding_net, im, W, alpha, img_rows = 512, img_cols = 512, patch_rows = 32, patch_cols = 32, block_size = 8, Is_mean_normalized = True, mean_normalize = 128.0, std_normalize = 255.0):
    assert patch_rows == patch_cols, 'Patches must have same rows and columns'
    assert img_rows % patch_rows == 0 and img_cols % patch_cols == 0, 'Image size must be dividable by the patch size'

    w_rows = int((patch_rows) / block_size) #4
    w_cols = int((patch_cols) / block_size) #4
    bits_per_patch = w_rows * w_cols #16
    n_patches = int((img_rows * img_cols) / (patch_rows * patch_cols)) #1024
    total_cap = n_patches * bits_per_patch

    message_length = c.wm_cap

    assert total_cap % message_length == 0, 'Total Capacity must be dividable by message length'
    n_redundancy = total_cap // message_length

    # print(im.shape)

    if im.dtype == np.float32 or im.dtype == np.float64:
        im = (im * 255).astype(np.uint8)  # Convert to uint8
    if im.shape[-1] == 3:
        # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray, G, R = cv2.split(im)
    else:
        im_gray = im
    
    # Normalize image
    Im_normalized = (im_gray.copy() - mean_normalize if Is_mean_normalized else 0) / std_normalize 
    # print(Im_normalized.shape)
    num_batch = (img_rows * img_cols) // (patch_rows * patch_cols)

    Im_32x32_patchs = partitioning(Im_normalized, p_size=patch_rows)
    # Convert the list to a NumPy array
    W = np.array(W).astype(np.float32)

    # Reshape to the desired shape (num_batch, w_rows, w_cols, 1)
    W = np.reshape(W, [-1, w_rows, w_cols]).astype(np.float32)

    #Redundant embedding
    N_mtx = (img_rows//patch_rows) # assuming img_rows == img_cols
    W_robust = np.zeros([N_mtx, N_mtx, w_rows, w_cols], dtype=np.float32)
    k = 0
    n_repeats = 0
    for d in range(N_mtx):
        for i in range(N_mtx):
            if k >= W.shape[0]:
                break
            W_robust[i, (i+d)%N_mtx, :, :] = W[k, :, :] ##### % 8 for 256x256
            n_repeats += 1
            if n_repeats >= n_redundancy:
                n_repeats = 0
                k += 1
                
    W_robust = np.reshape(W_robust, [-1, w_rows, w_cols, 1])
    # Apply embedding network
    # print("Im_32x32_patchs: ", Im_32x32_patchs.shape)
    # print("W_robust", W_robust.shape)
    Iw_batch = embedding_net.predict_on_batch([Im_32x32_patchs, W_robust, alpha*np.ones_like(W_robust)])
    # reconstruct Iw
    Iw = tiling(Iw_batch, rec_size=img_rows)
    Iw *= std_normalize
    Iw += mean_normalize if Is_mean_normalized else 0
    Iw[Iw > 255] = 255
    Iw[Iw < 0] = 0
    Iw = np.uint8(Iw.squeeze())
    Iw = Iw.astype(G.dtype)
    Iw_tmp = cv2.merge([Iw, G, R])
    return Iw_tmp

def extract_watermark(extractor_net, Iw_attacked, img_rows = 512, img_cols = 512, patch_rows = 32, patch_cols = 32, block_size = 8, Is_mean_normalized = True, mean_normalize = 128.0, std_normalize = 255.0):
    
    assert patch_rows == patch_cols, 'Patches must have same rows and columns'
    assert img_rows % patch_rows == 0 and img_cols % patch_cols == 0, 'Image size must be dividable by the patch size'

    w_rows = int((patch_rows) / block_size) #4
    w_cols = int((patch_cols) / block_size) #4
    bits_per_patch = w_rows * w_cols #16
    n_patches = int((img_rows * img_cols) / (patch_rows * patch_cols)) #1024
    total_cap = n_patches * bits_per_patch
    n_redundancy = total_cap // c.wm_cap
    # Apply Attack
    # Iw_attacked = attack['func'](Iw, attack_params)

    W = np.random.randint(low=0, high=2, size=(c.wm_cap,1)).astype(np.float32)
    W = np.reshape(W, [-1, w_rows, w_cols])

    if Iw_attacked.dtype == np.float32 or Iw_attacked.dtype == np.float64:
        Iw_attacked = (Iw_attacked * 255).astype(np.uint8)  # Convert to uint8
    if Iw_attacked.shape[-1] == 3:
        # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        Iw_attacked, G, R = cv2.split(Iw_attacked)
    else:
        Iw_attacked = Iw_attacked
    
    Iw_attacked = (Iw_attacked - mean_normalize if Is_mean_normalized else 0) / std_normalize
                                          
    Iw_attacked_patchs = partitioning(Iw_attacked, p_size=patch_rows)

    # Feed to extractor
    w_batch = extractor_net.predict_on_batch([Iw_attacked_patchs])
    w_batch = w_batch > 0.5

    # Majority voting
    N_mtx = (img_rows//patch_rows) # assuming img_rows == img_cols
    w_batch = np.reshape(w_batch, [N_mtx, N_mtx, w_rows, w_cols])
    w_extracted = np.zeros_like(W)
    
    k = 0
    n_repeats = 0
    
    for d in range(N_mtx):
        for i in range(N_mtx):
            if k >= w_extracted.shape[0]:
                break
            w_extracted[k, :, :] += w_batch[i, (i+d)%N_mtx, :, :] ####### % 8 for 256x256 ?
            n_repeats += 1
            if n_repeats >= n_redundancy:
                n_repeats = 0
                k += 1
    
    w_extracted = (w_extracted > n_redundancy//2)

    w_extracted = w_extracted.astype(np.float32)


    w_extracted_flat = w_extracted.flatten()

    w_extracted_reshaped = w_extracted_flat.reshape((c.wm_cap, 1))

    return w_extracted_reshaped
    
