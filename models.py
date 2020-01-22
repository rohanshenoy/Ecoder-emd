from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import keras as kr
import qkeras.qkeras as qkr


def autoCNN(N_filters=[16,8,4],conv_filter=(3,3),pool_filter=(2,2),shape=(12,4,1),weights_f=''):

  input_img = Input(shape)  # adapt this if using `channels_first` image data format

  for i,ifilter in enumerate(N_filters):
    if i==0:
      ### use input_img for first conv layer
      x = Conv2D(ifilter, conv_filter, activation='relu', padding='same')(input_img)
    else:
      x = Conv2D(ifilter, conv_filter, activation='relu', padding='same')(x)
    if i!=len(N_filters)-1:
      x = MaxPooling2D(pool_filter, padding='same')(x)
    else:
      ### make encoded tensor for last pooling layer 
      encoded = MaxPooling2D(pool_filter, padding='same')(x)
  
  ## decoder part
  for i,ifilter in enumerate(N_filters[::-1]):
    if i==0:
      ### use input_img for first conv layer
      x = Conv2D(ifilter, conv_filter, activation='relu', padding='same')(encoded)
    elif i==len(N_filters)-1:
      x = Conv2D(ifilter, conv_filter, activation='relu')(x)        # no padding for last conv layer to keep the shape
    else:
      x = Conv2D(ifilter, conv_filter, activation='relu', padding='same')(x)
    x = UpSampling2D(pool_filter)(x)

  ## output layer
  decoded = Conv2D(1, conv_filter, activation='sigmoid', padding='same')(x)
  
  encoder     = Model(input_img, encoded)
  autoencoder = Model(input_img, decoded)
  autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
  
  if not weights_f=='':
    autoencoder.load_weights(weights_f)
  #autoencoder.summary()

  return autoencoder,encoder

#def deepAuto(encoding_dim = 4,input_dim=48,weights_f=''):
def deepAuto(dims=[],weights_f=''):

  # dims = sequential number of layers up-to the deepest one
  encoding_dim = dims[-1] 
  input_dim    = dims[0]
  
  # "encoded" is the encoded representation of the input
  input_img = Input(shape=(input_dim,))
  encoded = Dense(dims[1], activation='relu')(input_img)
  for d in dims[2:]:
    encoded = Dense(d, activation='relu')(encoded)
  r_dims = dims[::-1]
  decoded = Dense(r_dims[1], activation='relu')(encoded)
  for d in r_dims[2:-1]:
    decoded = Dense(d, activation='relu')(decoded)  ## repeat this line
  decoded = Dense(input_dim, activation='sigmoid')(decoded)

  encoder     = Model(input_img, encoded)
  autoencoder = Model(input_img, decoded)
  autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
  #autoencoder.summary()
  
  if not weights_f=='':
    autoencoder.load_weights(weights_f)

  return autoencoder,encoder

def QautoCNN(shape=(12,4,1), weights_f=''):
  input_img = Input(shape=shape,name='input')    
  x = Conv2D(16, (3, 3), activation='relu', padding='same',name='conv2d_0_m')(input_img)
  x = MaxPooling2D((2, 2), padding='same',name='mp_0')(x)
  x = Conv2D(8, (3, 3), activation='relu', padding='same',name='conv2d_1_m')(x)
  x = MaxPooling2D((2, 2), padding='same',name='mp_1')(x)
  x = Conv2D(4, (3, 3), activation='relu', padding='same',name='conv2d_2_m')(x)
  encoded = MaxPooling2D((2, 2), padding='same',name='mp_2')(x)
  
  # at this point the representation is (4, 4, 8) i.e. 128-dimensional
  
  x = Conv2D(4, (3, 3), activation='relu', padding='same',name='conv2d_3_m')(encoded)
  x = UpSampling2D((2, 2),name='up_0')(x)
  x = Conv2D(8, (3, 3), activation='relu', padding='same',name='conv2d_4_m')(x)
  x = UpSampling2D((2, 2),name='up_1')(x)
  x = Conv2D(16, (3, 3), activation='relu',name='conv2d_5_m')(x)
  x = UpSampling2D((2, 2),name='up_2')(x)
  decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same',name='conv2d_6_m')(x)
 
  q_dict = {
      "conv2d_0_m": {
          "kernel": "quantized_bits(4)",
          "bias": "quantized_bits(4)"
      },
      "conv2d_1_m": {
          "kernel": "quantized_bits(4)",
          "bias": "quantized_bits(4)"
      },
      "conv2d_2_m": {
          "kernel": "quantized_bits(4)",
          "bias": "quantized_bits(4)"
      },
      "conv2d_3_m": {
          "kernel": "quantized_bits(4)",
          "bias": "quantized_bits(4)"
      },
      "conv2d_4_m": {
          "kernel": "quantized_bits(4)",
          "bias": "quantized_bits(4)"
      },
      "conv2d_5_m": {
          "kernel": "quantized_bits(4)",
          "bias": "quantized_bits(4)"
      },    
      "conv2d_6_m": {
          "kernel": "quantized_bits(4)",
          "bias": "quantized_bits(4)"
      },    
      "mp_0": {
          "kernel": "quantized_bits(4)",
          "bias": "quantized_bits(4)"
      },        
      "mp_1": {
          "kernel": "quantized_bits(4)",
          "bias": "quantized_bits(4)"
      },        
      "mp_2": {
          "kernel": "quantized_bits(4)",
          "bias": "quantized_bits(4)"
      },   
      "up_0": {
          "kernel": "quantized_bits(4)",
          "bias": "quantized_bits(4)"
      }, 
      "up_1": {
          "kernel": "quantized_bits(4)",
          "bias": "quantized_bits(4)"
      }, 
      "up_2": {
          "kernel": "quantized_bits(4)",
          "bias": "quantized_bits(4)"
      }  
  }

  autoencoder     = kr.Model(input_img, decoded)
  encoder         = kr.Model(input_img, encoded)
  qautoencoder, _ = qkr.model_quantize(autoencoder, q_dict, 4)
  qencoder, _     = qkr.model_quantize(encoder, q_dict, 4)
  qautoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
  #qautoencoder.summary()

  if not weights_f=='':
    qautoencoder.load_weights(weights_f)

  return qautoencoder,qencoder
 
