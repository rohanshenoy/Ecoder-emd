from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Conv2DTranspose, Reshape, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import numpy as np
import json
from telescope import telescopeMSE2

import tensorflow as tf
##Need to use 32 bits for telescopeMSE
tf.keras.backend.set_floatx('float32')

from denseCNN import denseCNN

class dense2DkernelCNN(denseCNN):
    def __init__(self,name='',weights_f=''):
        self.name=name
        self.pams ={
            'CNN_layer_nodes'  : [8],  #n_filters
            'CNN_kernel_size'  : [3],
            'CNN_padding'      : ['same'],  
            'CNN_pool'         : [False],
            'share_filters'    : True,
            'Dense_layer_nodes': [], #does not include encoded layer
            'encoded_dim'      : 12,
            'shape'            : (4,4,3),
            'channels_first'   : False,
            'arrange'          : [],
            'arrMask'          : [],
            'calQMask'         : [],
            'n_copy'           : 0,      # no. of copy for hi occ datasets
            'loss'             : '',
            'optimizer'       : 'adam',
        }

        self.weights_f =weights_f
        

    def init(self,printSummary=True):
        encoded_dim = self.pams['encoded_dim']

        CNN_layer_nodes   = self.pams['CNN_layer_nodes']
        CNN_kernel_size   = self.pams['CNN_kernel_size']
        CNN_padding       = self.pams['CNN_padding']
        CNN_pool          = self.pams['CNN_pool']
        Dense_layer_nodes = self.pams['Dense_layer_nodes'] #does not include encoded layer
        channels_first    = self.pams['channels_first']
        share_filters      = self.pams['share_filters']

        # fix to one cnn layer for now
        nnodes     =CNN_layer_nodes[0] #8
        CNN_kernel =CNN_kernel_size[0] #3
        CNN_padding=CNN_padding[0]

        inputs = Input(shape=self.pams['shape'], name='input_1')
        x = inputs
        
        x1 = Lambda(lambda x: x[:,:,:,0:1], name='lambda_1')(x)
        x2 = Lambda(lambda x: x[:,:,:,1:2], name='lambda_2')(x)
        x3 = Lambda(lambda x: x[:,:,:,2:3], name='lambda_3')(x)

        if share_filters:
            conv = Conv2D(nnodes, CNN_kernel, activation='relu',padding=CNN_padding, name='conv2d_1')
            x1 = conv(x1)
            x2 = conv(x2)
            x3 = conv(x3)
        else:
            x1 = Conv2D(nnodes, CNN_kernel, activation='relu',padding=CNN_padding)(x1)
            x2 = Conv2D(nnodes, CNN_kernel, activation='relu',padding=CNN_padding)(x2)
            x3 = Conv2D(nnodes, CNN_kernel, activation='relu',padding=CNN_padding)(x3)

        if CNN_pool[0]:
           x1 = MaxPooling2D( (2,2), padding='same')(x1) 
           x2 = MaxPooling2D( (2,2), padding='same')(x2) 
           x3 = MaxPooling2D( (2,2), padding='same')(x3) 

        conv_vol_slice = K.int_shape(x1)
        x1 = Flatten(name='flatten_1')(x1)
        x2 = Flatten(name='flatten_2')(x2)
        x3 = Flatten(name='flatten_3')(x3)

        x = [x1,x2,x3]
        x = Concatenate(axis=-1,name='concat_1')(x)

        conv_vol = K.int_shape(x)

        encodedLayer = Dense(encoded_dim, activation='relu',name='encoded_vector')(x)

        # Instantiate Encoder Model
        self.encoder = Model(inputs, encodedLayer, name='encoder')
        if printSummary:
          self.encoder.summary()

        encoded_inputs = Input(shape=(encoded_dim,), name='decoder_input')
        x = encoded_inputs

        x = Dense(conv_vol[1], activation='relu',name='dense_2')(x)

        x = Reshape((conv_vol_slice[1],conv_vol_slice[2],nnodes,3,),name='reshape_1')(x)
        
        x1 = Lambda(lambda x: x[:,:,:,:,0],  name='lambda_4')(x)
        x2 = Lambda(lambda x: x[:,:,:,:,1],  name='lambda_5')(x)
        x3 = Lambda(lambda x: x[:,:,:,:,2],  name='lambda_6')(x)

        if CNN_pool[0]:
           x1 = UpSampling2D( (2,2) )(x1) 
           x2 = UpSampling2D( (2,2) )(x2) 
           x3 = UpSampling2D( (2,2) )(x3) 

        ## Use n filter here
        conv_t = Conv2DTranspose(nnodes, CNN_kernel, activation='relu', padding=CNN_padding, name='conv2d_transpose_1')
        x1 = conv_t(x1)
        x2 = conv_t(x2)
        x3 = conv_t(x3)
        ## Always use 1 filter
        conv_t2 = Conv2DTranspose(1, CNN_kernel, activation=None, padding='same', name='conv2d_transpose_2')
        x1 = conv_t2(x1)
        x2 = conv_t2(x2)
        x3 = conv_t2(x3)

        x = [x1,x2,x3]
        x = Concatenate(axis=-1,name='concat_2')(x)

        outputs = Activation('sigmoid', name='decoder_output')(x)

        self.decoder = Model(encoded_inputs, outputs, name='decoder')
        if printSummary:
          self.decoder.summary()

        self.autoencoder = Model(inputs, self.decoder(self.encoder(inputs)), name='autoencoder')
        if printSummary:
          self.autoencoder.summary()

        self.compileModels()

        CNN_layers=''
        if len(CNN_layer_nodes)>0:
            CNN_layers += '_Conv'
            for i,n in enumerate(CNN_layer_nodes):
                CNN_layers += f'_{n}x{CNN_kernel_size[i]}'
                if CNN_pool[i]:
                    CNN_layers += 'pooled'
        Dense_layers = ''
        if len(Dense_layer_nodes)>0:
            Dense_layers += '_Dense'
            for n in Dense_layer_nodes:
                Dense_layers += f'_{n}'

        self.name = f'Autoencoded{CNN_layers}{Dense_layers}_Encoded_{encoded_dim}'
        
        if not self.weights_f=='':
            self.autoencoder.load_weights(self.weights_f)
