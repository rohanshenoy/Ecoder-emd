from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Conv2DTranspose, Reshape, Activation, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import numpy as np
import json
from telescope import telescopeMSE2

class dense2DkernelCNN:
    def __init__(self,name='',weights_f=''):
        self.name=name
        self.pams ={
            'CNN_layer_nodes'  : [8],  #n_filters
            'CNN_kernel_size'  : [3],
            'CNN_pool'         : [False],
            'Dense_layer_nodes': [], #does not include encoded layer
            'encoded_dim'      : 12,
            'shape'            : (1,4,4),
            'channels_first'   : False,
            'arrange'          : [],
            'arrMask'          : [],
            'n_copy'           : 0,      # no. of copy for hi occ datasets
            'loss'             : ''
        }

        self.weights_f =weights_f
        

    def setpams(self,in_pams):
        for k,v in in_pams.items():
            self.pams[k] = v

    def shuffle(self,arr):
        order = np.arange(48)
        np.random.shuffle(order)
        return arr[:,order]
    
    def cloneInput(self,input_q,n_copy,occ_low,occ_hi):
        shape = self.pams['shape']
        nonzeroQs = np.count_nonzero(input_q.reshape(len(input_q),48),axis=1)
        selection = np.logical_and(nonzeroQs<=occ_hi,nonzeroQs>occ_low)
        occ_q     = input_q[selection]
        occ_q_flat= occ_q.reshape(len(occ_q),48)
        self.pams['cloned_fraction'] = len(occ_q)/len(input_q)
        for i in range(0,n_copy):
            clone   = self.shuffle(occ_q_flat)
            clone   = clone.reshape(len(clone),shape[0],shape[1],shape[2])
            input_q = np.concatenate([input_q,clone])
        return input_q
            
    def prepInput(self,normData):
      shape = self.pams['shape']
      
      if len(self.pams['arrange'])>0:
          arrange = self.pams['arrange']
          inputdata = normData[:,arrange]
      else:
          inputdata = normData
      if len(self.pams['arrMask'])>0:
          arrMask = self.pams['arrMask']
          inputdata[:,arrMask==0]=0  #zeros out repeated entries
      
      shaped_data = inputdata.reshape(len(inputdata),shape[0],shape[1],shape[2])

      if self.pams['n_copy']>0:
        n_copy  = self.pams['n_copy']
        occ_low = self.pams['occ_low']
        occ_hi = self.pams['occ_hi']
        shaped_data = self.cloneInput(shaped_data,n_copy,occ_low,occ_hi)

      return shaped_data

    def weightedMSE(self, y_true, y_pred):
        y_true = K.cast(y_true, y_pred.dtype)
        loss   = K.mean(K.square(y_true - y_pred)*K.maximum(y_pred,y_true),axis=(-1))
        return loss
            
    def init(self,printSummary=True):
        encoded_dim = self.pams['encoded_dim']

        CNN_layer_nodes   = self.pams['CNN_layer_nodes']
        CNN_kernel_size   = self.pams['CNN_kernel_size']
        CNN_pool          = self.pams['CNN_pool']
        Dense_layer_nodes = self.pams['Dense_layer_nodes'] #does not include encoded layer
        channels_first    = self.pams['channels_first']

        # fix to one cnn layer for now
        nnodes=CNN_layer_nodes[0] #8
        CNN_kernel=CNN_kernel_size[0] #3

        inputs = Input(shape=self.pams['shape'])
        x = inputs

        x1 = x[:,:,:,0]
        x2 = x[:,:,:,1]
        x3 = x[:,:,:,2]

        x1 = Reshape((4,4,1))(x1)
        x2 = Reshape((4,4,1))(x2)
        x3 = Reshape((4,4,1))(x3)

        # testing
        # print ("\n\nTESTING")
        # print( x.shape )
        # print( x2.shape )
        # #print( x[0].shape )
        # print ("\n")

        x1 = Conv2D(nnodes, CNN_kernel, activation='relu')(x1)
        x2 = Conv2D(nnodes, CNN_kernel, activation='relu')(x2)
        x3 = Conv2D(nnodes, CNN_kernel, activation='relu')(x3)

        x1 = Flatten()(x1)
        x2 = Flatten()(x2)
        x3 = Flatten()(x3)

        x = [x1,x2,x3]
        x = concatenate(x)

        shape = K.int_shape(x)
        #print("KINT SHAPE",shape)
        
        encodedLayer = Dense(encoded_dim, activation='relu',name='encoded_vector')(x)

        # Instantiate Encoder Model
        self.encoder = Model(inputs, encodedLayer, name='encoder')
        if printSummary:
          self.encoder.summary()
          plot_model(self.encoder, show_shapes=True, to_file="model1.png")

        encoded_inputs = Input(shape=(encoded_dim,), name='decoder_input')
        x = encoded_inputs

        x = Dense(96, activation='relu')(x)

        x = Reshape((2,2,nnodes,3))(x)
        x1 = x[:,:,:,:,0]
        x2 = x[:,:,:,:,1]
        x3 = x[:,:,:,:,2]

        nchan=1
        x1 = Conv2DTranspose(nchan, CNN_kernel, activation='relu')(x1)
        x2 = Conv2DTranspose(nchan, CNN_kernel, activation='relu')(x2)
        x3 = Conv2DTranspose(nchan, CNN_kernel, activation='relu')(x3)

        x = [x1,x2,x3]
        x = concatenate(x)

        outputs = Activation('sigmoid', name='decoder_output')(x)

        self.decoder = Model(encoded_inputs, outputs, name='decoder')
        if printSummary:
          self.decoder.summary()
          plot_model(self.decoder, show_shapes=True, to_file="model2.png")

        self.autoencoder = Model(inputs, self.decoder(self.encoder(inputs)), name='autoencoder')
        if printSummary:
          self.autoencoder.summary()
          plot_model(self.autoencoder, show_shapes=True, to_file="model.png")

        # if self.pams['loss']=="weightedMSE":
        #     self.autoencoder.compile(loss=self.weightedMSE, optimizer='adam')
        #     self.encoder.compile(loss=self.weightedMSE, optimizer='adam')
        # elif self.pams['loss'] == 'telescopeMSE':
        self.autoencoder.compile(loss=telescopeMSE2, optimizer='adam', run_eagerly=True)
        self.encoder.compile(loss=telescopeMSE2, optimizer='adam', run_eagerly=True)
        # elif self.pams['loss']!='':
        #     self.autoencoder.compile(loss=self.pams['loss'], optimizer='adam')
        #     self.encoder.compile(loss=self.pams['loss'], optimizer='adam')
        # else:
        #     self.autoencoder.compile(loss='mse', optimizer='adam')
        #     self.encoder.compile(loss='mse', optimizer='adam')

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
    def get_models(self):
       return self.autoencoder,self.encoder
           
    def predict(self,x):
        decoded_Q = self.autoencoder.predict(x)
        encoded_Q = self.encoder.predict(x)
        s = self.pams['shape'] 
        if self.pams['channels_first']:
            shaped_x  = np.reshape(x,(len(x),s[0]*s[1],s[2]))
            decoded_Q = np.reshape(decoded_Q,(len(decoded_Q),s[0]*s[1],s[2]))
            encoded_Q = np.reshape(encoded_Q,(len(encoded_Q),self.pams['encoded_dim'],1))
        else:
            shaped_x  = np.reshape(x,(len(x),s[2]*s[1],s[0]))
            decoded_Q = np.reshape(decoded_Q,(len(decoded_Q),s[2]*s[1],s[0]))
            encoded_Q = np.reshape(encoded_Q,(len(encoded_Q),self.pams['encoded_dim'],1))
        return shaped_x,decoded_Q, encoded_Q

    def summary(self):
      self.encoder.summary()
      self.decoder.summary()
      self.autoencoder.summary()

    ##get pams for writing json
    def get_pams(self):
      jsonpams={}
      for k,v in self.pams.items():
          if type(v)==type(np.array([])):
              jsonpams[k] = v.tolist()
          else:
              jsonpams[k] = v 
      return jsonpams   
      
