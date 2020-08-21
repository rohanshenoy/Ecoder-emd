import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, \
    Conv2DTranspose, Reshape, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import qkeras as qkr
from qkeras import QDense, QConv2D, QActivation
#from qkeras.qlayers import QConv2D,QActivation,QDense
import numpy as np
import json

# for sinkhorn metric
import ot_tf
import ot

from telescope import telescopeMSE2

hexCoords = np.array([ 
    [0.0, 0.0], [0.0, -2.4168015], [0.0, -4.833603], [0.0, -7.2504044], 
    [2.09301, -1.2083969], [2.09301, -3.6251984], [2.09301, -6.042], [2.09301, -8.458794], 
    [4.18602, -2.4168015], [4.18602, -4.833603], [4.18602, -7.2504044], [4.18602, -9.667198], 
    [6.27903, -3.6251984], [6.27903, -6.042], [6.27903, -8.458794], [6.27903, -10.875603], 
    [-8.37204, -10.271393], [-6.27903, -9.063004], [-4.18602, -7.854599], [-2.0930138, -6.6461945], 
    [-8.37204, -7.854599], [-6.27903, -6.6461945], [-4.18602, -5.4377975], [-2.0930138, -4.229393], 
    [-8.37204, -5.4377975], [-6.27903, -4.229393], [-4.18602, -3.020996], [-2.0930138, -1.8125992], 
    [-8.37204, -3.020996], [-6.27903, -1.8125992], [-4.18602, -0.6042023], [-2.0930138, 0.6042023], 
    [4.7092705, -12.386101], [2.6162605, -11.177696], [0.5232506, -9.969299], [-1.5697594, -8.760895], 
    [2.6162605, -13.594498], [0.5232506, -12.386101], [-1.5697594, -11.177696], [-3.6627693, -9.969299], 
    [0.5232506, -14.802895], [-1.5697594, -13.594498], [-3.6627693, -12.386101], [-5.7557793, -11.177696], 
    [-1.5697594, -16.0113], [-3.6627693, -14.802895], [-5.7557793, -13.594498], [-7.848793, -12.386101]])
hexMetric = tf.constant( ot.dist(hexCoords, hexCoords, 'euclidean'), tf.float32)

def myfunc(a):
    reg=0.5
    y_true, y_pred = tf.split(a,num_or_size_splits=2,axis=1)
    tf_sinkhorn_loss = ot_tf.sink(y_true, y_pred, hexMetric, (48, 48), reg)
    return tf_sinkhorn_loss
def sinkhorn_loss(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    y_pred = K.reshape(y_pred, (-1,48,1))
    y_true = K.reshape(y_true, (-1,48,1))
    cc = tf.concat([y_true, y_pred], axis=2)
    return K.mean( tf.map_fn(myfunc, cc), axis=(-1) )

SCmask = np.array([
    [ 0,  1,  4,  5], #indices for 1 supercell
    [ 2,  3,  6,  7],
    [ 8,  9, 12, 13],
    [10, 11, 14, 15],
    [16, 17, 20, 21],
    [18, 19, 22, 23],
    [24, 25, 28, 29],
    [26, 27, 30, 31],
    [32, 33, 36, 37],
    [34, 35, 38, 39],
    [40, 41, 44, 45],
    [43, 43, 46, 47]])
# 48 x 12 matrix mapping TCs to SC
Remap_48_12 = np.zeros((48,12))
for isc,sc in enumerate(SCmask): 
    for tc in sc: 
        Remap_48_12[tc,isc]=1
tf_Remap_48_12 = tf.constant(Remap_48_12,dtype=tf.float32)
# 12 x 3 mapping
Remap_12_3 = np.zeros((12,3))
for i in range(12): Remap_12_3[i,int(i/4)]=1
tf_Remap_12_3 = tf.constant(Remap_12_3,dtype=tf.float32)

def telescopeMSE(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)

    # TC-level MSE
    y_pred_rs = K.reshape(y_pred, (-1,48))
    y_true_rs = K.reshape(y_true, (-1,48))
    # lossTC1 = K.mean(K.square(y_true_rs - y_pred_rs), axis=(-1))
    lossTC1 = K.mean(K.square(y_true_rs - y_pred_rs) * K.maximum(y_pred_rs, y_true_rs), axis=(-1))

    # map TCs to 2x2 supercells and compute MSE
    y_pred_12 = tf.matmul(y_pred_rs, tf_Remap_48_12)
    y_true_12 = tf.matmul(y_true_rs, tf_Remap_48_12)
    # lossTC2 = K.mean(K.square(y_true_12 - y_pred_12), axis=(-1))
    lossTC2 = K.mean(K.square(y_true_12 - y_pred_12) * K.maximum(y_pred_12, y_true_12), axis=(-1))
  
    # map 2x2 supercells to 4x4 supercells and compute MSE
    y_pred_3 = tf.matmul(y_pred_12, tf_Remap_12_3)
    y_true_3 = tf.matmul(y_true_12, tf_Remap_12_3)
    # lossTC3 = K.mean(K.square(y_true_3 - y_pred_3), axis=(-1))
    lossTC3 = K.mean(K.square(y_true_3 - y_pred_3) * K.maximum(y_pred_3, y_true_3), axis=(-1))

    # sum MSEs
    #return lossTC1 + lossTC2 + lossTC3
    return 4*lossTC1 + 2*lossTC2 + lossTC3


class qDenseCNN:
    def __init__(self, name='', weights_f=''):
        self.name = name
        self.pams = {
            'CNN_layer_nodes': [8],  # n_filters
            'CNN_kernel_size': [3],
            'CNN_pool': [False],
            'Dense_layer_nodes': [],  # does not include encoded layer
            'encoded_dim': 12,
            'shape': (1, 4, 4),
            'channels_first': False,
            'arrange': [],
            'arrMask': [],
            'n_copy': 0,  # no. of copy for hi occ datasets
            'loss': '',
            'activation': 'relu',
        }

        self.weights_f = weights_f
        # self.extend = False

    def setpams(self, in_pams):
        for k, v in in_pams.items():
            self.pams[k] = v

    def shuffle(self, arr):
        order = np.arange(48)
        np.random.shuffle(order)
        return arr[:, order]

    def cloneInput(self, input_q, n_copy, occ_low, occ_hi):
        shape = self.pams['shape']
        nonzeroQs = np.count_nonzero(input_q.reshape(len(input_q), 48), axis=1)
        selection = np.logical_and(nonzeroQs <= occ_hi, nonzeroQs > occ_low)
        occ_q = input_q[selection]
        occ_q_flat = occ_q.reshape(len(occ_q), 48)
        self.pams['cloned_fraction'] = len(occ_q) / len(input_q)
        for i in range(0, n_copy):
            clone = self.shuffle(occ_q_flat)
            clone = clone.reshape(len(clone), shape[0], shape[1], shape[2])
            input_q = np.concatenate([input_q, clone])
        return input_q

    def prepInput(self, normData):
        shape = self.pams['shape']

        if len(self.pams['arrange']) > 0:
            arrange = self.pams['arrange']
            inputdata = normData[:, arrange]
        else:
            inputdata = normData
        if len(self.pams['arrMask']) > 0:
            arrMask = self.pams['arrMask']
            inputdata[:, arrMask == 0] = 0  # zeros out repeated entries

        shaped_data = inputdata.reshape(len(inputdata), shape[0], shape[1], shape[2])

        if self.pams['n_copy'] > 0:
            n_copy = self.pams['n_copy']
            occ_low = self.pams['occ_low']
            occ_hi = self.pams['occ_hi']
            shaped_data = self.cloneInput(shaped_data, n_copy, occ_low, occ_hi)
        # if self.pams['skimOcc']:
        #  occ_low = self.pams['occ_low']
        #  occ_hi = self.pams['occ_hi']
        #  nonzeroQs = np.count_nonzero(shaped_data.reshape(len(shaped_data),48),axis=1)
        #  selection = np.logical_and(nonzeroQs<=occ_hi,nonzeroQs>occ_low)
        #  shaped_data     = shaped_data[selection]

        return shaped_data

    def weightedMSE(self, y_true, y_pred):
        y_true = K.cast(y_true, y_pred.dtype)
        loss = K.mean(K.square(y_true - y_pred) * K.maximum(y_pred, y_true), axis=(-1))
        return loss
        
    def GetQbits(self, inp, keep_negative=1):
        print("Setting bits {} {} with keep negative = {}".format(inp['total'], inp['integer'], keep_negative))
        b =  qkr.quantized_bits(bits=inp['total'], integer=inp['integer'], keep_negative=keep_negative, alpha=1)
        print('max = %s, min = %s'%(b.max(),b.min()))
        print('str representation:%s'%(str(b)))
        print('config = ',b.get_config())
        return b
        
    def init(self, printSummary=True): # keep_negitive = 0 on inputs, otherwise for weights keep default (=1)
        encoded_dim = self.pams['encoded_dim']

        CNN_layer_nodes = self.pams['CNN_layer_nodes']
        CNN_kernel_size = self.pams['CNN_kernel_size']
        CNN_pool = self.pams['CNN_pool']
        Dense_layer_nodes = self.pams['Dense_layer_nodes']  # does not include encoded layer
        channels_first = self.pams['channels_first']

        inputs = Input(shape=self.pams['shape'])  # adapt this if using `channels_first` image data format

        # load bits to quantize
        nBits_input  = self.pams['nBits_input']
        nBits_accum  = self.pams['nBits_accum']
        nBits_weight = self.pams['nBits_weight']
        nBits_encod  = self.pams['nBits_encod']
        nBits_dense  = self.pams['nBits_dense'] if 'nBits_dense' in self.pams else nBits_weight
        nBits_conv   = self.pams['nBits_conv' ] if 'nBits_conv'  in self.pams else nBits_weight

        input_Qbits  = self.GetQbits(nBits_input, nBits_input['keep_negative']) 
        accum_Qbits  = self.GetQbits(nBits_accum, nBits_accum['keep_negative'])
        dense_Qbits  = self.GetQbits(nBits_dense, nBits_dense['keep_negative'])
        conv_Qbits   = self.GetQbits(nBits_conv , nBits_conv ['keep_negative'])
        encod_Qbits  = self.GetQbits(nBits_encod, nBits_encod['keep_negative'])
        # keeping weights and bias same precision for now

        # define model
        x = inputs
        x = QActivation(input_Qbits, name='input_qa')(x)
        for i, n_nodes in enumerate(CNN_layer_nodes):
            if channels_first:
                x = QConv2D(n_nodes, CNN_kernel_size[i], activation='relu', padding='same',
                            data_format='channels_first', name="conv2d_"+str(i)+"_m",
                            kernel_quantizer=conv_Qbits, bias_quantizer=conv_Qbits)(x)
            else:
                x = QConv2D(n_nodes, CNN_kernel_size[i], activation='relu', padding='same', name="conv2d_"+str(i)+"_m",
                            kernel_quantizer=conv_Qbits, bias_quantizer=conv_Qbits)(x)
            if CNN_pool[i]:
                if channels_first:
                    x = MaxPooling2D((2, 2), padding='same', data_format='channels_first', name="mp_"+str(i))(x)
                else:
                    x = MaxPooling2D((2, 2), padding='same', name="mp_"+str(i))(x)

        shape = K.int_shape(x)
        x = QActivation(accum_Qbits, name='accum1_qa')(x)
        x = Flatten(name="flatten")(x)
        
        # extended inputs fed forward to the dense layer
        # if self.extend:
        #     inputs2 = Input(shape=(2,))  # maxQ, occupancy
            # input2_Qbits  = self.GetQbits(nBits_input, keep_negative=1) #oddly fails if keep_neg=0
            # input2_Qbits
            # x = inputs
            # x = QActivation(input_Qbits, name='input_qa')(x)
            

        # encoder dense nodes
        for i, n_nodes in enumerate(Dense_layer_nodes):
            x = QDense(n_nodes, activation='relu', name="en_dense_"+str(i),
                           kernel_quantizer=dense_Qbits, bias_quantizer=dense_Qbits)(x)


        #x = QDense(encoded_dim, activation='relu', name='encoded_vector',
        #                      kernel_quantizer=dense_Qbits, bias_quantizer=dense_Qbits)(x)
        x = QDense(encoded_dim, activation=self.pams['activation'], name='encoded_vector',
                              kernel_quantizer=dense_Qbits, bias_quantizer=dense_Qbits)(x)
        encodedLayer = QActivation(encod_Qbits, name='encod_qa')(x)

        # Instantiate Encoder Model
        self.encoder = Model(inputs, encodedLayer, name='encoder')
        if printSummary:
            self.encoder.summary()

        encoded_inputs = Input(shape=(encoded_dim,), name='decoder_input')
        x = encoded_inputs

        # decoder dense nodes
        for i, n_nodes in enumerate(Dense_layer_nodes):
            x = Dense(n_nodes, activation='relu', name="de_dense_"+str(i))(x)

        x = Dense(shape[1] * shape[2] * shape[3], activation='relu', name='de_dense_final')(x)
        x = Reshape((shape[1], shape[2], shape[3]),name="de_reshape")(x)

        for i, n_nodes in enumerate(CNN_layer_nodes):

            if CNN_pool[i]:
                if channels_first:
                    x = UpSampling2D((2, 2), data_format='channels_first', name="up_"+str(i))(x)
                else:
                    x = UpSampling2D((2, 2), name="up_"+str(i))(x)

            if channels_first:
                x = Conv2DTranspose(n_nodes, CNN_kernel_size[i], activation='relu', padding='same',
                                    data_format='channels_first', name="conv2D_t_"+str(i))(x)
            else:
                x = Conv2DTranspose(n_nodes, CNN_kernel_size[i], activation='relu', padding='same',
                                    name="conv2D_t_"+str(i))(x)

        if channels_first:
            # shape[0] will be # of channel
            x = Conv2DTranspose(filters=self.pams['shape'][0], kernel_size=CNN_kernel_size[0], padding='same',
                                data_format='channels_first', name="conv2d_t_final")(x)

        else:
            x = Conv2DTranspose(filters=self.pams['shape'][2], kernel_size=CNN_kernel_size[0], padding='same',
                                name="conv2d_t_final")(x)
        x = QActivation(input_Qbits, name='q_decoder_output')(x) #Verify this step needed?
        outputs = Activation('sigmoid', name='decoder_output')(x)

        self.decoder = Model(encoded_inputs, outputs, name='decoder')
        if printSummary:
            self.decoder.summary()

        self.autoencoder = Model(inputs, self.decoder(self.encoder(inputs)), name='autoencoder')
        if printSummary:
            self.autoencoder.summary()

        if self.pams['loss'] == "weightedMSE":
            self.autoencoder.compile(loss=self.weightedMSE, optimizer='adam')
            self.encoder.compile(loss=self.weightedMSE, optimizer='adam')
        elif self.pams['loss'] == 'telescopeMSE':
            self.autoencoder.compile(loss=telescopeMSE2, optimizer='adam')
            self.encoder.compile(loss=telescopeMSE2, optimizer='adam')
        elif self.pams['loss'] == 'sink':
            self.autoencoder.compile(loss=sinkhorn_loss, optimizer='adam')
            self.encoder.compile(loss=sinkhorn_loss, optimizer='adam')
        elif self.pams['loss'] != '':
            self.autoencoder.compile(loss=self.pams['loss'], optimizer='adam')
            self.encoder.compile(loss=self.pams['loss'], optimizer='adam')
        else:
            self.autoencoder.compile(loss='mse', optimizer='adam')
            self.encoder.compile(loss='mse', optimizer='adam')

        CNN_layers = ''
        if len(CNN_layer_nodes) > 0:
            CNN_layers += '_Conv'
            for i, n in enumerate(CNN_layer_nodes):
                CNN_layers += f'_{n}x{CNN_kernel_size[i]}'
                if CNN_pool[i]:
                    CNN_layers += 'pooled'
        Dense_layers = ''
        if len(Dense_layer_nodes) > 0:
            Dense_layers += '_Dense'
            for n in Dense_layer_nodes:
                Dense_layers += f'_{n}'

        self.name = f'Autoencoded{CNN_layers}{Dense_layers}_Encoded_{encoded_dim}'

        if not self.weights_f == '':
            self.autoencoder.load_weights(self.weights_f)

    def get_models(self):
        return self.autoencoder, self.encoder

    def invertArrange(self,arrange):
        remap =[]
        hashmap = {}
        for i in range(len(arrange)):
            hashmap[arrange[i]]=i
        for i in range(len(arrange)):        
            remap.append(hashmap[i])
        return np.array(remap)

    ## remap input/output of autoencoder into CALQs orders
    def mapToCalQ(self,x):
        if len(self.pams['arrange']) > 0:
            arrange = self.pams['arrange']
            remap   = self.invertArrange(arrange)
            return x.reshape(len(x),48)[:,remap]
        else:
            return x.reshap(len(x),48)
        

    def predict(self, x):
        decoded_Q = self.autoencoder.predict(x)
        encoded_Q = self.encoder.predict(x)
        encoded_Q = np.reshape(encoded_Q, (len(encoded_Q), self.pams['encoded_dim'], 1))
        #if self.pams['channels_first']:
        #    shaped_x = np.reshape(x, (len(x), s[0] * s[1], s[2]))
        #    decoded_Q = np.reshape(decoded_Q, (len(decoded_Q), s[0] * s[1], s[2]))
        #    encoded_Q = np.reshape(encoded_Q, (len(encoded_Q), self.pams['encoded_dim'], 1))
        #else:
        #    shaped_x = np.reshape(x, (len(x), s[2] * s[1], s[0]))
        #    decoded_Q = np.reshape(decoded_Q, (len(decoded_Q), s[2] * s[1], s[0]))
        return x, decoded_Q, encoded_Q

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.autoencoder.summary()

    ##get pams for writing json
    def get_pams(self):
        jsonpams = {}
        for k, v in self.pams.items():
            if type(v) == type(np.array([])):
                jsonpams[k] = v.tolist()
            else:
                jsonpams[k] = v
        return jsonpams

