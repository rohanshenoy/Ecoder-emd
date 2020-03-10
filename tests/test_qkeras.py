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

def main():
    test_inputs()
    test_dense()
    exit(0)

def test_inputs():
    data = getData()
    
    nBits=8
    nBitsInt=4
    qbits_param_input = qkr.quantized_bits(bits=nBits,integer=nBitsInt,keep_negative=0)

    # simple model only quantizes
    inputs = Input(shape=(4, 4, 3))
    x = inputs
    x = Flatten(name="flatten")(x)
    x = QActivation(qbits_param_input, name='q_decoder_output')(x)
    model = Model(inputs, x, name='encoder')
    model.summary()
    
    model.compile(loss='mse', optimizer='adam')
    
    val_input, train_input = split(data, 0.5)
    train_output = np.ones(240).reshape(5,48) # garbage outputs for training
    val_output   = np.ones(240).reshape(5,48) # garbage outputs for validation

    es = kr.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    history = model.fit(train_input, train_output,
                        epochs=1,
                        batch_size=500,
                        shuffle=True,
                        validation_data=(val_input,val_output),
                        callbacks=[es])
    
    val_output = model.predict(val_input)

    print('\nTEST INPUTS')
    print('\nRaw validation output: \n',val_output)
    print('\nMultiplied by 2^(decimal bits): \n',val_output*(2**(nBits-nBitsInt)))    
    return


def test_dense():
    data = getData()
    
    nBits=8
    nBitsInt=4
    qbits_param_input = qkr.quantized_bits(bits=nBits,integer=nBitsInt,keep_negative=0)
    qbits_param = qkr.quantized_bits(bits=nBits,integer=nBitsInt,keep_negative=1)

    # simple model only quantizes
    inputs = Input(shape=(4, 4, 3))
    x = inputs
    x = Flatten(name="flatten")(x)
    x = QActivation(qbits_param_input, name='q_decoder_output')(x)
    encodedLayer = QDense(10, activation='relu', name='encoded_vector',
                          kernel_quantizer=qbits_param, bias_quantizer=qbits_param)(x)
    model = Model(inputs, encodedLayer, name='encoder')
    model.summary()
    
    model.compile(loss='mse', optimizer='adam')
    
    val_input, train_input = split(data, 0.5)
    train_output = np.ones(50).reshape(5,10) # garbage outputs for training
    val_output   = np.ones(50).reshape(5,10) # garbage outputs for validation

    es = kr.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    history = model.fit(train_input, train_output,
                        epochs=1,
                        batch_size=500,
                        shuffle=True,
                        validation_data=(val_input,val_output),
                        callbacks=[es])
    
    val_output = model.predict(val_input)

    print('\nTEST DENSE')
    print('\nRaw validation output: \n',val_output)
    print('\nMultiplied by 2^(decimal bits): \n  Results should be integers * weight precision... \n',val_output*(2**(nBits-nBitsInt)))
    return



def getData():
    # read 10 events of 48 inputs
    d = np.array(
        [[73,6,13,41,28,21,37,48,13,46,51,10,29,19,19,53,0,0,4,12,0,45,20,79,26,68,23,61,35,14,14,36,9,52,15,21,0,0,0,0,0,0,0,0,0,0,0,0],
         [41,59,3,0,32,24,12,6,18,140,24,53,34,44,127,15,0,0,0,0,0,0,0,0,0,0,0,7,0,0,3,13,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [25,11,24,56,10,33,9,33,70,35,29,0,3,22,5,53,0,0,0,118,0,0,0,21,0,0,0,25,0,0,0,44,32,68,21,11,8,16,47,0,47,34,14,0,82,0,0,0],
         [0,0,0,0,0,0,0,9,22,26,25,42,5,105,23,56,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,58,103,29,0,86,23,0,0,62,8,0,0,75,7,0,0],
         [0,0,38,61,0,10,27,36,26,6,43,53,42,24,7,13,37,59,29,78,0,0,0,0,0,0,0,0,0,0,0,0,15,14,20,55,26,27,55,97,73,31,66,112,9,25,28,42],
         [0,0,0,0,0,0,0,0,0,0,0,14,0,0,0,60,23,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,10,3,0,2,44,29,10,4,7,9,53,67,10,39,124],
         [0,0,0,23,0,0,0,52,0,0,0,50,0,0,7,11,79,45,18,30,34,627,23,20,58,44,28,0,76,13,0,0,64,7,31,21,24,14,40,20,47,70,85,25,41,73,10,8],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,62,14,0,50,17,10,0,111,42,0,0,8,6,0,0,0,0,0,0,0,0,9,19,10,21,37,76,19,63,58,37],
         [160,8,15,104,0,0,0,0,0,0,0,0,0,0,0,0,6,10,29,13,14,21,73,38,45,23,95,139,8,43,13,35,0,0,127,40,0,24,22,40,26,62,79,23,167,80,61,32],
         [29,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,28,33,0,0,49,47,52,5,44,26,37,59,9,10,85,22,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,2],],
        dtype=np.float32)
    # normalize
    for i in range(len(d)): 
        d[i] = d[i]*1./d[i].sum()
        
    #sort inputs
    arrange443 = np.array([0,16, 32,
                           1,17, 33,
                           2,18, 34,
                           3,19, 35,
                           4,20, 36,
                           5,21, 37,
                           6,22, 38,
                           7,23, 39,
                           8,24, 40,
                           9,25, 41,
                           10,26, 42,
                           11,27, 43,
                           12,28, 44,
                           13,29, 45,
                           14,30, 46,
                           15,31, 47])
    d = d[:, arrange443]    
    d = d.reshape(len(d),4,4,3)
    return d

def split(shaped_data, validation_frac):
  N = round(len(shaped_data)*validation_frac)
  
  #randomly select 25% entries
  index = np.random.choice(shaped_data.shape[0], N, replace=False)  
  #select the indices of the other 75%
  full_index = np.array(range(0,len(shaped_data)))
  train_index = np.logical_not(np.in1d(full_index,index))
  
  val_input = shaped_data[index]
  train_input = shaped_data[train_index]

  print(train_input.shape)
  print(val_input.shape)

  return val_input,train_input

    

if __name__ == "__main__":
    main()
