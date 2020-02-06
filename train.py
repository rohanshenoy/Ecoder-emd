import keras as kr
import numpy as np
import tensorflow as tf
import pandas as pd
import optparse

import os
import matplotlib.pyplot as plt
from models import *

import numba
from denseCNN import denseCNN 
@numba.jit
def normalize(data):
    for i in range(len(data)):
        data[i] = 1.*data[i]/data[i].max()
    return data

def prepInput(shape=[],nrows=None,reorder=False):
  data = pd.read_csv("CALQ_output_10x.csv",nrows=nrows)  ## big  300k file
  
  inputdata = normalize(data.values.copy())

  if reorder:
      newOrder = np.array([
          28,29,30,31,0,4,8,12,
          24,25,26,27,1,5,9,13,
          20,21,22,23,2,6,10,14,
          16,17,18,19,3,7,11,15,
          47,43,39,35,35,34,33,32,
          46,42,38,34,39,38,37,36,
          45,41,37,33,43,42,41,40,
          44,40,36,32,47,46,45,44])
      orderMask = np.array([
              1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,
              1,1,1,0,0,1,1,1,
              1,1,0,0,0,0,0,1,
              1,0,0,0,0,0,0,1,])
      inputdata = inputdata[:,newOrder]
      inputdata[:,orderMask==0]=0  #zeros out repeated entries

  if len(shape)>0: 
    shaped_data = np.reshape(inputdata,(len(inputdata),shape[0],shape[1],shape[2]))
  else:
    shaped_data = inputdata

def split(shaped_data, validation_frac=0.2):
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

def train(autoencoder,encoder,train_input,val_input,name):

  es = kr.callbacks.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
  history = autoencoder.fit(train_input,train_input,
                epochs=150,
                batch_size=200,
                shuffle=True,
                validation_data=(val_input,val_input),
                callbacks=[es]
                )

  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss %s'%name)
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.savefig("history_%s.jpg"%name)
  #plt.show()

  
  json_string = autoencoder.to_json()
  with open('./%s.json'%name,'w') as f:
      f.write(json_string)
  json_string = encoder.to_json()
  with open('./%s.json'%("encoder_"+name),'w') as f:
      f.write(json_string)
  autoencoder.save_weights('%s.hdf5'%name)
  encoder.save_weights('%s.hdf5'%("encoder_"+name))

  return history

def predict(x,autoencoder,encoder,reshape=True):
  decoded_Q = autoencoder.predict(x)
  encoded_Q = encoder.predict(x)
 
  #need reshape for CNN layers  
  if reshape :
    decoded_Q = np.reshape(decoded_Q,(len(decoded_Q),12,4))
    encoded_shape = encoded_Q.shape
    encoded_Q = np.reshape(encoded_Q,(len(encoded_Q),encoded_shape[3],encoded_shape[1]))
  return decoded_Q, encoded_Q

### sum of squared difference
def SSD(x,y):
    ssd  = np.sum(((x-y)**2),(1,2)) ## sum x-y dim.
    return ssd

### cross correlation of input/output 
def cross_corr(x,y):
    cov = np.cov(x.flatten(),y.flatten())
    std = np.sqrt(np.diag(cov))
    corr = cov / np.multiply.outer(std, std)
    return corr[0,1]

def visualize(input_Q,decoded_Q,encoded_Q,index,name='model_X'):
  if index.size==0:
    Nevents=8
    #randomly pick Nevents if index is not specified
    index = np.random.choice(input_Q.shape[0], Nevents, replace=False) 
  else:
    Nevents = len(index) 
  
  inputImg    = input_Q[index]
  encodedImg  = encoded_Q[index]
  outputImg   = decoded_Q[index]
  
  fig, axs = plt.subplots(3, Nevents, figsize=(16, 10))
  
  for i in range(0,Nevents):
      if i==0:
          axs[0,i].set(xlabel='',ylabel='cell_y',title='Input_%i'%i)
      else:
          axs[0,i].set(xlabel='',title='Input_%i'%i)        
      c1=axs[0,i].imshow(inputImg[i])
 
  for i in range(0,Nevents):
      if i==0:
          axs[1,i].set(xlabel='cell_x',ylabel='cell_y',title='CNN Ouput_%i'%i)        
      else:
          axs[1,i].set(xlabel='cell_x',title='CNN Ouput_%i'%i)
      c1=axs[1,i].imshow(outputImg[i])
     
  for i in range(0,Nevents):
    if i==0:
        axs[2,i].set(xlabel='latent dim',ylabel='depth',title='Encoded_%i'%i)
    else:
        axs[2,i].set(xlabel='latent dim',title='Encoded_%i'%i)
    c1=axs[2,i].imshow(encodedImg[i])
    plt.colorbar(c1,ax=axs[2,i])
  
  plt.tight_layout()
  plt.savefig("%s_examples.jpg"%name)
 
def visMetric(input_Q,decoded_Q,name): 
  #plt.show()
  def plothist(y,xlabel,name):
    plt.figure(figsize=(6,4))
    plt.hist(y,50)
    
    mu = np.mean(y)
    std = np.std(y)
    ax = plt.axes()
    plt.text(0.1, 0.9, name,transform=ax.transAxes)
    plt.text(0.1, 0.8, r'$\mu=%.3f,\ \sigma=%.3f$'%(mu,std),transform=ax.transAxes)
    plt.xlabel(xlabel)
    plt.ylabel('Entry')
    plt.title('%s on validation set'%xlabel)
    plt.savefig("hist_%s.jpg"%name)
    #plt.show()

  cross_corr_arr = np.array( [cross_corr(input_Q[i],decoded_Q[i]) for i in range(0,len(decoded_Q))]  )
  ssd_arr        = np.sum((input_Q-decoded_Q)**2,(1,2))

  plothist(cross_corr_arr,'cross correlation',name+"_corr")
  plothist(ssd_arr,'sum squared difference',name+"_ssd")

  return cross_corr_arr,ssd_arr


def vis1d(input_Q,decoded_Q,encoded_Q,index,name='model_X'):
  #Nevents = 50
  #index = np.random.choice(encoded_Q.shape[0], Nevents, replace=False)  
  
  fig, axs = plt.subplots(1, 4, figsize=(16, 5))
  fig.suptitle(name,fontsize=16)
  
  axs[0].set(xlabel='ChargeQ',title='Input layer')
  axs[1].set(xlabel='Encoded unit',title='Encoded layer')
  axs[2].set(xlabel='ChargeQ'     ,title='Decoded layer')
  axs[3].set(xlabel='(Decoded - Input)**2',title='(Decoded - Input)**2')
  c1=axs[0].imshow(input_Q[index])
  c2=axs[1].imshow(encoded_Q[index])
  c3=axs[2].imshow(decoded_Q[index])
  c4=axs[3].imshow((decoded_Q[index]-input_Q[index])**2)
  plt.colorbar(c1,ax=axs[0])
  plt.colorbar(c2,ax=axs[1])
  plt.colorbar(c3,ax=axs[2])
  plt.colorbar(c4,ax=axs[3])
  plt.savefig("%s.jpg"%name)
  #plt.show()

  def plothist(y,xlabel,name):
    plt.figure(figsize=(6,4))
    plt.hist(y,50)
    
    mu = np.mean(y)
    std = np.std(y)
    ax = plt.axes()
    plt.text(0.1, 0.9, name,transform=ax.transAxes)
    plt.text(0.1, 0.8, r'$\mu=%.3f,\ \sigma=%.3f$'%(mu,std),transform=ax.transAxes)
    plt.xlabel(xlabel)
    plt.ylabel('Entry')
    plt.title('%s on validation set'%xlabel)
    plt.savefig("hist_%s.jpg"%name)
    #plt.show()

  cross_corr_arr = np.array( [cross_corr(input_Q[i],decoded_Q[i]) for i in range(0,len(decoded_Q))]  )
  ssd_arr        = np.sum((input_Q-decoded_Q)**2,1)

  plothist(cross_corr_arr,'cross correlation',name+"_corr")
  plothist(ssd_arr,'sum squared difference',name+"_ssd")

  return cross_corr_arr,ssd_arr

def trainDeepAutoEncoder(options,args):

  nrows = 30000
  Nevents = 50
  full_input, val_input, train_input = prepInput(reshape=False,nrows=nrows)

  index = np.random.choice(val_input.shape[0], Nevents, replace=False)  
  models = [
    {'dims':[48,24,12,6  ]         ,'ws':'./deepAutos_48_24_12_6.hdf5'},
    {'dims':[48,24,12,6,4]         ,'ws':'./deepAutos_48_24_12_6_4.hdf5'},
    {'dims':[48,24,12,8,4]         ,'ws':'./deepAutos_48_24_12_8_4.hdf5'},
    {'dims':[48,24,12,6,3]         ,'ws':'./deepAutos_48_24_12_6_3.hdf5'},
    {'dims':[48,24,12,6,2]         ,'ws':'./deepAutos_48_24_12_6_2.hdf5'},
    {'dims':[48,24,12,10,6  ]      ,'ws':'./deepAutos_48_24_12_10_6.hdf5'},
    {'dims':[48,24,12,10,10,10,6  ],'ws':'./deepAutos_48_24_12_10_10_10_6.hdf5'},
    {'dims':[48,6,2]               ,'ws':'./deepAutos_48_6_2.hdf5'}     
  ]

  summary = pd.DataFrame(columns=['name','corr','ssd'])
  os.chdir('./deepAutos/')
  for model in models:
    dims = model['dims']
    model_name = "deepAutos_"+"_".join([str(d) for d in dims])
    if not os.path.exists(model_name):
      os.mkdir(model_name)
    os.chdir(model_name)
    m_deepAuto, m_deepAutoEn = deepAuto(dims,model['ws'])
    #m_deepAuto.summary()
    #m_deepAutoEn.summary()
    if model['ws']=='':
      history = train(m_deepAuto,m_deepAutoEn,train_input,val_input,name=model_name)
    deQ, enQ = predict(val_input,m_deepAuto,m_deepAutoEn,False)
    corr_arr, ssd_arr = vis1d(val_input,deQ,enQ,index,model_name)
    model['corr'] = np.round(np.mean(corr_arr),3)
    model['ssd'] = np.round(np.mean(ssd_arr),3)

    summary = summary.append({'name':model_name,'corr':model['corr'],'ssd':model['ssd']},ignore_index=True)
    for k in model.keys():
      print(k,model[k])
    os.chdir('../')
  print(summary)

def trainCNN(options,args):

  data = pd.read_csv("CALQ_output_10x.csv",dtype=np.float64)  ## big  300k file
  normdata = normalize(data.values.copy())

  models = [
    #{'filters':[16,8,4]         ,'ws':'CNN16_8_4.hdf5'},
    #{'filters':[16,8,4]         ,'ws':'CNN16_8_4.hdf5'},
    #{'name':'denseCNN',  'ws':'denseCNN.hdf5', 'pams':{'shape':(1,8,8) } },
    {'name':'denseCNN_2',  'ws':'denseCNN_2.hdf5', 'pams':{'shape':(8,8,1) } },
  ]

  summary = pd.DataFrame(columns=['name','corr','ssd'])
  os.chdir('./CNN/')
  for model in models:
    model_name = model['name']
    if not os.path.exists(model_name):
      os.mkdir(model_name)
    os.chdir(model_name)
   
    m = denseCNN(weights_f=model['ws']) 
    m.setpams(model['pams'])
    m.init()
    shaped_data                = m.prepInput(normdata)
    val_input, train_input     = split(shaped_data)
    m_autoCNN , m_autoCNNen    = m.get_models()
    if model['ws']=='':
      history = train(m_autoCNN,m_autoCNNen,train_input,val_input,name=model_name)

    Nevents = 8 
    input_Q            = np.reshape(val_input,(len(val_input),8,8))
    index = np.random.choice(input_Q.shape[0], Nevents, replace=False)  

    cnn_deQ ,cnn_enQ   = m.predict(val_input)
    corr_arr, ssd_arr  = visMetric(input_Q,cnn_deQ,name=model_name)

    visualize(input_Q,cnn_deQ,cnn_enQ,index,name=model_name)
    index = np.random.choice(np.where(corr_arr>0.9)[0], Nevents, replace=False)  
    visualize(input_Q,cnn_deQ,cnn_enQ,index,name=model_name+"_corr0.9")
    index = np.random.choice(np.where(corr_arr<0.2)[0], Nevents, replace=False)  
    visualize(input_Q,cnn_deQ,cnn_enQ,index,name=model_name+"_corr0.1")

    model['corr'] = np.round(np.mean(corr_arr),3)
    model['ssd'] = np.round(np.mean(ssd_arr),3)

    summary = summary.append({'name':model_name,'corr':model['corr'],'ssd':model['ssd']},ignore_index=True)

    #print('CNN ssd: ' ,np.round(SSD(input_Q,cnn_deQ),3))

  print(summary)


if __name__== "__main__":

    parser = optparse.OptionParser()
    parser.add_option('-o',"--odir", type="string", default = 'ntuple.root',dest="inputFile", help="input TSG ntuple")

    (options, args) = parser.parse_args()
    #trainDeepAutoEncoder(options,args)
    trainCNN(options,args)


