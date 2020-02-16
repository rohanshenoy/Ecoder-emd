import keras as kr
import numpy as np
import tensorflow as tf
import pandas as pd
import optparse

import os
import matplotlib.pyplot as plt
from models import *

import numba
import json
from denseCNN import denseCNN 

@numba.jit
def normalize(data):
    norm =[]
    for i in range(len(data)):
        norm.append( data[i].max() )
        data[i] = 1.*data[i]/data[i].max()
    return data,np.array(norm)

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

def train(autoencoder,encoder,train_input,val_input,name,n_epochs=100):

  es = kr.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
  history = autoencoder.fit(train_input,train_input,
                epochs=n_epochs,
                batch_size=500,
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

### cross correlation of input/output 
def cross_corr(x,y):
    cov = np.cov(x.flatten(),y.flatten())
    std = np.sqrt(np.diag(cov))
    corr = cov / np.multiply.outer(std, std)
    return corr[0,1]

def ssd(x,y):
    ssd=np.sum(((x-y)**2).flatten())
    ssd = ssd/(np.sum(x**2)*np.sum(y**2))**0.5
    return ssd


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
 
def visMetric(input_Q,decoded_Q,maxQ,name): 
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


  cross_corr_arr = np.array( [cross_corr(input_Q[i],decoded_Q[i]) for i in range(0,len(decoded_Q))]  )
  ssd_arr        = np.array([ssd(decoded_Q[i],input_Q[i]) for i in range(0,len(decoded_Q))])

  plothist(cross_corr_arr,'cross correlation',name+"_corr")
  plothist(ssd_arr,'sum squared difference',name+"_ssd")

  plt.figure(figsize=(6,4))
  plt.hist([input_Q.flatten(),decoded_Q.flatten()],20,label=['input','output'])
  plt.yscale('log')
  plt.legend(loc='upper right')
  plt.xlabel('Charge fraction')
  plt.savefig("hist_Qfr_%s.jpg"%name)

  input_Q_abs   = np.array([input_Q[i] * maxQ[i] for i in range(0,len(input_Q))])
  decoded_Q_abs = np.array([decoded_Q[i]*maxQ[i] for i in range(0,len(decoded_Q))])

  plt.figure(figsize=(6,4))
  plt.hist([input_Q_abs.flatten(),decoded_Q_abs.flatten()],20,label=['input','output'])
  plt.yscale('log')
  plt.legend(loc='upper right')
  plt.xlabel('Charge')
  plt.savefig("hist_Qabs_%s.jpg"%name)

  nonzeroQs = np.count_nonzero(input_Q_abs.reshape(len(input_Q_abs),48),axis=1)
  occbins = [0,5,10,20,48]
  fig, axes = plt.subplots(1,len(occbins)-1, figsize=(16, 4))
  for i,ax in enumerate(axes):
      selection=np.logical_and(nonzeroQs<occbins[i+1],nonzeroQs>occbins[i])
      label = '%i<occ<%i'%(occbins[i],occbins[i+1])
      mu = np.mean(cross_corr_arr[selection])
      std = np.std(cross_corr_arr[selection])
      plt.text(0.1, 0.8, r'$\mu=%.3f,\ \sigma=%.3f$'%(mu,std),transform=ax.transAxes)
      ax.hist(cross_corr_arr[selection],40)
      ax.set(xlabel='corr',title=label)
  plt.tight_layout()
  #plt.show()
  plt.savefig('corr_vs_occ_%s.jpg'%name)

  return cross_corr_arr,ssd_arr

def trainCNN(options,args):

  data = pd.read_csv(options.inputFile,dtype=np.float64)  ## big  300k file
  normdata,maxdata = normalize(data.values.copy())

  arrange8x8 = np.array([
              28,29,30,31,0,4,8,12,
              24,25,26,27,1,5,9,13,
              20,21,22,23,2,6,10,14,
              16,17,18,19,3,7,11,15,
              47,43,39,35,35,34,33,32,
              46,42,38,34,39,38,37,36,
              45,41,37,33,43,42,41,40,
              44,40,36,32,47,46,45,44])
  arrMask  =  np.array([
              1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,
              1,1,1,0,0,1,1,1,
              1,1,0,0,0,0,0,1,
              1,0,0,0,0,0,0,1,])

  models = [
    #{'name':'denseCNN',  'ws':'denseCNN.hdf5', 'pams':{'shape':(1,8,8) } },
    #{'name':'denseCNN_2',  'ws':'denseCNN_2.hdf5', 
    #  'pams':{'shape':(8,8,1) ,'arrange': arrange8x8,'arrMask':arrMask  } },

    #{'name':'8x8_nomask','ws':'','pams':{'shape':(8,8,1) ,'arrange': arrange8x8  }},
    #{'name':'nfil4','ws':'','pams':{'shape':(8,8,1) ,'arrange': arrange8x8,'arrMask':arrMask,  'CNN_layer_nodes':[4]}},
    #{'name':'nfils_842','ws':'nfils_842.hdf5','pams':{'shape':(8,8,1) ,'arrange': arrange8x8,'arrMask':arrMask,
    #        'CNN_layer_nodes':[8,4,2],
    #        'CNN_kernel_size':[3,3,3],
    #        'CNN_pool':[False,False,False],     
    #}} , 
    #{'name':'nfils_842_pool2','ws':'','pams':{'shape':(8,8,1) ,'arrange': arrange8x8,'arrMask':arrMask,
    #        'CNN_layer_nodes':[8,4,2],
    #        'CNN_kernel_size':[3,3,3],
    #        'CNN_pool':[False,True,False],     
    #}} , 
    #{'name':'8x8_dim10','ws':'','vis_shape':(8,8),'pams':{'shape':(8,8,1) ,'arrange': arrange8x8,'arrMask':arrMask,  'encoded_dim':10}},
    #{'name':'8x8_dim8','ws':'','pams':{'shape':(8,8,1) ,'arrange': arrange8x8,'arrMask':arrMask,  'encoded_dim':8}},
    #{'name':'8x8_dim4','ws':'','pams':{'shape':(8,8,1) ,'arrange': arrange8x8,'arrMask':arrMask,  'encoded_dim':4}},
    #{'name':'12x4_norm','ws':'','vis_shape':(12,4),'pams':{'shape':(12,4,1),
    #        'CNN_layer_nodes':[8,4,4,2],
    #        'CNN_kernel_size':[3,3,3,3],
    #        'CNN_pool':[False,False,False,False],     
    #}},

    #{'name':'4x4_norm','ws':'','pams':{'shape':(3,4,4) ,'channels_first':True }},
    #{'name':'4x4_norm_d10','ws':'','pams':{'shape':(3,4,4) ,'channels_first':True ,'encoded_dim':10}},
    #{'name':'4x4_norm_d8','ws':'','pams':{'shape':(3,4,4) ,'channels_first':True ,'encoded_dim':8}},
    #{'name':'4x4_norm_v4','ws':'','pams':{'shape':(3,4,4) ,'channels_first':True, 
    #     'CNN_layer_nodes':[4,4,4],
    #     'CNN_kernel_size':[3,3,3],
    #     'CNN_pool':[False,False,False],
    #}},
    #{'name':'4x4_norm_v5','ws':'','pams':{'shape':(3,4,4) ,'channels_first':True ,
    #     'CNN_layer_nodes':[8,4,2],
    #     'CNN_kernel_size':[3,3,3],
    #     'CNN_pool':[False,False,False],
    #}},
    #{'name':'4x4_norm_v6','ws':'','pams':{'shape':(3,4,4) ,'channels_first':True ,
    #     'CNN_layer_nodes':[8,4,2],
    #     'CNN_kernel_size':[5,5,3],
    #     'CNN_pool':[False,False,False],
    #}},
    #{'name':'4x4_norm_v7','ws':'','pams':{'shape':(3,4,4) ,'channels_first':True, 
    #     'CNN_layer_nodes':[4,4,4],
    #     'CNN_kernel_size':[5,5,3],
    #     'CNN_pool':[False,False,False],
    #}},
    {'name':'4x4_norm_v8','ws':'','pams':{'shape':(3,4,4) ,'channels_first':True, 
         'CNN_layer_nodes':[8,4,4,4,2],
         'CNN_kernel_size':[3,3,3,3,3],
         'CNN_pool':[0,0,0,0,0],
    }},

    #{'name':'4x4_v1',  'ws':'','vis_shape':(4,12),'pams':{'shape':(3,4,4) ,'channels_first':True,
    #     'CNN_layer_nodes':[8,8],
    #     'CNN_kernel_size':[3,3],
    #     'CNN_pool':[False,False], 
    #}},
    #{'name':'4x4_v2',  'ws':'','vis_shape':(4,12),'pams':{'shape':(3,4,4) ,'channels_first':True,
    #     'CNN_layer_nodes':[8,8],
    #     'CNN_kernel_size':[3,3],
    #     'CNN_pool':[False,False], 
    #     'Dense_layer_nodes':[16],
    #}},
    #{'name':'4x4_v3' ,'ws':'4x4_v3.hdf5','pams':{'shape':(3,4,4) ,'channels_first':True ,'CNN_kernel_size':[2]}},
  ]

  summary = pd.DataFrame(columns=['name','en_pams','tot_pams','corr','ssd'])
  #os.chdir('./CNN/')
  #os.chdir('./12x4/')
  os.chdir(options.odir)
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
    if model['ws']=='' :
      history = train(m_autoCNN,m_autoCNNen,train_input,val_input,name=model_name,n_epochs = options.epochs)

    Nevents = 8 

    input_Q,cnn_deQ ,cnn_enQ   = m.predict(val_input)
    index = np.random.choice(input_Q.shape[0], Nevents, replace=False)  
    corr_arr, ssd_arr  = visMetric(input_Q,cnn_deQ,maxdata,name=model_name)

    visualize(input_Q,cnn_deQ,cnn_enQ,index,name=model_name)
    if len(np.where(corr_arr>0.9)[0])>0:
        index = np.random.choice(np.where(corr_arr>0.9)[0], Nevents, replace=False)  
        visualize(input_Q,cnn_deQ,cnn_enQ,index,name=model_name+"_corr0.9")
    
    if len(np.where(corr_arr>0.2)[0])>0:
        index = np.random.choice(np.where(corr_arr<0.2)[0], Nevents, replace=False)  
        visualize(input_Q,cnn_deQ,cnn_enQ,index,name=model_name+"_corr0.2")

    model['corr'] = np.round(np.mean(corr_arr),3)
    model['ssd'] = np.round(np.mean(ssd_arr),3)

    summary = summary.append({'name':model_name,
                              'corr':model['corr'],
                              'ssd':model['ssd'],
                              'en_pams' : m_autoCNNen.count_params(),
                              'tot_pams': m_autoCNN.count_params(),
                              'ssd':model['ssd'],
                              },ignore_index=True)

    #print('CNN ssd: ' ,np.round(SSD(input_Q,cnn_deQ),3))
    with open(model_name+"_pams.json",'w') as f:
        f.write(json.dumps(m.get_pams(),indent=4))

    os.chdir('../')
  print(summary)


if __name__== "__main__":

    parser = optparse.OptionParser()
    parser.add_option('-o',"--odir", type="string", default = 'CNN/',dest="odir", help="input TSG ntuple")
    parser.add_option('-i',"--inputFile", type="string", default = 'CALQ_output_10x.csv',dest="inputFile", help="input TSG ntuple")
    parser.add_option("--dryRun", action='store_true', default = False,dest="dryRun", help="dryRun")
    parser.add_option("--epochs", type='int', default = 100, dest="epochs", help="n epoch to train")

    (options, args) = parser.parse_args()
    trainCNN(options,args)


