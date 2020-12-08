import numpy as np
import tensorflow as tf
import pandas as pd
import optparse
from tensorflow.python.client import device_lib
from tensorflow.keras import callbacks
from tensorflow import keras as kr
from tensorflow.keras import losses
import os
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import scipy.stats
from qkeras import get_quantizer,QActivation
from qkeras.utils import model_save_quantized_weights

##from utils import plotHist

import numba
import json
import pickle

#from models import *
from qDenseCNN import qDenseCNN
from denseCNN import denseCNN
from dense2DkernelCNN import dense2DkernelCNN

#for earth movers distance calculation
import ot
import graphUtil
import plotWafer
from get_flops import get_flops_from_model


def double_data(data):
    doubled=[]
    i=0
    while i<= len(data)-2:
        doubled.append( data[i] + data[i+1] )
        i+=2
    return np.array(doubled)

@numba.jit
def normalize(data,rescaleInputToMax=False, sumlog2=True):
    maxes =[]
    sums =[]
    sums_log2=[]
    for i in range(len(data)):
        maxes.append( data[i].max() )
        sums.append( data[i].sum() )
        sums_log2.append( 2**(np.floor(np.log2(data[i].sum()))) )
        if sumlog2:
            data[i] = 1.*data[i]/(sums_log2[-1] if sums_log2[-1] else 1.)
        efif rescaleInputToMax:
            data[i] = 1.*data[i]/(data[i].max() if data[i].max() else 1.)
        else:
            data[i] = 1.*data[i]/(data[i].sum() if data[i].sum() else 1.)
    if sumlog2: data,np.array(maxes),np.array(sums_log2)
    return data,np.array(maxes),np.array(sums)

@numba.jit
def unnormalize(norm_data,maxvals,rescaleOutputToMax=False, sumlog2=True):
    for i in range(len(norm_data)):
        if rescaleOutputToMax:
            norm_data[i] =  norm_data[i] * maxvals[i] / (norm_data[i].max() if norm_data[i].max() else 1.)
        else:
            if sumlog2:
                sumlog2 = 2**(np.floor(np.log2(data[i].sum())))
                norm_data[i] =  norm_data[i] * maxvals[i] / (sumlog2 if sumlog2 else 1.)
            else:
                norm_data[i] =  norm_data[i] * maxvals[i] / (norm_data[i].sum() if norm_data[i].sum() else 1.)
    return norm_data

def StringToTextFile(fname,s):
    with open(fname,'w') as f:
        f.write(s)

def plotHist(vals,name,odir='.',xtitle="",ytitle="",nbins=40,lims=None,
             stats=True, logy=False, leg=None):
    plt.figure(figsize=(6,4))
    if leg:
        n, bins, patches = plt.hist(vals, nbins, range=lims, label=leg)
    else:
        n, bins, patches = plt.hist(vals, nbins, range=lims)
    # print('bins',bins)
    # print('n',n)
    ax = plt.gca()
    plt.text(0.1, 0.9, name,transform=ax.transAxes)
    if stats:
        mu = np.mean(vals)
        std = np.std(vals)
        plt.text(0.1, 0.8, r'$\mu=%.3f,\ \sigma=%.3f$'%(mu,std),transform=ax.transAxes)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle if ytitle else 'Entries')
    if logy: plt.yscale('log')
    pname = odir+"/"+name+".pdf"
    print("Saving "+pname)
    tname = pname.replace('.pdf','.txt')
    StringToTextFile(tname, "{}, {}, {}, \t {}, {}, \n".format(np.quantile(vals,0.5), np.quantile(vals,0.5-0.68/2), np.quantile(vals,0.5+0.68/2), np.mean(vals), np.std(vals)))
    plt.savefig(pname)
    plt.close()
    return

def getWeights(vals, n=None, a=None, b=None):
    if a==None: a=min(vals)
    if b==None: b=max(vals)
    if n==None: b=20
    contents, bins, patches = plt.hist(vals, n, range=(a,b))

    # print("weight histo", contents)
    # print("weight vals ", [1./c if c else 0. for c in contents])
    
    def _getBin(x,bins):
        if x < bins[0]: return 0
        if x >= bins[-1]: return len(bins)-2
        for i in range(len(bins)-1):
            if x>= bins[i] and x<bins[i+1]: return i
        print ('bin logic error',x,bins)
        return 0

    _bins = np.array([_getBin(x,bins)for x in vals])
    return np.array([1./contents[b] for b in _bins]) # must be filled by construction///
    #return np.array([1./contents[b] if contents[b] else 1.0 for b in _bins])



def plotProfile(x,y,name,odir='.',xtitle="",ytitle="Entries",nbins=40,lims=None,
                stats=True, logy=False, leg=None, text=""):

    #median_result = scipy.stats.binned_statistic(x, y, bins=nbins, statistic='median')
    if lims==None: lims = (x.min(),x.max())
    median_result = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5))
    lo_result     = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5-0.68/2))
    hi_result     = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5+0.68/2))
    median = np.nan_to_num(median_result.statistic)
    hi = np.nan_to_num(hi_result.statistic)
    lo = np.nan_to_num(lo_result.statistic)
    hie = hi-median
    loe = median-lo
    bin_edges = median_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.

    # means_result = scipy.stats.binned_statistic(x, [y, y**2], bins=nbins, statistic='mean')
    # means, means2 = means_result.statistic
    # standard_deviations = np.sqrt(means2 - means**2)
    # bin_edges = means_result.bin_edges
    # bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    
    plt.figure(figsize=(6,4))
    plt.errorbar(x=bin_centers, y=median, yerr=[loe,hie], linestyle='none', marker='.', label=leg)

    printstr=""
    for i,b in enumerate(bin_centers):
        printstr += "{} {} {} {} \n".format(b, median[i], loe[i], hie[i])

    ax = plt.gca()
    plt.text(0.1, 0.9, name,transform=ax.transAxes)
    if text: plt.text(0.1, 0.82, text.replace('MAX','inf'), transform=ax.transAxes)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    if logy: plt.yscale('log')
    pname = odir+"/"+name+".pdf"
    print("Saving "+pname)
    tname = pname.replace('.pdf','.txt')
    StringToTextFile(tname, printstr)
    plt.savefig(pname)
    plt.close()
#    return bin_centers, means, standard_deviations
    return bin_centers, median, [loe,hie]

def OverlayPlots(results, name, xtitle="",ytitle="Entries",odir='.',text="",ylim=None):
    #print('overlay: ',name)
    centers = results[0][1][0]
    wid = centers[1]-centers[0]
    offset = 0.33*wid

    plt.figure(figsize=(6,4))

    for ir,r in enumerate(results):
        lab = r[0]
        dat = r[1]
        off = offset * (ir-1)/2 * (-1. if ir%2 else 1.) # .1 left, .1 right, .2 left, ...
        plt.errorbar(x=dat[0]+off, y=dat[1], yerr=dat[2], label=lab)
        #plt.errorbar(x=r[0], y=r[1], yerr=r[2], linestyle='none', marker='.', label=leg)

    ax = plt.gca()
    plt.text(0.1, 0.9, name, transform=ax.transAxes)
    if text: plt.text(0.1, 0.82, text.replace('MAX','inf'), transform=ax.transAxes)
    if ylim is not None:
        plt.ylim(ylim[0],ylim[1])
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(loc='upper right')
    #if logy: plt.yscale('log')
    pname = odir+"/"+name+".pdf"
    print("Saving "+pname)
    plt.savefig(pname)
    plt.close()

    return

def split(shaped_data, validation_frac=0.2,randomize=False):
    N = round(len(shaped_data)*validation_frac)
    
    if randomize:
        #randomly select 25% entries
        val_index = np.random.choice(shaped_data.shape[0], N, replace=False)
        #select the indices of the other 75%
        full_index = np.array(range(0,len(shaped_data)))
        train_index = np.logical_not(np.in1d(full_index,val_index))
      
        val_input = shaped_data[val_index]
        train_input = shaped_data[train_index]
    else:
        val_input = shaped_data[:N]
        train_input = shaped_data[N:]
        val_index = np.arange(N)
        train_index = np.arange(len(shaped_data))[N:]
    
    print('training shape',train_input.shape)
    print('validation shape',val_input.shape)

    return val_input,train_input,val_index,train_index

def train(autoencoder,encoder,train_input,train_target,val_input,name,n_epochs=100, train_weights=None):

    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    #reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3)
    if train_weights != None:
        history = autoencoder.fit(train_input,train_target,
                                  #sample_weight=train_weights,
                                  epochs=n_epochs,
                                  batch_size=500,
                                  shuffle=True,
                                  validation_data=(val_input,val_input),
                                  callbacks=[es]
        )
    else:
        history = autoencoder.fit(train_input,train_target,
                                  epochs=n_epochs,
                                  batch_size=500,
                                  shuffle=True,
                                  validation_data=(val_input,val_input),
                                  callbacks=[es]
        )

    plt.figure(figsize=(8,6))
    plt.yscale('log')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss %s'%name)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig("history_%s.pdf"%name)
    plt.close()
    plt.clf()

    with open('./history_%s.pkl'%name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    isQK = False
    for layer in autoencoder.layers[1].layers:
        if QActivation == type(layer): isQK = True
    save_models(autoencoder,name,isQK)

    return history

def save_models(autoencoder, name, isQK=False):
    json_string = autoencoder.to_json()
    encoder = autoencoder.get_layer("encoder")
    decoder = autoencoder.get_layer("decoder")
    with open('./%s.json'%name,'w') as f:        f.write(autoencoder.to_json())
    with open('./%s.json'%("encoder_"+name),'w') as f:            f.write(encoder.to_json())
    with open('./%s.json'%("decoder_"+name),'w') as f:            f.write(decoder.to_json())
    autoencoder.save_weights('%s.hdf5'%name)
    encoder.save_weights('%s.hdf5'%("encoder_"+name))
    decoder.save_weights('%s.hdf5'%("decoder_"+name))
    if isQK:
       encoder_qWeight = model_save_quantized_weights(encoder) 
       with open('encoder_'+name+'.pkl','wb') as f:       pickle.dump(encoder_qWeight,f)
       encoder = graphUtil.setQuanitzedWeights(encoder,'encoder_'+name+'.pkl')
       print('unique weights',len(np.unique(encoder.layers[5].get_weights()[0])))
    graphUtil.outputFrozenGraph(encoder,'encoder_'+name+'.pb')
    graphUtil.outputFrozenGraph(encoder,'encoder_'+name+'.pb.ascii','./',True)
    graphUtil.outputFrozenGraph(decoder,'decoder_'+name+'.pb')
    graphUtil.outputFrozenGraph(decoder,'decoder_'+name+'.pb.ascii','./',True)
    graphUtil.plotWeights(autoencoder)
    graphUtil.plotWeights(encoder)
    graphUtil.plotWeights(decoder)

    return

### cross correlation of input/output 
def cross_corr(x,y):
    if (np.sum(x)==0): return -1.
    if (np.sum(y)==0): return -0.5
    cov = np.cov(x.flatten(),y.flatten())
    std = np.sqrt(np.diag(cov))
    stdsqr = np.multiply.outer(std, std)
    corr = np.divide(cov, stdsqr, out=np.zeros_like(cov), where=(stdsqr!=0))
    return corr[0,1]

def ssd(x,y):
    if (np.sum(x)==0): return -1.
    if (np.sum(y)==0): return -0.5
    if (np.sum(x)==0 or np.sum(y)==0): return 1.
    ssd=np.sum(((x-y)**2).flatten())
    ssd = ssd/(np.sum(x**2)*np.sum(y**2))**0.5
    return ssd

# calculate "earth mover's distance"
# (cost, in distance, to move earth from one config to another)
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

#normalize so that distance between small cells (there are 4 per TC) is 1
oneHexCell = 0.5 * 2.4168015
#oneHexCell = 0.5 * np.min(ot.dist(hexCoords[:16],hexCoords[:16],'euclidean'))
hexCoords = hexCoords / oneHexCell
# for later normalization
HexSigmaX = np.std(hexCoords[:,0]) 
HexSigmaY = np.std(hexCoords[:,1])
# pairwise distances
hexMetric = ot.dist(hexCoords, hexCoords, 'euclidean')
MAXDIST = np.max(hexMetric)
def emd(_x, _y, threshold=-1):
    if (np.sum(_x)==0): return -1.
    if (np.sum(_y)==0): return -0.5
    x = np.array(_x, dtype=np.float64)
    y = np.array(_y, dtype=np.float64)
    x = (1./x.sum() if x.sum() else 1.)*x.flatten()
    y = (1./y.sum() if y.sum() else 1.)*y.flatten()

    if threshold > 0:
        # only keep entries above 2%, e.g.
        x = np.where(x>threshold,x,0)
        y = np.where(y>threshold,y,0)
        x = 1.*x/x.sum()
        y = 1.*y/y.sum()

    return ot.emd2(x, y, hexMetric)

def d_weighted_mean(x, y):
    if (np.sum(x)==0): return -1.
    if (np.sum(y)==0): return -0.5
    x = (1./x.sum() if x.sum() else 1.)*x.flatten()
    y = (1./y.sum() if y.sum() else 1.)*y.flatten()
    dx = hexCoords[:,0].dot(x-y)
    dy = hexCoords[:,1].dot(x-y)
    return np.sqrt(dx*dx+dy*dy)

def get_rms(coords, weights):
    mu_x = coords[:,0].dot(weights)
    mu_y = coords[:,1].dot(weights)
    sig2 = np.power((coords[:,0]-mu_x)/HexSigmaX, 2) \
         + np.power((coords[:,1]-mu_y)/HexSigmaY, 2)
    w2 = np.power(weights,2)
    return np.sqrt(sig2.dot(w2))
    
def d_weighted_rms(a, b):
    if (np.sum(a)==0): return -1.
    if (np.sum(b)==0): return -0.5
    # weights
    a = (1./a.sum() if a.sum() else 1.)*a.flatten()
    b = (1./b.sum() if b.sum() else 1.)*b.flatten()
    return get_rms(hexCoords,a) - get_rms(hexCoords,b)
def d_abs_weighted_rms(a, b):
    if (np.sum(a)==0): return -1.
    if (np.sum(b)==0): return -0.5
    return np.abs(d_weighted_rms(a, b))

STC4mask = np.array([
    [ 0,  1,  4,  5], #indices for 1 super trigger cell
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
STC16mask = np.array(range(16))
STC16mask = np.array([STC16mask,STC16mask+16,STC16mask+32])

def make_supercells(inQ, shareQ=False, stc16=True):
    outQ = inQ.copy()
    inshape = inQ[0].shape
    for i in range(len(inQ)):
        inFlat = inQ[i].flatten()
        outFlat = outQ[i].flatten()
        for sc in (STC16mask if stc16 else STC4mask):
            # set max cell to sum
            if shareQ:
                mysum = np.sum( inFlat[sc] )
                outFlat[sc]=mysum/4.
            else:
                ii = np.argmax( inFlat[sc] )
                mysum = np.sum( inFlat[sc] )
                outFlat[sc]=0
                outFlat[sc[ii]]=mysum
        outQ[i] = outFlat.reshape(inshape)
    return outQ

def best_choice(inQ, n):
    outQ = inQ.copy()
    inshape = inQ[0].shape
    for i in range(len(inQ)):
        inFlat = inQ[i].flatten()
        outFlat = outQ[i].flatten()
        outFlat[np.argsort(outFlat)[:-n]]=0
        # get indices of all but n largest Q, set elements to zero
        outQ[i] = outFlat.reshape(inshape)
    return outQ

# unused
# def threshold(_x, cut):
#     x = _x.copy()
#     # # reshape to allow broadcasting to all cells
#     # norm_shape = norm.reshape((norm.shape[0],)+(1,)*(x.ndim-1))
#     # x = np.where(x*norm_shape>=cut,x,0)
#     x = np.where(x>=cut,x,0)
#     return x

def invertArrange(arrange):
    remap =[]
    hashmap = {}
    for i in range(len(arrange)):
        hashmap[arrange[i]]=i
    for i in range(len(arrange)):        
        remap.append(hashmap[i])
    return remap 


def visDisplays(index,input_Q,input_calQ,decoded_Q,encoded_Q=np.array([]),conv2d=None,name='model_X'):
    Nevents = len(index)
        
    inputImg    = input_Q[index]
    inputImgCalQ= input_calQ[index]
    outputImg   = decoded_Q[index]

  
    fig, axs = plt.subplots(3, Nevents, figsize=(16, 10))
    for i in range(Nevents):
        if i==0:
            axs[0,i].set(xlabel='',ylabel='cell_y',title='Input_%i'%i)
        else:
            axs[0,i].set(xlabel='',title='Input_%i'%i)       
        plotWafer.plotWafer( inputImgCalQ[i], fig, axs[0,i])
            
    for i in range(Nevents):
        if i==0:
            axs[1,i].set(xlabel='cell_x',ylabel='cell_y',title='CNN Ouput_%i'%i)        
        else:
            axs[1,i].set(xlabel='cell_x',title='CNN Ouput_%i'%i)
        plotWafer.plotWafer( outputImg[i], fig, axs[1,i])

    if len(encoded_Q):
        encodedImg  = encoded_Q[index]
        for i in range(0,Nevents):
            if i==0:
                axs[2,i].set(xlabel='latent dim',ylabel='depth',title='Encoded_%i'%i)
            else:
                axs[2,i].set(xlabel='latent dim',title='Encoded_%i'%i)
            c1=axs[2,i].imshow(encodedImg[i])
            if i==Nevents:
                plt.colorbar(c1,ax=axs[2,i])
            
    #plt.tight_layout()
    plt.savefig("%s_examples.pdf"%name)
    plt.close()

    if conv2d is not None:
        actImg      = conv2d.predict(inputImg.reshape(Nevents,4,4,3))
        nFilters    = actImg.shape[-1]
        nrows = nFilters 
        fig, axs = plt.subplots(nrows, Nevents, figsize=(16, 20))

        for i in range(0,Nevents):
            for k in range(0,nFilters):
                if i==0:
                    axs[k,i].set(xlabel='cell_x',ylabel='cell_y',title='Activation_%i'%k)
                else:
                    axs[k,i].set(xlabel='cell_x',title='Activation_%i'%k)
                c1=axs[k,i].imshow(actImg[i,:,:,k])
                plt.colorbar(c1,ax=axs[k,i])

        plt.savefig("%s_activations.pdf"%name)
        plt.close()
  

def visMetric(input_Q,decoded_Q,metric,name,odir,skipPlot=False):

    plotHist(vals,name,options.odir,xtitle=longMetric[mname])
    plt.figure(figsize=(6,4))
    plt.hist([input_Q.flatten(),decoded_Q.flatten()],20,label=['input','output'])
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.xlabel('Charge fraction')
    plt.savefig("hist_Qfr_%s.pdf"%name)
    plt.close()

    input_Q_abs   = np.array([input_Q[i] * maxQ[i] for i in range(0,len(input_Q))])
    decoded_Q_abs = np.array([decoded_Q[i]*maxQ[i] for i in range(0,len(decoded_Q))])

    nonzeroQs = np.count_nonzero(input_Q_abs.reshape(len(input_Q_abs),48),axis=1)
    occbins = [0,5,10,20,48]
    fig, axes = plt.subplots(1,len(occbins)-1, figsize=(16, 4))
    for i,ax in enumerate(axes):
        #print(cross_corr_arr[selection])
        selection=np.logical_and(nonzeroQs<occbins[i+1],nonzeroQs>occbins[i])
        label = '%i<occ<%i'%(occbins[i],occbins[i+1])
        mu = np.mean(cross_corr_arr[selection])
        std = np.std(cross_corr_arr[selection])
        plt.text(0.1, 0.8, r'$\mu=%.3f,\ \sigma=%.3f$'%(mu,std),transform=ax.transAxes)
        ax.hist(cross_corr_arr[selection],40)
        ax.set(xlabel='corr',title=label)
        #plt.tight_layout()
        #plt.show()
        plt.savefig('corr_vs_occ_%s.pdf'%name)
        plt.close()

    return cross_corr_arr,ssd_arr,emd_arr

def GetBitsString(In, Accum, Weight, Encoded, Dense=False, Conv=False):
    s=""
    s += "Input{}b{}i".format(In['total'],In['integer'])
    s += "_Accum{}b{}i".format(Accum['total'],Accum['integer'])
    if Dense:
        s += "_Dense{}b{}i".format(Dense['total'], Dense['integer'])
        if Conv:
            s += "_Conv{}b{}i".format(Conv['total'], Conv['integer'])
        else:
            s += "_Conv{}b{}i".format(Weight['total'], Weight['integer'])
    elif Conv:
        s += "_Dense{}b{}i".format(Weight['total'], Weight['integer'])
        s += "_Conv{}b{}i".format(Conv['total'], Conv['integer'])
    else:
        s += "_Weight{}b{}i".format(Weight['total'],Weight['integer'])
    s += "_Encod{}b{}i".format(Encoded['total'], Encoded['integer'])
    return s

def sumTCQ(x): return x.reshape(len(x),48).sum(axis=1)

def buildmodels(options,pam_updates):
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

    # Fix # weights in bins (present as scan over output nodes)
    # 4b/10b output: 128*16 * 6b = 12 288
    # 5b/12b output: 128*12 * 8b = 12 288
    # 6b/15b output: 128*10 * 10b = 12 800
    # 8b/20b output: 128* 8 * 12b = 12 288
    # 10b/25b output:128* 6 * 16b = 12 288
    # (output precision) 2/5 links --- 128*out * (weight precision)


    # 4b/10b output: 128*16 * 6b = 12 288
    
    if(options.nElinks==2):
        nBits_encod  = {'total':  3, 'integer': 1,'keep_negative':0}
        nBits_encod_total = 3 
    elif(options.nElinks==3):
        nBits_encod  = {'total':  5, 'integer': 1,'keep_negative':0}
        nBits_encod_total = 5
    elif(options.nElinks==4):
        nBits_encod  = {'total':  7, 'integer': 1,'keep_negative':0}
        nBits_encod_total = 7
    elif(options.nElinks==5):
        nBits_encod  = {'total':  9, 'integer': 1,'keep_negative':0}
        nBits_encod_total = 9
    else:
        print("must specify encoding bits for nElink =",options.nElinks)

    ##default
    nBits_input  = {'total': 10, 'integer': 3, 'keep_negative':1}
    nBits_accum  = {'total': 11, 'integer': 3, 'keep_negative':1}
    nBits_weight = {'total':  5, 'integer': 1, 'keep_negative':1} # sign bit not included
    edim = 16

    if options.quantize:
        mymodname = "Jul24_qKeras"
        #mymodname = "Jul24_qKeras_{}out{}b{}_{}b{}weights".format(edim,
        #                                                         nBits_encod['total'], nBits_encod['integer'],
        #                                                         nBits_weight['total'], nBits_weight['integer'])
    else:
        mymodname = "Jul24_keras"

    from martinModels import models
    for m in models:
        m['pams'].update({'nBits_encod':nBits_encod})
    #models = [

        #{'name': "Aug11_qKeras_smoothSigmoid", 'ws': '', # custom
        #'pams': {'shape': (4, 4, 3),
        #         'channels_first': False,
        #         'arrange': arrange443,
        #         'encoded_dim': edim,
        #         'loss': 'telescopeMSE',
        #         'activation': 'smooth_sigmoid',
        #     },
        # 'isQK':True,
        #},
        #{'name': "Aug11_Keras" , 'ws': '', # custom
        #'pams': {'shape': (4, 4, 3),
        #         'channels_first': False,
        #         'arrange': arrange443,
        #         'encoded_dim': edim,
        #         'loss': 'telescopeMSE',
        #         'activation': 'sigmoid',
        #     },
        # 'isQK':False,
        #},
        #{'name': "Aug14_qKeras_optA", 'ws': '', # custom
        #'pams': {'shape': (4, 4, 3),
        #         'channels_first': False,
        #         'arrange': arrange443,
        #         'encoded_dim': edim,
        #         'loss': 'telescopeMSE',
        #         'nBits_encod'  : {'total':  nBits_encod_total, 'integer': 1,'keep_negative':0},
        #         'nBits_input'  : {'total': 10,                 'integer': 3,'keep_negative':1},
        #         'nBits_accum'  : {'total': 11,                 'integer': 3,'keep_negative':1},
        #         'nBits_weight' : {'total':  5,                 'integer': 1,'keep_negative':1},
        #     },
        # 'isQK':True,
        #},
        #{'name': "Aug14_qKeras_optC", 'ws': '', # custom
        #'pams': {'shape': (4, 4, 3),
        #         'channels_first': False,
        #         'arrange': arrange443,
        #         'encoded_dim': edim,
        #         'loss': 'telescopeMSE',
        #         'activation': 'smooth_sigmoid',
        #         'nBits_encod'  : {'total':  nBits_encod_total, 'integer': 0,'keep_negative':0},
        #         'nBits_input'  : {'total': 10,                 'integer': 3,'keep_negative':1},
        #         'nBits_accum'  : {'total': 11,                 'integer': 3,'keep_negative':1},
        #         'nBits_weight' : {'total':  5,                 'integer': 1,'keep_negative':1},
        #     },
        # 'isQK':True,
        #},
        #{'name': "Aug13_qKeras_optD", 'ws': '', # custom
        #'pams': {'shape': (4, 4, 3),
        #         'channels_first': False,
        #         'arrange': arrange443,
        #         'encoded_dim': edim,
        #         'loss': 'telescopeMSE',
        #         'activation': 'sigmoid',
        #     },
        # 'isQK':False,
        #},

    #]    
    for m in models:
        if m['isQK']:
            #m['pams'].update({
            #    'nBits_weight':nBits_weight,
            #    'nBits_input':nBits_input,
            #    'nBits_accum':nBits_accum,
            #    'nBits_encod': nBits_encod,
            #})
            print('nBits_weight:', m['pams']['nBits_weight'])
            print( 'nBits_input:', m['pams']['nBits_input' ])
            print( 'nBits_accum:', m['pams']['nBits_accum' ])
            print( 'nBits_encod:', m['pams']['nBits_encod' ])
            bit_str = GetBitsString(m['pams']['nBits_input'],  m['pams']['nBits_accum'],
                                    m['pams']['nBits_weight'], m['pams']['nBits_encod'],
                                   (m['pams']['nBits_dense'] if 'nBits_dense'  in m['pams'] else False),
                                   (m['pams']['nBits_conv'] if 'nBits_conv' in m['pams'] else False))
            #mymodname += "_" + bit_str
   
        mymodname = m['name'] 
        if os.path.exists(options.odir+mymodname+"/"+mymodname+".hdf5"):
            if options.retrain:
                print('Found weights, but going to re-train as told.')            
                m['ws'] = ""
            else:
                print('Found weights, using it by default')
                m['ws'] = mymodname+".hdf5"
        else:
            print('Have not found trained weights in dir:',options.odir+mymodname+"/"+mymodname+".hdf5")


        if pam_updates:
            m['pams'].update(pam_updates)
            print ('updated parameters for model',m['name'])
        if options.loss:
            m['pams']['loss'] = options.loss
        print(m)
    return models

def compareModels(models,perf_dict,eval_settings,options):

    algnames = eval_settings['algnames']
    metrics  = eval_settings['metrics']
    occ_nbins    =eval_settings[  "occ_nbins"  ] 
    occ_range    =eval_settings[  "occ_range"   ]
    occ_bins     =eval_settings[  "occ_bins"    ]
    chg_nbins    =eval_settings[  "chg_nbins"   ]
    chg_range    =eval_settings[  "chg_range"   ]
    chglog_nbins =eval_settings[  "chglog_nbins"]
    chglog_range =eval_settings[  "chglog_range"]
    chg_bins     =eval_settings[  "chg_bins"    ]
    occTitle    =eval_settings["occTitle"   ]
    logMaxTitle =eval_settings["logMaxTitle"]
    logTotTitle =eval_settings["logTotTitle"]


    summary_entries=['name','en_pams','tot_pams','en_flops']
    for algname in algnames:
        for mname in metrics:
            name = mname+"_"+algname
            summary_entries.append(mname+"_"+algname)
            summary_entries.append(mname+"_"+algname+"_err")
    summary = pd.DataFrame(columns=summary_entries)

    with open('./performance.pkl', 'wb') as file_pi:
        pickle.dump(perf_dict, file_pi)

    if(not options.skipPlot):
        # overlay different metrics
        for mname in metrics:
            chgs=[]
            occs=[]
            for model_name in perf_dict:
                plots = perf_dict[model_name]
                # name = mname+"_ae"
                short_model = model_name
                #short_model = model_name.split('_')[-1]
                chgs += [(short_model, plots["chg_"+mname+"_ae"])]
                occs += [(short_model, plots["occ_"+mname+"_ae"])]
            xt = logMaxTitle if options.rescaleInputToMax else logTotTitle
            OverlayPlots(chgs,"ae_comp_chg_"+mname,xtitle=xt,ytitle=mname)
            OverlayPlots(occs,"ae_comp_occ_"+mname,xtitle=occTitle,ytitle=mname)
            
            # binned profiles 
            #for iocc, occ_lo in enumerate(occ_bins):
            #    occ_hi = 9e99 if iocc+1==len(occ_bins) else occ_bins[iocc+1]
            #    occ_hi_s = 'MAX' if iocc+1==len(occ_bins) else str(occ_hi)
            #    pname = "{}occ{}".format(occ_lo,occ_hi_s)
            #    chgs=[ (model_name.split('_')[-1], perf_dict[model_name]["chg_{}_{}_ae".format(pname,mname)]) for model_name in perf_dict]
            #    xt = logMaxTitle if options.rescaleInputToMax else logTotTitle
            #    OverlayPlots(chgs,"ae_comp_chg_{}_{}".format(mname,pname),xtitle=xt,ytitle=mname)
            #for ichg, chg_lo in enumerate(chg_bins):
            #    chg_hi = 9e99 if ichg+1==len(chg_bins) else chg_bins[ichg+1]
            #    chg_hi_s = 'MAX' if ichg+1==len(chg_bins) else str(chg_hi)
            #    pname = "{}chg{}".format(chg_lo,chg_hi_s)
            #    occs=[ (model_name.split('_')[-1], perf_dict[model_name]["occ_{}_{}_ae".format(pname,mname)]) for model_name in perf_dict]
            #    OverlayPlots(occs,"ae_comp_occ_{}_{}".format(mname,pname),xtitle=occTitle,ytitle=mname)
    for model in models:
        print('Summary_dict',model['summary_dict'])
        summary = summary.append(model['summary_dict'], ignore_index=True)
    print(summary)


def evalModel(model,charges,aux_arrs,eval_settings,options):

    ### input arrays
    input_Q      = charges['input_Q']       #input Q image ,     (Nevent,12,4)
    input_Q_abs  = charges['input_Q_abs']
    input_calQ   = charges['input_calQ']
    output_calQ  = charges['output_calQ']
    output_calQ_fr  = charges['output_calQ_fr']
    cnn_deQ      = charges['cnn_deQ']
    cnn_enQ      = charges['cnn_enQ']
    val_sum      = charges['val_sum']
    val_max      = charges['val_max']

    #print('input_Q'    ,input_Q[1]*35)
    #print('input_Q_abs',input_Q_abs[1]*35)
    #print('input_calQ' ,input_calQ[1]*35)
    #print('val_sum' ,val_sum[1]*35)
    ae_out      = output_calQ 
    ae_out_frac = normalize(output_calQ.copy())

    #ae_out = unnormalize(cnn_deQ.copy(), val_max if options.rescaleOutputToMax else val_sum, rescaleOutputToMax=options.rescaleOutputToMax)
    #ae_out_frac = normalize(cnn_deQ.copy())
    ### axilliary arrays with shapes 
    occupancy_1MT = aux_arrs['occupancy_1MT']

    # visualize conv2d activations
    if not model['isQK']:
        #conv2d = kr.models.Model(
        #    inputs =model['m_autoCNNen'].inputs,
        #    outputs=model['m_autoCNNen'].get_layer("conv2d").output
        #)
        conv2d  = None
    else:
        conv2d = kr.models.Model(
            inputs =model['m_autoCNNen'].inputs,
            outputs=model['m_autoCNNen'].get_layer("conv2d_0_m").output
        )


    algnames    =eval_settings[  "algnames"  ] 
    metrics     =eval_settings[  "metrics"  ] 
    occ_nbins    =eval_settings[  "occ_nbins"  ] 
    occ_range    =eval_settings[  "occ_range"   ]
    occ_bins     =eval_settings[  "occ_bins"    ]
    chg_nbins    =eval_settings[  "chg_nbins"   ]
    chg_range    =eval_settings[  "chg_range"   ]
    chglog_nbins =eval_settings[  "chglog_nbins"]
    chglog_range =eval_settings[  "chglog_range"]
    chg_bins     =eval_settings[  "chg_bins"    ]
    occTitle    =eval_settings["occTitle"   ]
    logMaxTitle =eval_settings["logMaxTitle"]
    logTotTitle =eval_settings["logTotTitle"]


    longMetric = {'cross_corr':'cross correlation',
                  'SSD':'sum of squared differences',
                  'EMD':'earth movers distance',
                  'dMean':'difference in energy-weighted mean',
                  'dRMS':'difference in energy-weighted RMS',
                  'zero_frac':'zero fraction',}

    print("Running non-AE algorithms")
    if options.AEonly:
        alg_outs = {'ae' : ae_out}
    else:
        thr_lo_Q = np.where(input_Q_abs>1.35,input_Q_abs,0) # 1.35 transverse MIPs
        stc_Q = make_supercells(input_Q_abs, stc16=(options.nElinks!=5))
        nBC={2:4, 3:6, 4:9, 5:14} #4, 6, 9, 14 (for 2,3,4,5 e-links)
        bc_Q = best_choice(input_Q_abs, nBC[options.nElinks]) 
        alg_outs = {
            'ae' : ae_out,
            'stc': stc_Q,
            'bc': bc_Q,
            'thr_lo': thr_lo_Q,
        }

    if False and options.full:
        thr_hi_Q = np.where(input_Q_abs>2.0,input_Q_abs,0) # 2.0  transverse MIPs
        alg_outs['thr_hi']=thr_hi_Q 
        ae_thr_lo = np.where(ae_out>1.35,ae_out,0) # 1.35  transverse MIPs
        bc_thr_lo = np.where(bc_Q>1.35,bc_Q,0) # 1.35  transverse MIPs
        alg_outs['ae_thr_lo']=ae_thr_lo
        alg_outs['bc_thr_lo']=bc_thr_lo

    # to generate event displays
    Nevents = 8
    index = np.random.choice(input_Q.shape[0], Nevents, replace=False)

    model_name = model['name']
    plots={}
    summary_dict = {
        'name':model_name,
        'en_pams' : model['m_autoCNNen'].count_params(),
        'en_flops' : get_flops_from_model(model['m_autoCNNen']),
        'tot_pams': model['m_autoCNN'].count_params(),
    }
    if (not options.skipPlot): plotHist(np.log10(val_sum.flatten()),
                                        "sumQ_validation",xtitle=logTotTitle,ytitle="Entries",
                                        stats=True,logy=True,nbins=chglog_nbins,lims = chglog_range)
    if (not options.skipPlot): plotHist([np.log10(val_max.flatten())],
                                        "maxQ_validation",xtitle=logMaxTitle,ytitle="Entries",
                                        stats=True,logy=True,nbins=chglog_nbins,lims = chglog_range)

    # compute metrics for each alg
    for algname, alg_out in alg_outs.items():
        print('Calculating metrics for '+algname)
        ## charge fraction comparison
        #if (not options.skipPlot): plotHist([input_Q.flatten(),alg_out.flatten()],
        #                                    algname+"_fracQ",xtitle="charge fraction",ytitle="Cells",
        #                                    stats=False,logy=True,leg=['input','output'])
        ## abs charge comparison
        #if(not options.skipPlot): plotHist([input_Q_abs.flatten(),alg_out.flatten()],
        #                                   algname+"_absQ",xtitle="absolute charge",ytitle="Cells",
        #                                   stats=False,logy=True,leg=['input','output'])
        ## abs tower charge comparison (xcheck)
        #if(not options.skipPlot): plotHist([sumTCQ(input_Q_abs),sumTCQ(alg_out)],
        #                                   algname+"_absSumTCQ",xtitle="absolute charge",ytitle="48 TC arrays",
        #                                   stats=False,logy=True,leg=['input','output'])

        ## Encoded space distibution 
        #if((not options.skipPlot) and algname=='ae'): plotHist([cnn_enQ.flatten()],
        #                                   "hist_"+algname+"_encoded",xtitle="AE encoded vector",ytitle="Entries",
        #                                   stats=True,logy=True)

        # event displays
        if(not options.skipPlot): visDisplays(index, input_Q, input_calQ, alg_out, (cnn_enQ if algname=='ae' else np.array([])),(conv2d if algname=='ae' else None), name=algname)
        for mname, metric in metrics.items():
            print('  '+mname)
            name = mname+"_"+algname
            if (algname =='ae' and mname=='EMD'):
                #vals = np.array([metric(input_Q_abs[i],alg_out[i]) for i in range(0,len(input_Q_abs))])
                vals = np.array([metric(input_calQ[i],alg_out[i]) for i in range(0,len(input_Q_abs))])
                #low_index = (np.where(vals<np.quantile(vals,0.1)))[0]
                #print("np.quantile(vals,0.1) =",np.quantile(vals,0.1))
                #print("EMD: input calQ[1] =",np.round(input_calQ[low_index[0]],3))
                #print("EMD: output Q[1] =",np.round(alg_out[low_index[0]],3))
                #print("EMD: metric Q[1] =",metric(input_calQ[low_index[0]],alg_out[low_index[0]]))
            else:
                #vals = np.array([metric(input_Q_abs[i],alg_out[i]) for i in range(0,len(input_Q_abs))])
                vals = np.array([metric(input_calQ[i],alg_out[i]) for i in range(0,len(input_Q_abs))])
            model[name]        = np.round(np.mean(vals), 3)
            model[name+'_err'] = np.round(np.std(vals), 3)
            summary_dict[name]        = model[name]
            summary_dict[name+'_err'] = model[name+'_err']
            if(not options.skipPlot) and (not('zero_frac' in mname)):
                # metric distribution
                plotHist(vals,"hist_"+name,xtitle=longMetric[mname])
                plotHist(vals[vals>-1e-9],"hist_nonzero_"+name,xtitle=longMetric[mname])
                plotHist(np.where(vals>-1e-9,1,0),"hist_iszero_"+name,xtitle=longMetric[mname])
                # 1d profiles
                plots["occ_"+name] = plotProfile(occupancy_1MT, vals,"profile_occ_"+name,
                                                 nbins=occ_nbins, lims=occ_range,
                                                 xtitle=occTitle,ytitle=longMetric[mname])
                plots["chg_"+name] = plotProfile(np.log10(val_max), vals,"profile_maxQ_"+name,ytitle=longMetric[mname],
                                                 nbins=chglog_nbins, lims=chglog_range,
                                                 xtitle=logMaxTitle if options.rescaleInputToMax else logTotTitle)
                #plotHist(vals[val_max<1],"hist_0chg1_"+name,xtitle=longMetric[mname])
                #plotHist(vals[val_max<2],"hist_0chg2_"+name,xtitle=longMetric[mname])
                #plotHist(vals[val_max<5],"hist_0chg5_"+name,xtitle=longMetric[mname])
                #plotHist(vals[val_max<10],"hist_0chg10_"+name,xtitle=longMetric[mname])
                #plotHist(vals[(val_max>=100)],"hist_chg10+_"+name,xtitle=longMetric[mname])
                # plotHist(vals[occupancy_1MT<10],"hist_0occ10_"+name,xtitle=longMetric[mname])
                # plotHist(vals[occupancy_1MT>10],"hist_10occMAX_"+name,xtitle=longMetric[mname])
                # plotHist(vals[occupancy_1MT>15],"hist_15occMAX_"+name,xtitle=longMetric[mname])
                # plotHist(vals[occupancy_1MT>20],"hist_20occMAX_"+name,xtitle=longMetric[mname])
                # plotHist(vals[occupancy_1MT>30],"hist_30occMAX_"+name,xtitle=longMetric[mname])
                # plotHist(vals[occupancy_1MT>40],"hist_40occMAX_"+name,xtitle=longMetric[mname])
                # binned profiles 
                for iocc, occ_lo in enumerate(occ_bins):
                    occ_hi = 9e99 if iocc+1==len(occ_bins) else occ_bins[iocc+1]
                    occ_hi_s = 'MAX' if iocc+1==len(occ_bins) else str(occ_hi)
                    indices = (occupancy_1MT >= occ_lo) & (occupancy_1MT < occ_hi)
                    pname = "chg_{}occ{}_{}".format(occ_lo,occ_hi_s,name)
                    plots[pname] = plotProfile(np.log10(val_max[indices]), vals[indices],"profile_"+pname,
                                               xtitle=logMaxTitle,
                                               nbins=chglog_nbins, lims=chglog_range,
                                               ytitle=longMetric[mname],
                                               text="{} <= occupancy < {}".format(occ_lo,occ_hi_s,name))
                    #plotHist(vals[indices].flatten(),"hist_{}_{}occ{}".format(mname,occ_lo,occ_hi_s),
                    #                           xtitle=longMetric[mname],
                    #                           nbins=chglog_nbins, lims=None,
                    #                           )

                    print('filling1', model_name, pname)
                for ichg, chg_lo in enumerate(chg_bins):
                    chg_hi = 9e99 if ichg+1==len(chg_bins) else chg_bins[ichg+1]
                    chg_hi_s = 'MAX' if ichg+1==len(chg_bins) else str(chg_hi)
                    indices = (val_max >= chg_lo) & (val_max < chg_hi)
                    pname = "occ_{}chg{}_{}".format(chg_lo,chg_hi_s,name)
                    plots[pname] = plotProfile(occupancy_1MT[indices], vals[indices],"profile_"+pname,
                                               xtitle=occTitle,
                                               ytitle=longMetric[mname],
                                               nbins=occ_nbins, lims=occ_range,
                                               text="{} <= Max Q < {}".format(chg_lo,chg_hi_s,name))
                    #plotHist(vals[indices].flatten(),"hist_{}_{}chg{}".format(mname,chg_lo,chg_hi_s),
                    #                           xtitle=longMetric[mname],
                    #                           nbins=chglog_nbins, lims=None,
                    #                           )

                    print('filling2', model_name, pname)
                    
                # displays
                #hi_index = (np.where(vals>np.quantile(vals,0.9)))[0]
                #lo_index = (np.where(vals<np.quantile(vals,0.2)))[0]
                #if len(hi_index)>0:
                #    hi_index = np.random.choice(hi_index, min(Nevents,len(hi_index)), replace=False)
                #    visDisplays(hi_index, input_Q,input_calQ, alg_out, (cnn_enQ if algname=='ae' else np.array([])),(conv2d if algname=='ae' else None), name=name+"_Q90")
                #    hi_index = np.random.choice(hi_index, min(Nevents,len(hi_index)), replace=False)
                #    visDisplays(hi_index, input_Q,input_calQ, alg_out, (cnn_enQ if algname=='ae' else np.array([])),(conv2d if algname=='ae' else None), name=name+"_Q90_2")
                #if len(lo_index)>0:
                #    lo_index = np.random.choice(lo_index, min(Nevents,len(lo_index)), replace=False)
                #    visDisplays(lo_index, input_Q,input_calQ, alg_out, (cnn_enQ if algname=='ae' else np.array([])),(conv2d if algname=='ae' else None), name=name+"_Q20")
                #    lo_index = np.random.choice(lo_index, min(Nevents,len(lo_index)), replace=False)
                #    visDisplays(lo_index, input_Q,input_calQ, alg_out, (cnn_enQ if algname=='ae' else np.array([])),(conv2d if algname=='ae' else None), name=name+"_Q20_2")
            
    # overlay different metrics
    for mname in metrics:
        chgs=[]
        occs=[]
        if(not options.skipPlot):
            for algname in alg_outs:
                name = mname+"_"+algname
                chgs += [(algname, plots["chg_"+mname+"_"+algname])]
                occs += [(algname, plots["occ_"+mname+"_"+algname])]
            xt = logMaxTitle if options.rescaleInputToMax else logTotTitle
            OverlayPlots(chgs,"overlay_chg_"+mname,xtitle=xt,ytitle=mname)
            OverlayPlots(occs,"overlay_occ_"+mname,xtitle=occTitle,ytitle=mname)

            # binned comparisons
            for iocc, occ_lo in enumerate(occ_bins):
                occ_hi = 9e99 if iocc+1==len(occ_bins) else occ_bins[iocc+1]
                occ_hi_s = 'MAX' if iocc+1==len(occ_bins) else str(occ_hi)
                pname = "chg_{}occ{}_{}".format(occ_lo,occ_hi_s,name)
                pname = "chg_{}occ{}".format(occ_lo,occ_hi_s)
                chgs=[(algname, plots[pname+"_"+mname+"_"+algname]) for algname in alg_outs]
                OverlayPlots(chgs,"overlay_chg_{}_{}occ{}".format(mname,occ_lo,occ_hi_s),
                             xtitle=logMaxTitle,ytitle=mname,
                             text="{} <= occupancy < {}".format(occ_lo,occ_hi_s,name))
            for ichg, chg_lo in enumerate(chg_bins):
                chg_hi = 9e99 if ichg+1==len(chg_bins) else chg_bins[ichg+1]
                chg_hi_s = 'MAX' if ichg+1==len(chg_bins) else str(chg_hi)
                pname = "occ_{}chg{}".format(chg_lo,chg_hi_s)
                occs=[(algname, plots[pname+"_"+mname+"_"+algname]) for algname in alg_outs]
                OverlayPlots(occs,"overlay_occ_{}_{}chg{}".format(mname,chg_lo,chg_hi_s),
                             xtitle=occTitle, ytitle=mname,
                             text="{} <= Max Q < {}".format(chg_lo,chg_hi_s,name))

    return plots,summary_dict



def trainCNN(options, args, pam_updates=None):
    # List devices:
    print(device_lib.list_local_devices())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Is GPU available? ", tf.config.list_physical_devices('GPU'))

       
    if ("nElinks_%s"%options.nElinks not in options.inputFile):
       if not options.overrideInput:
         print ("Are you sure you're using the right input file??")
         print ("nElinks={0} while 'nElinks_{0}' isn't in '{1}'".format(options.nElinks,options.inputFile))
         print ("Otherwise BC, STC settings will be wrong!!")
         print ("Exiting...")
         exit(0)

    # from tensorflow.keras import backend
    # backend.set_image_data_format('channels_first')
    if os.path.isdir(options.inputFile):
        df_arr = []
        for infile in os.listdir(options.inputFile):
            if os.path.isdir(options.inputFile+infile): continue
            infile = os.path.join(options.inputFile,infile)
            df_arr.append(pd.read_csv(infile, dtype=np.float64, header=0, nrows = options.nrowsPerFile, usecols=[*range(0, 48)]))
            #df_arr.append(pd.read_csv(infile, dtype=np.float64, header=0,  usecols=[*range(0, 48)]))
        data = pd.concat(df_arr)
        data = data.loc[(data.sum(axis=1) != 0)] #drop rows where occupancy = 0
        data.describe()
    else:
        data = pd.read_csv(options.inputFile, dtype=np.float64, usecols=[*range(0, 48)])
        data = data.loc[(data.sum(axis=1) != 0)] #drop rows where occupancy = 0
    print('input data shape:',data.shape)

    data_values = data.values

    if(options.double):
        doubled_data = double_data(data_values.copy())
        print ('doubled the data. new shape is',doubled_data.shape)
        data_values = doubled_data

    # plotHist(data.values.flatten(),"TCQ_all",xtitle="Q (all cells)",ytitle="TCs",
    #              stats=False,logy=True,nbins=200,lims=[-0.5,199.5])
    # from 20 to 200 ADCs, distribution is approx f(x) = -8.05067e+03 + 1.26147e+06/x + 1.48390e+08/x^2
    # for nelink=2 sample
    # >>> f2 =  ROOT.TF1( "f2", "[1]/x+[0]+[2]/pow(x,2)",20,199)
    # >>> h.Fit(f2,"","",20,199)


    occupancy_all = np.count_nonzero(data_values,axis=1)
    occupancy_all_1MT = np.count_nonzero(data_values>35,axis=1)
    normdata,maxdata,sumdata = normalize(data_values.copy(),rescaleInputToMax=options.rescaleInputToMax)
    maxdata = maxdata / 35. # normalize to units of transverse MIPs
    sumdata = sumdata / 35. # normalize to units of transverse MIPs

    if options.occReweight:
        weights_occ = getWeights(occupancy_all_1MT,50,0,50)
        weights_maxQ = getWeights(maxdata,50,0,50)

    models = buildmodels(options,pam_updates)

    eval_settings={
        # compression algorithms, autoencoder and more traditional benchmarks
        'algnames' : ['ae','stc','thr_lo','thr_hi','bc'],
        # metrics to compute on the validation dataset
        'metrics' : {
            'EMD'      :emd,
            #'dMean':d_weighted_mean,
            #'dRMS':d_abs_weighted_rms,
            #'cross_corr':cross_corr,
        },
        "occ_nbins"   :12,
        "occ_range"   :(0,24),
        "occ_bins"    : [0,2,5,10,15],
        "chg_nbins"   :20,
        "chg_range"   :(0,200),
        "chglog_nbins":20,
        "chglog_range":(0,2.5),
        "chg_bins"    :[0,2,5,10,50],
        "occTitle"    :r"occupancy [1 MIP$_{\mathrm{T}}$ TCs]"       , 
        "logMaxTitle" :r"log10(Max TC charge/MIP$_{\mathrm{T}}$)",
        "logTotTitle" :r"log10(Sum of TC charges/MIP$_{\mathrm{T}}$)",
    }
    if options.full:
        more_metrics = {
            #'dMean':d_weighted_mean,
            #'dRMS':d_abs_weighted_rms,
            #'zero_frac':(lambda x,y: np.all(y==0)),
            # 'SSD'      :ssd,
        }
        eval_settings['metrics'].update(more_metrics)

    orig_dir = os.getcwd()
    if not os.path.exists(options.odir): os.mkdir(options.odir)
    os.chdir(options.odir)
    # plot occupancy once
    if(not options.skipPlot): 
        plotHist(occupancy_all.flatten(),"occ_all",xtitle="occupancy (all cells)",ytitle="evts",
                 stats=False,logy=True,nbins=50,lims=[0,50])
        plotHist(occupancy_all_1MT.flatten(),"occ_1MT",xtitle=r"occupancy (1 MIP$_{\mathrm{T}}$ cells)",ytitle="evts",
                 stats=False,logy=True,nbins=50,lims=[0,50])
        plotHist(np.log10(maxdata.flatten()),"maxQ_all",xtitle=eval_settings['logMaxTitle'],ytitle="evts",
                 stats=False,logy=True,nbins=20,lims=[0,2.5])
        plotHist(np.log10(sumdata.flatten()),"sumQ_all",xtitle=eval_settings['logTotTitle'],ytitle="evts",
                 stats=False,logy=True,nbins=20,lims=[0,2.5])
    # keep track of each models performance
    perf_dict={}
    for model in models:
        model_name = model['name']
        if not os.path.exists(model_name): os.mkdir(model_name)
        os.chdir(model_name)

        if model['isQK']:
            m = qDenseCNN(weights_f=model['ws'])
            print ("m is a qDenseCNN")
            #m.extend = True # for extra inputs
        elif model['isDense2D']:
            m = dense2DkernelCNN(weights_f=model['ws'])
            print ("m is a dense2DkernelCNN")
        else:
            m = denseCNN(weights_f=model['ws'])
            print ("m is a denseCNN")
        m.setpams(model['pams'])
        m.init()
        shaped_data                     = m.prepInput(normdata)
        if options.evalOnly:
            val_input = shaped_data
            val_ind = np.array(range(len(shaped_data)))
            train_input = val_input[:0] #empty with correct shape
            train_ind = val_ind[:0]
            print('training shape',train_input.shape)
            print('validation shape',val_input.shape)
        else:
            val_input, train_input, val_ind, train_ind = split(shaped_data)
        m_autoCNN , m_autoCNNen         = m.get_models()
        model['m_autoCNN'] = m_autoCNN
        model['m_autoCNNen'] = m_autoCNNen

        val_max = maxdata[val_ind]
        val_sum = sumdata[val_ind]
        if options.occReweight:
           train_weights = np.multiply(weights_maxQ[train_ind], weights_occ[train_ind])
        else:
           train_weights = np.ones(len([train_input]))

        if options.maxVal>0:
            print('clipping outputs')
            val_input = val_input[:options.maxVal]
            val_max = val_max[:options.maxVal]
            val_sum = val_sum[:options.maxVal]


        if model['ws']=='':
            if options.quickTrain: 
                train_input = train_input[:5000]
                train_weights = train_weights[:5000]
            if options.occReweight:
                history = train(m_autoCNN,m_autoCNNen,
                                train_input,train_input,val_input,
                                name=model_name,
                                n_epochs = options.epochs,
                                train_weights=train_weights)
            else:
                history = train(m_autoCNN,m_autoCNNen,
                                train_input,train_input,val_input,
                                name=model_name,
                                n_epochs = options.epochs,
                                )
        else:
            # do we want to save models if we do not train?
            #save_models(m_autoCNN,model_name,model['isQK'])
            pass


        print("Evaluate AE")
        input_Q, cnn_deQ, cnn_enQ = m.predict(val_input)
        ## use physical arrangements for display
        print('input_Q shape',input_Q.shape)
        input_calQ  = m.mapToCalQ(input_Q)   # shape = (N,48) in CALQ order
        print('input_calQ shape',input_calQ.shape)
        output_calQ_fr = m.mapToCalQ(cnn_deQ)   # shape = (N,48) in CALQ order
        print("Save CSVs")
        ## csv files for RTL verification
        N_csv= (options.nCSV if options.nCSV>=0 else input_Q.shape[0]) # about 80k
        AEvol = m.pams['shape'][0]* m.pams['shape'][1] *  m.pams['shape'][2] 
        np.savetxt("verify_input.csv", input_Q[0:N_csv].reshape(N_csv,AEvol), delimiter=",",fmt='%.12f')
        np.savetxt("verify_input_calQ.csv", input_calQ[0:N_csv].reshape(N_csv,48), delimiter=",",fmt='%.12f')
        np.savetxt("verify_output.csv",cnn_enQ[0:N_csv].reshape(N_csv,m.pams['encoded_dim']), delimiter=",",fmt='%.12f')
        np.savetxt("verify_decoded.csv",cnn_deQ[0:N_csv].reshape(N_csv,AEvol), delimiter=",",fmt='%.12f')
        np.savetxt("verify_decoded_calQ.csv",output_calQ_fr[0:N_csv].reshape(N_csv,48), delimiter=",",fmt='%.12f')



        # re-normalize outputs of AE for comparisons
        print("Restore normalization")
        input_Q_abs = np.array([input_Q[i]*(val_max[i] if options.rescaleInputToMax else val_sum[i]) for i in range(0,len(input_Q))])
        input_calQ  = np.array([input_calQ[i]*(val_max[i] if options.rescaleInputToMax else val_sum[i]) for i in range(0,len(input_calQ)) ])  # shape = (N,48) in CALQ order
        output_calQ =  unnormalize(output_calQ_fr.copy(), val_max if options.rescaleOutputToMax else val_sum, rescaleOutputToMax=options.rescaleOutputToMax)
        #occupancy_0MT = np.count_nonzero(input_Q_abs.reshape(len(input_Q),48),axis=1)
        #occupancy_1MT = np.count_nonzero(input_Q_abs.reshape(len(input_Q),48)>1.,axis=1)
        occupancy_0MT = np.count_nonzero(input_calQ.reshape(len(input_Q),48),axis=1)
        occupancy_1MT = np.count_nonzero(input_calQ.reshape(len(input_Q),48)>1.,axis=1)

        charges = {
            'input_Q'    : input_Q,               # shape = (N,4,4,3)
            'input_Q_abs': input_Q_abs,           # shape = (N,4,4,3) (in abs Q)
            'input_calQ' : input_calQ,            # shape = (N,48) (in abs Q)   (in CALQ 1-48 order)
            'output_calQ': output_calQ,           # shape = (N,48) (in abs Q)   (in CALQ 1-48 order)
            'output_calQ_fr': output_calQ_fr,     # shape = (N,48) (in Q fr)   (in CALQ 1-48 order)
            'cnn_deQ'    : cnn_deQ,
            'cnn_enQ'    : cnn_enQ,
            'val_sum'    : val_sum,
            'val_max'    : val_max,
        }
        aux_arrs = {
           'occupancy_1MT':occupancy_1MT 
        } 
        
        #perf_dict[model_name] , model['summary_dict'] = evalModel(model,charges,aux_arrs,eval_settings,options)
        perf_dict[model['label']] , model['summary_dict'] = evalModel(model,charges,aux_arrs,eval_settings,options)

        if not options.skipPlot:
            with open('./performance_%s.pkl'%model_name, 'wb') as file_pi:
                pickle.dump({ model['label']:  perf_dict[model['label']]}  , file_pi)
        occupancy=occupancy_0MT
        if(not options.skipPlot): plotHist(occupancy.flatten(),"occ",xtitle="occupancy",ytitle="evts",
                                               stats=False,logy=True,nbins=50,lims=[0,50])

        # keep track of plot results
        with open(model_name+"_pams.json",'w') as f:
            f.write(json.dumps(m.get_pams(),indent=4))
        
        os.chdir('../')

    # compare the relative performance of each model
    compareModels(models,perf_dict,eval_settings,options)

    os.chdir(orig_dir)
    return     

if __name__== "__main__":

    parser = optparse.OptionParser()
    parser.add_option('-o',"--odir", type="string", default = 'CNN/',dest="odir", help="input TSG ntuple")
    parser.add_option("--loss", type="string", default = '',dest="loss", help="force loss function to use")
    parser.add_option('-i',"--inputFile", type="string", default = 'CALQ_output_10x.csv',dest="inputFile", help="input TSG ntuple")
    parser.add_option("--quantize", action='store_true', default = False,dest="quantize", help="Quantize the model with qKeras. Default precision is 16,6 for all values.")
    parser.add_option("--dryRun", action='store_true', default = False,dest="dryRun", help="dryRun")
    parser.add_option("--epochs", type='int', default = 100, dest="epochs", help="n epoch to train")
    parser.add_option("--nELinks", type='int', default = 5, dest="nElinks", help="n of e-links")
    parser.add_option("--skipPlot", action='store_true', default = False,dest="skipPlot", help="skip the plotting step")
    parser.add_option("--full", action='store_true', default = False,dest="full", help="run all algorithms and metrics")
    parser.add_option("--quickTrain", action='store_true', default = False,dest="quickTrain", help="train w only 5k events for testing purposes")
    parser.add_option("--retrain", action='store_true', default = False,dest="retrain", help="retrain models even if weights are already present for testing purposes")
    parser.add_option("--double", action='store_true', default = False,dest="double", help="test PU400 by combining PU200 events")
    parser.add_option("--evalOnly", action='store_true', default = False,dest="evalOnly", help="only evaluate the NN on the input sample, no train")
    parser.add_option("--overrideInput", action='store_true', default = False,dest="overrideInput", help="disable safety check on inputs")
    parser.add_option("--nCSV", type='int', default = 50, dest="nCSV", help="n of validation events to write to csv")
    parser.add_option("--maxVal", type='int', default = -1, dest="maxVal", help="n of validation events to consider")
    parser.add_option("--AEonly", type='int', default=1, dest="AEonly", help="run only AE algo")
    parser.add_option("--rescaleInputToMax", type='int', default=0, dest="rescaleInputToMax", help="recale the input images so the maximum deposit is 1. Else normalize")
    parser.add_option("--rescaleOutputToMax", type='int', default=0, dest="rescaleOutputToMax", help="recale the output images to match the initial sum")
    parser.add_option("--nrowsPerFile", type='int', default=500000, dest="nrowsPerFile", help="load nrowsPerFile in a directory")
    parser.add_option("--occReweight", action='store_true', default = False,dest="occReweight", help="Train with per-event weight on TC occupancy")
    (options, args) = parser.parse_args()
    trainCNN(options,args)
