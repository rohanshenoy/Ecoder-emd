import numpy as np
import tensorflow as tf
import pandas as pd
import optparse
from tensorflow.python.client import device_lib
from tensorflow import keras as kr
from tensorflow.keras import losses
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numba
import json

#for earth movers distance calculation
import ot

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
hexMetric = ot.dist(hexCoords, hexCoords, 'euclidean')
MAXDIST = 16.08806614
def emd(_x, _y, threshold=-1):
    if (np.sum(_x)==0 or np.sum(_y)==0): return MAXDIST
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

@numba.jit
def normalize(data,rescaleInputToMax=False):
    norm =[]
    for i in range(len(data)):
        if rescaleInputToMax:
            norm.append( data[i].max() )
            data[i] = 1.*data[i]/data[i].max()
        else:
            norm.append( data[i].sum() )
            data[i] = 1.*data[i]/data[i].sum()
    return data,np.array(norm)

def AddEMD(options, args, pam_updates=None):
  
    # from tensorflow.keras import backend
    # backend.set_image_data_format('channels_first')
    if os.path.isdir(options.inputFile):
        df_arr = []
        for infile in os.listdir(options.inputFile):
            infile = os.path.join(options.inputFile,infile)
            df_arr.append(pd.read_csv(infile, dtype=np.float64, header=0, usecols=[*range(1, 49)]))
        data = pd.concat(df_arr)
        data = data.loc[(data.sum(axis=1) != 0)] #drop rows where occupancy = 0
        print(data.shape)
        data.describe()
    else:
        data = pd.read_csv(options.inputFile,dtype=np.float64)
    #data = pd.read_csv(options.inputFile,dtype=np.float64)  ## big  300k file
    normdata,maxdata = normalize(data.values.copy(),rescaleInputToMax=options.rescaleInputToMax)
    
        shaped_data                     = m.prepInput(normdata)
        val_input, train_input, val_ind = split(shaped_data)
        m_autoCNN , m_autoCNNen         = m.get_models()
        val_max = maxdata[val_ind]

        if model['ws']=='':
            if options.quickTrain: train_input = train_input[:5000]
            history = train(m_autoCNN,m_autoCNNen,train_input,val_input,name=model_name,n_epochs = options.epochs)
        else:
            save_models(m_autoCNN,model_name)

        summary_dict = {
            'name':model_name,
            'en_pams' : m_autoCNNen.count_params(),
            'tot_pams': m_autoCNN.count_params(),}

        input_Q,cnn_deQ ,cnn_enQ   = m.predict(val_input)
        
        ## csv files for RTL verification
        N_csv= (options.nCSV if options.nCSV>=0 else input_Q.shape[0]) # about 80k
        np.savetxt("verify_input.csv", input_Q[0:N_csv].reshape(N_csv,48), delimiter=",",fmt='%.12f')
        np.savetxt("verify_output.csv",cnn_enQ[0:N_csv].reshape(N_csv,m.pams['encoded_dim']), delimiter=",",fmt='%.12f')
        np.savetxt("verify_decoded.csv",cnn_deQ[0:N_csv].reshape(N_csv,48), delimiter=",",fmt='%.12f')
        
        stc1_Q = make_supercells(input_Q)
        stc2_Q = make_supercells(input_Q,shareQ=True)
        thr_lo_Q = threshold(input_Q,val_max,47) # 1.35 transverse MIPs
        thr_hi_Q = threshold(input_Q,val_max,69) # 2.0  transverse MIPs
        occupancy = np.count_nonzero(input_Q.reshape(len(input_Q),48),axis=1)
        alg_outs = {'ae' : cnn_deQ,
                    'stc1': stc1_Q,
                    'stc2': stc2_Q,
                    'thr_lo': thr_lo_Q,
                    'thr_hi': thr_hi_Q,
                }

        # to generate event displays
        Nevents = 8
        index = np.random.choice(input_Q.shape[0], Nevents, replace=False)

        # compute metrics for each alg
        for algname, alg_out in alg_outs.items():
            # charge fraction comparison
            if(not options.skipPlot): plotHist([input_Q.flatten(),alg_out.flatten()],
                                               algname+"_fracQ",xtitle="charge fraction",ytitle="Cells",
                                               stats=False,logy=True,leg=['input','output'])
            input_Q_abs = np.array([input_Q[i]*val_max[i] for i in range(0,len(input_Q))])
            alg_out_abs = np.array([alg_out[i]*val_max[i] for i in range(0,len(alg_out))])
            if(not options.skipPlot): plotHist([input_Q_abs.flatten(),alg_out_abs.flatten()],
                                               algname+"_absQ",xtitle="absolute charge",ytitle="Cells",
                                               stats=False,logy=True,leg=['input','output'])
            # event displays
            if(not options.skipPlot): visDisplays(index, input_Q, alg_out, (cnn_enQ if algname=='ae' else np.array([])), name=algname)
            for mname, metric in metrics.items():
                name = mname+"_"+algname
                vals = np.array([metric(input_Q[i],alg_out[i]) for i in range(0,len(input_Q))])
                vals = np.sort(vals)
                model[name]        = np.round(np.mean(vals), 3)
                model[name+'_err'] = np.round(np.std(vals), 3)
                summary_dict[name]        = model[name]
                summary_dict[name+'_err'] = model[name+'_err']
                if(not options.skipPlot) and (not('zero_frac' in mname)):
                    plotHist(vals,"hist_"+name,xtitle=longMetric[mname])
                    hi_index = (np.where(vals>np.quantile(vals,0.9)))[0]
                    lo_index = (np.where(vals<np.quantile(vals,0.2)))[0]
                    # visualize(input_Q,cnn_deQ,cnn_enQ,index,name=model_name)
                    if len(hi_index)>0:
                        hi_index = np.random.choice(hi_index, min(Nevents,len(hi_index)), replace=False)
                        visDisplays(hi_index, input_Q, alg_out, (cnn_enQ if algname=='ae' else np.array([])), name=algname)
                    if len(lo_index)>0:
                        lo_index = np.random.choice(lo_index, min(Nevents,len(lo_index)), replace=False)
                        visDisplays(lo_index, input_Q, alg_out, (cnn_enQ if algname=='ae' else np.array([])), name=algname)
                
        print('summary_dict',summary_dict)
        summary = summary.append(summary_dict, ignore_index=True)

        with open(model_name+"_pams.json",'w') as f:
            f.write(json.dumps(m.get_pams(),indent=4))
        
        os.chdir('../')
    os.chdir(orig_dir)
    print(summary)
    return summary    

if __name__== "__main__":

    parser = optparse.OptionParser()
    parser.add_option('-o',"--odir", type="string", default = 'CNN/',dest="odir", help="input TSG ntuple")
    parser.add_option('-i',"--inputFile", type="string", default = 'CALQ_output_10x.csv',dest="inputFile", help="input TSG ntuple")
    parser.add_option("--quantize", action='store_true', default = False,dest="quantize", help="Quantize the model with qKeras. Default precision is 16,6 for all values.")
    parser.add_option("--dryRun", action='store_true', default = False,dest="dryRun", help="dryRun")
    parser.add_option("--epochs", type='int', default = 100, dest="epochs", help="n epoch to train")
    parser.add_option("--skipPlot", action='store_true', default = False,dest="skipPlot", help="skip the plotting step")
    parser.add_option("--quickTrain", action='store_true', default = False,dest="quickTrain", help="train w only 5k events for testing purposes")
    parser.add_option("--nCSV", type='int', default = 50, dest="nCSV", help="n of validation events to write to csv")
    parser.add_option("--rescaleInputToMax", action='store_true', default = False,dest="rescaleInputToMax", help="recale the input images so the maximum deposit is 1. Else normalize")
    (options, args) = parser.parse_args()
    trainCNN(options,args)
