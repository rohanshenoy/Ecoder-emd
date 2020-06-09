import numpy as np
import pandas as pd
import optparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

from train import trainCNN
from utils import plotGraphErr

def plotScan(x,outs,name,odir,xtitle="n bits"):
    outs = pd.concat(outs)
    for metric in ['ssd','corr','emd']:
        plotGraphErr(x, outs[metric], outs[metric+'_err'], name+"_"+metric,
                 odir,xtitle=xtitle,ytitle=metric)
    outs.to_csv(odir+"/"+name+".csv")
    return

def BitScan(options, args):

    if False:
        # test inputs
        bits = [i+3 for i in range(6)]
        updates = [{'nBits_input':{'total': b, 'integer': 2}} for b in bits]
        outputs = [trainCNN(options,args,u) for u in updates]
        plotScan(bits,outputs,"test_input_bits",options.odir,xtitle="total input bits")

    if False:
        # test weights
        bits = [i+1 for i in range(8)]
        updates = [{'nBits_weight':{'total': 2*b+1, 'integer': b}} for b in bits]
        outputs = [trainCNN(options,args,u) for u in updates]
        plotScan(bits,outputs,"test_weight_bits",options.odir,xtitle="total weight bits")

    if True:
        # test encoded bits
        bits = [4,6,8,10,12,16]
        updates = [{'nBits_encod':{'total': b, 'integer': b/2},'encoded_dim':int(64/b)} for b in bits]
        outputs = [trainCNN(options,args,u) for u in updates]
        plotScan(bits,outputs,"test_encod_bits",options.odir,xtitle="bits per encoded node")

    exit(0)

    

if __name__== "__main__":

    parser = optparse.OptionParser()
    parser.add_option('-o',"--odir", type="string", default = 'CNN/',dest="odir", help="input TSG ntuple")
    parser.add_option('-i',"--inputFile", type="string", default = 'CALQ_output_10x.csv',dest="inputFile", help="input TSG ntuple")
    parser.add_option("--quantize", action='store_true', default = False,dest="quantize", help="Quantize the model with qKeras. Default precision is 16,6 for all values.")
    parser.add_option("--dryRun", action='store_true', default = False,dest="dryRun", help="dryRun")
    parser.add_option("--epochs", type='int', default = 100, dest="epochs", help="n epoch to train")
    parser.add_option("--skipPlot", action='store_true', default = False,dest="skipPlot", help="skip the plotting step")
    parser.add_option("--nCSV", type='int', default = 50, dest="nCSV", help="n of validation events to write to csv")
    parser.add_option("--rescaleInputToMax", action='store_true', default = False,dest="rescaleInputToMax", help="recale the input images so the maximum deposit is 1. Else normalize")
    (options, args) = parser.parse_args()
    #trainCNN(options,args)
    BitScan(options,args)
