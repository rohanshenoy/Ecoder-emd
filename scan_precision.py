import numpy as np
import pandas as pd
import optparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

from train import trainCNN

def plotHist(x,y,ye, name, odir,xtitle, ytitle):
    plt.figure()
    plt.errorbar(x,y,ye)
    plt.title('')
    plt.ylabel(ytitle)
    plt.xlabel(xtitle)
    plt.legend(['others 16,6'], loc='upper right')
    plt.savefig(odir+"/"+name+".png")
    plt.close()
    return

def plotScan(x,outs,name,odir,xtitle="n bits"):
    outs = pd.concat(outs)
    for metric in ['ssd','corr','emd']:
        plotHist(x, outs[metric], outs[metric+'_err'], name+"_"+metric,
                 odir,xtitle=xtitle,ytitle=metric)
    return

def BitScan(options, args):
    og_odir = options.odir
    '''
    options.odir = og_odir+"/input"
    # test inputs
    bits = [i+3 for i in range(6)]
    bits = [i+3 for i in range(2)]
    updates = [{'nBits_input':{'total': b, 'integer': 2}} for b in bits]
    outputs = [trainCNN(options,args,u) for u in updates]
    plotScan(bits,outputs,"test_input_bits",options.odir,xtitle="total input bits")

    exit(0)
    '''
    '''
    # test weights
    options.odir = og_odir+"/weight"
    bits = [i+1 for i in range(8)]
    updates = [{'nBits_weight':{'total': 2*b+1, 'integer': b}} for b in bits]
    outputs = [trainCNN(options,args,u) for u in updates]
    plotScan([2*b+1 for b in bits],outputs,"test_weight_bits",og_odir,xtitle="total weight bits")
    '''

    #exit(0)

    # test accumulator
    options.odir = og_odir+"/accum"
    bits = [i+1 for i in range(8)]
    updates = [{'nBits_accum': {'total': 2*b+1, 'integer': b}} for b in bits]
    outputs = [trainCNN(options,args,u) for u in updates]
    plotScan([2*b+1 for b in bits],outputs,"test_accum_bits",og_odir,xtitle="total accumulator layer bits")

    #exit(0)

    # test encoder layer bits
    options.odir = og_odir+"/encod"
    bits = [i+1 for i in range(8)]
    updates = [{'nBits_encod': {'total': 2*b+1, 'integer': b}} for b in bits]
    outputs = [trainCNN(options,args,u) for u in updates]
    plotScan([2*b+1 for b in bits],outputs,"test_encoded_bits",og_odir,xtitle="total encoder layer bits")

    #exit(0)

    #test dense alone (conv constant)
    options.odir = og_odir+"/dense"
    bits = [i+1 for i in range(8)]
    updates = [{'nBits_dense': {'total': 2*b+1, 'integer': b}} for b in bits]
    outputs = [trainCNN(options,args,u) for u in updates]
    plotScan([2*b+1 for b in bits],outputs,"test_dense_bits",og_odir,xtitle="total dense layer bits")


    #exit(0)

    # test conv alone (dense constant)
    options.odir = og_odir+"/conv"
    bits = [i+1 for i in range(8)]
    updates = [{'nBits_conv': {'total': 2*b+1, 'integer': b}} for b in bits]
    outputs = [trainCNN(options,args,u) for u in updates]
    plotScan([2*b+1 for b in bits],outputs,"test_conv_bits",og_odir,xtitle="total convolutional layer bits")

    

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
