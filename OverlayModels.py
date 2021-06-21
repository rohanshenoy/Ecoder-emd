# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from qDenseCNN import qDenseCNN
from denseCNN import denseCNN

import matplotlib.pyplot as plt
import matplotlib

import graphUtil
import numba
import pickle

from plotWafer import plotWafer

eval_settings={
    "occ_nbins"   :12,
    "occ_range"   :(0,24),
    "occ_bins"    : [0,2,5,10,15],
    "chg_nbins"   :20,
    "chg_range"   :(0,200),
    "chglog_nbins":10,
    "chglog_range":(0,2.5),
    "chg_bins"    :[0,2,5,10,50],
    "occTitle"    :r"occupancy [1 MIP$_{\mathrm{T}}$ TCs]"       ,
    "logMaxTitle" :r"log10(Max TC charge/MIP$_{\mathrm{T}}$)",
    "logTotTitle" :r"log10(Sum of TC charges/MIP$_{\mathrm{T}}$)",
}

odir = '/home/rsshenoy/ecoder/perf_plots'
tag = ''
flist = [
    "C:/Users/Rohan/DuarteLab/ecoder/emd_loss_6a/performance.pkl",
]

def loadPickles(flist):
    perf_dict = {}
    for f in flist:
        with open(f,'rb') as f_pkl:
            perf_dict.update(pickle.load(f_pkl))
    return perf_dict

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

perf_dict = loadPickles(flist)
print(perf_dict)