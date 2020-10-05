
from qDenseCNN import qDenseCNN
from denseCNN import denseCNN

import matplotlib.pyplot as plt
import matplotlib

from martinModels import models
import graphUtil
import numba
import pickle

from plotWafer import plotWafer

from train import OverlayPlots,emd,d_weighted_mean,d_abs_weighted_rms


def loadPickles(flist):
    perf_dict = {}
    for f in flist:
        with open(f,'rb') as f_pkl:
            perf_dict.update( pickle.load(f_pkl))
    return perf_dict

def makePlots(flist,eval_settings,odir='.',tag=''):

    perf_dict = loadPickles(flist)
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

    for mname in metrics:
        chgs=[]
        occs=[]
        for model_name in perf_dict:
            plots = perf_dict[model_name]
            chgs += [(model_name, plots["chg_"+mname+"_ae"])]
            occs += [(model_name, plots["occ_"+mname+"_ae"])]
        xt =eval_settings['logTotTitle']
        OverlayPlots(occ ,"ae_comp_%s_occ_"%tag+mname,xtitle=xt,ytitle=mname,odir=odir,ylim=(0,4))
        OverlayPlots(chgs,"ae_comp_%s_chg_"%tag+mname,xtitle=xt,ytitle=mname,odir=odir,ylim=(0,5))

        #        for iocc, occ_lo in enumerate(occ_bins):
        #            occ_hi = 9e99 if iocc+1==len(occ_bins) else occ_bins[iocc+1]
        #            occ_hi_s = 'MAX' if iocc+1==len(occ_bins) else str(occ_hi)
        #            indices = (occupancy_1MT >= occ_lo) & (occupancy_1MT < occ_hi)
        #            pname = "chg_{}occ{}_{}".format(occ_lo,occ_hi_s,name)
        #            plots[pname] = plotProfile(np.log10(val_max[indices]), vals[indices],"profile_"+pname,
        #                                       xtitle=logMaxTitle,
        #                                       nbins=chglog_nbins, lims=chglog_range,
        #                                       ytitle=longMetric[mname],
        #                                       text="{} <= occupancy < {}".format(occ_lo,occ_hi_s,name))
        #            print('filling1', model_name, pname)
        #        for ichg, chg_lo in enumerate(chg_bins):
        #            chg_hi = 9e99 if ichg+1==len(chg_bins) else chg_bins[ichg+1]
        #            chg_hi_s = 'MAX' if ichg+1==len(chg_bins) else str(chg_hi)
        #            indices = (val_max >= chg_lo) & (val_max < chg_hi)
        #            pname = "occ_{}chg{}_{}".format(chg_lo,chg_hi_s,name)
        #            plots[pname] = plotProfile(occupancy_1MT[indices], vals[indices],"profile_"+pname,
        #                                       xtitle=occTitle,
        #                                       ytitle=longMetric[mname],
        #                                       nbins=occ_nbins, lims=occ_range,
        #                                       text="{} <= Max Q < {}".format(chg_lo,chg_hi_s,name))
        #            print('filling2', model_name, pname)



eval_settings={
    # compression algorithms, autoencoder and more traditional benchmarks
    'algnames' : ['ae','stc','thr_lo','thr_hi','bc'],
    # metrics to compute on the validation dataset
    'metrics' : {
        'EMD'      :emd,
        #'dMean':d_weighted_mean,
        #'dRMS':d_abs_weighted_rms,
    },
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

odir = 'perf_plots/'
tag = ''
flist = [
    "../V11/signal/nElinks_5/Sep1_CNN_keras_norm/performance_Sep1_CNN_keras_norm.pkl",
    "../V11/signal/nElinks_5/Sep26_663/performance_Sep26_663.pkl",
     "../V11/signal/nElinks_5/Sep26_SepConv_663/performance_Sep26_SepConv_663.pkl", 
]
makePlots(flist,eval_settings,odir)
