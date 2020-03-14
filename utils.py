import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

from train import trainCNN

def plotGraph(x, y, name, odir, xtitle, ytitle, leg=None):
    plt.figure()
    plt.plot(x,y)
    plt.title('')
    plt.ylabel(ytitle)
    plt.xlabel(xtitle)
    if leg: plt.legend(leg, loc='upper right')
    plt.savefig(odir+"/"+name+".png")
    plt.close()
    return

def plotGraphErr(x, y, ye, name, odir, xtitle, ytitle, leg=None):
    plt.figure()
    plt.errorbar(x,y,ye)
    plt.title('')
    plt.ylabel(ytitle)
    plt.xlabel(xtitle)
    if leg: plt.legend(leg, loc='upper right')
    plt.savefig(odir+"/"+name+".png")
    plt.close()
    return

def plotHist(vals,name,odir,xtitle="",ytitle="",nbins=40):
    plt.figure()
    plt.hist(vals,nbins)
    mu = np.mean(vals)
    std = np.std(vals)
    ax = plt.axes()
    plt.text(0.1, 0.9, name,transform=ax.transAxes)
    plt.text(0.1, 0.8, r'$\mu=%.3f,\ \sigma=%.3f$'%(mu,std),transform=ax.transAxes)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle if ytitle else 'Entries')
    plt.savefig(odir+"/"+name+".png")
    plt.close()
    return
