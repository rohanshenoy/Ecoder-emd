import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def decode_ECON(mantissa, exp, n_mantissa=3,n_exp=4):
    if exp==0: return mantissa
    mantissa += (1<<n_mantissa)
    return mantissa << (exp-1)

def encode_ECON(val,n_mantissa=3,n_exp=4):
    # return (mantissa , exponent)
    if val==0: return (0, 0)
    msb = int(np.log2(val))
    if msb<n_mantissa: return (val, 0)
    exp = max(msb-n_mantissa+1,(1<<n_exp)-1)
    mantissa = (val>>exp) - (1<<(n_mantissa-1))
    return (mantissa,exp)

def test_econ():
    for m in range(1<<3):    
        for e in range(1<<4):
            val = decode_ECON(m,e)
            m1, e1 = encode_ECON(val)
            print(m,e,'-->',val,'-->',m1,e1)
