import numpy as np
import pandas as pd
import numba
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append("/home/therwig/data/sandbox/hgcal/Ecoder/")
from hgcal_encode import encode,decode

@numba.jit
def normalize(data,rescaleInputToMax=False):
    norm =[]
    sumd =[]
    for i in range(len(data)):
        norm.append( data[i].max() )
        sumd.append( data[i].sum() )
        data[i] = 1.*data[i]/data[i].sum()
    return data,np.array(norm),np.array(sumd)

# Encode fraction as integer, making the (inefficient)
#   choice to use 1 integer bit for 1.0)
#   fractions are 0, 1, ..., 2*(nBits-1)-1 
#   and 1.0 is 2*(nBits-1)
def encode_fraction(x, nBits):
    #always reserve 1 bit for sign
    nFracBits=nBits-1
    return np.floor(x * (2**nFracBits))
def decode_fraction(x, nBits):
    #always reserve 1 bit for sign
    nFracBits=nBits-1
    return x / (2**nFracBits)

def test():
    # import data
    inFileName = "/home/therwig/data/sandbox/hgcal/Ecoder/data/danny/nElinks_2/ttbar_Layer5_nELinks2.csv"
    if os.path.isdir(inFileName):
        df_arr = []
        for infile in os.listdir(inFileName):
            infile = os.path.join(inFileName,infile)
            df_arr.append(pd.read_csv(infile, dtype=np.float64, header=0, usecols=[*range(1, 49)]))
        data = pd.concat(df_arr)
        data = data.loc[(data.sum(axis=1) != 0)] #drop rows where occupancy = 0
        print(data.shape)
        data.describe()
    else:
        #data = pd.read_csv(inFileName,dtype=np.float64, header=0, usecols=[*range(1, 49)])
        data = pd.read_csv(inFileName,dtype=np.float64, header=0, usecols=[*range(1, 49)],nrows=1000)

    # (3) find the module sum with “full” 27/25b precision (21/19b per TC * 48 TC)
    # (4) find the AE input fractions using on the “full" precision 27/25b values
    tc_data = data.to_numpy()
    normdata,maxdata,sumdata = normalize(tc_data.copy())

    # The inputs from the CSV files are already encoded and decoded
    #
    # (1) perform 4E+3M ROC TC encoding to 7b
    #data_encoded = np.array([encode(int(x),expBits=4,mantBits=3,asInt=True) for x in data.to_numpy().flatten()]).reshape(normdata.shape)
    # (2) perform ECON TC decoding to 21/19 bits
    #data_decoded = np.array([decode(x,expBits=4,mantBits=3) for x in data_encoded.flatten()]).reshape(normdata.shape)

    results={}
    for sum_exp, sum_mant in [(5,3),(5,4),(5,5),(16,16)]:
        for frac_bits in [4,5,6,7,8,10,12,16]:

            # (5) encode the sum with 5E+3M, 5E+4M, and 5E+5M
            sum_encoded = np.array([encode(int(x),expBits=sum_exp,mantBits=sum_mant,asInt=True) for x in sumdata])
        
            # (6) encode the fractions with 4,5,…,8b 
            #normdata_encoded = np.array([encode_fraction(x,nBits=5) for x in normdata])
            normdata_encoded = encode_fraction(normdata,nBits=frac_bits)
        
            # (g) PART I: decode the sum and fractions ...
            sum_decoded = np.array([decode(x,expBits=sum_exp,mantBits=sum_mant) for x in sum_encoded])
            normdata_decoded = decode_fraction(normdata_encoded,nBits=frac_bits)

            # the decorded fractions may not sum to 1, so we can re-normalize
            renormdata_size = np.array([(normdata_decoded[i].sum()) for i in range(len(normdata_decoded))])
            renormdata_decoded = np.array([normdata_decoded[i]/(normdata_decoded[i].sum()) for i in range(len(normdata_decoded))])

            # (g) PART II: ... and multiply to find the value of each TC
            data_decoded_from_norm = np.array([normdata_decoded[i]*sum_decoded[i] for i in range(len(sum_decoded))])

            results[(sum_exp, sum_mant, frac_bits,1)] = renormdata_size

            # (h) plot TC residual  vs TC energy (as you already do for other encoding)
            results[(sum_exp, sum_mant, frac_bits)] = data_decoded_from_norm

            # print some checks
            if False and sum_exp==5 and sum_mant==5 and frac_bits==8:
                evtno=10
                print("Input data\n",tc_data[evtno].reshape(12,4))
                print("Sum data ",sumdata[evtno])
                print("Sum encoded ",sum_encoded[evtno])
                print("Sum decoded ",sum_decoded[evtno])
                print("Norm data\n",normdata[evtno].reshape(12,4))
                print("Norm encoded\n",normdata_encoded[evtno].reshape(12,4))
                print("Norm decoded\n",normdata_decoded[evtno].reshape(12,4))
                print("Norm*Sum decoded\n",data_decoded_from_norm[evtno].reshape(12,4))
                frac_diff = np.divide(data_decoded_from_norm-tc_data,tc_data,out=np.zeros_like(tc_data),where=(tc_data!=0))
                print("Frac diff\n",frac_diff[evtno].reshape(12,4))

                

    
    for r,resid in results.items():
        if len(r)==3:
            # scatter plot
            tag = "{}exp_{}mant_frac{}".format(*r)
            title = "Sum: {}b exp, {}b mant, Fraction: {}b".format(*r)
            #frac_diff = np.divide(resid-tc_data,tc_data,out=np.zeros_like(tc_data),where=(tc_data!=0))
            frac_diff = np.divide(resid,tc_data,out=np.zeros_like(tc_data),where=(tc_data!=0))
            plt.scatter(tc_data.flatten(), frac_diff.flatten())
            plt.title(title)
            # plt.ylabel('(Decoded - True) / True')
            plt.ylabel('Decoded / True')
            plt.xlabel('TC cell value')
            plt.xscale('log')
            plt.xlim(1,1e4)
            plt.savefig("plots/resid_{}.png".format(tag))
            plt.close()

            # heat map
            dflat = tc_data.flatten()
            logs = np.where(dflat>1,np.log10(dflat),0.)
            nonempty = logs > 0
            #fig, ax = plt.subplots(1, 1)
            plt.hist2d(logs[nonempty],(frac_diff.flatten())[nonempty], bins=40, norm=matplotlib.colors.LogNorm())
            plt.title(title)
            # plt.ylabel('(Decoded - True) / True')
            plt.ylabel('Decoded / True')
            plt.xlabel('log10(TC cell value)')
            plt.colorbar()#pcm, ax=ax[0], extend='max')
            plt.savefig("plots/resid_heat_{}.png".format(tag))
            plt.close()

            
    # quick hack to print normalizations
    for r,resid in results.items():
        if len(r)==4:
            tag = "{}exp_{}mant_frac{}_{}".format(*r)
            plt.hist(resid,50)
            plt.title('Model loss {}'.format(tag))
            plt.ylabel('')
            plt.xlabel('diff')
            #plt.legend(['Train', 'Test'], loc='upper right')
            plt.savefig("plots/sum_{}.png".format(tag))
            plt.close()


    
    return


if __name__== "__main__":
    test()
