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
        data[i] = 1.*data[i]/data[i].max()
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
    normdata,maxdata,sumdata = normalize(data.values.copy())

    # (1) perform 4E+3M ROC TC encoding to 7b
    data_encoded = np.array([encode(int(x),expBits=4,mantBits=3,asInt=True) for x in data.to_numpy().flatten()]).reshape(normdata.shape)

    # (2) perform ECON TC decoding to 21/19 bits
    data_decoded = np.array([decode(x,expBits=4,mantBits=3) for x in data_encoded.flatten()]).reshape(normdata.shape)

    # (5) encode the sum with 5E+3M, 5E+4M, and 5E+5M
    sum_encoded = np.array([encode(int(x),expBits=5,mantBits=4,asInt=True) for x in sumdata])

    # (6) encode the fractions with 4,5,…,8b 
    #normdata_encoded = np.array([encode_fraction(x,nBits=5) for x in normdata])
    normdata_encoded = encode_fraction(normdata,nBits=5)

    # (g) decode the sum and fractions and multiply to find the value of each TC
    sum_decoded = np.array([decode(x,expBits=5,mantBits=4) for x in sum_encoded])
    #normdata_decoded = np.array([decode_fraction(x,nBits=5) for x in normdata_encoded])
    normdata_decoded = decode_fraction(normdata_encoded,nBits=5)

    # (h) plot TC residual  vs TC energy (as you already do for other encoding)
    # ...


    
    return


if __name__== "__main__":
    test()
