import h5py
import numpy as np    

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt


def StringToTextFile(fname,s):
    with open(fname,'w') as f:
        f.write(s)
def plotHist(vals,name,odir='.',xtitle="weight values",ytitle="entries",nbins=40,lims=None,
             stats=False, logy=False, leg=None):
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


fnames = {
    'e2_nonQ':'jun1_e2_full_v1/may8_2elink_16out3b1_6b1weights/may8_2elink_16out3b1_6b1weights.hdf5',
    'e5_nonQ':'jun1_e5_full_v1/may8_2elink_16out9b1_6b1weights/may8_2elink_16out9b1_6b1weights.hdf5',
    'e2_Q':'jun1_e2_full_q_v1/may8_2elink_16out3b1_6b1weights_Input16b6i_Accum16b6i_Weight6b1i_Encod3b1i/may8_2elink_16out3b1_6b1weights_Input16b6i_Accum16b6i_Weight6b1i_Encod3b1i.hdf5',
    'e5_Q':'jun1_e5_full_q_v1/may8_2elink_16out9b1_6b1weights_Input16b6i_Accum16b6i_Weight6b1i_Encod9b1i/may8_2elink_16out9b1_6b1weights_Input16b6i_Accum16b6i_Weight6b1i_Encod9b1i.hdf5',
    #'/home/therwig/data/sandbox/hgcal/Ecoder/may27_nonquant_v7/may8_2elink_16out4b1_6b1weights/may8_2elink_16out4b1_6b1weights.hdf5'
}


if False:
    for n in fnames:
        f = h5py.File(fnames[n],'r+')    
    
        # check contents with f.keys(), etc...
        if 'nonQ' in n:
            conv_k = np.array(f['encoder']['conv2d']['kernel:0'])
            conv_b = np.array(f['encoder']['conv2d']['bias:0'])
            dense_k = np.array(f['encoder']['encoded_vector']['kernel:0'])
            dense_b = np.array(f['encoder']['encoded_vector']['bias:0'])
        else:
            conv_k = np.array(f['encoder']['conv2d_0_m']['kernel:0'])
            conv_b = np.array(f['encoder']['conv2d_0_m']['bias:0'])
            dense_k = np.array(f['encoder']['encoded_vector']['kernel:0'])
            dense_b = np.array(f['encoder']['encoded_vector']['bias:0'])
    
    
        plotHist(conv_k.flatten(), n+"_conv_kernel" , odir="weight_dump")
        plotHist(conv_b.flatten(), n+"_conv_bias"   , odir="weight_dump")
        plotHist(dense_k.flatten(),n+"_dense_kernel", odir="weight_dump")
        plotHist(dense_b.flatten(),n+"_dense_bias"  , odir="weight_dump")
    
    
fname = fnames['e2_Q']
fname='jun3_qtest_v1/may8_2elink_16out3b1_6b1weights_Input16b6i_Accum16b6i_Weight6b1i_Encod3b1i/may8_2elink_16out3b1_6b1weights_Input16b6i_Accum16b6i_Weight6b1i_Encod3b1i.hdf5'    
fname='jun3_qtest_v1/may8_2elink_16out3b1_6b1weights_Input16b6i_Accum16b6i_Weight6b1i_Encod3b1i/encoder_may8_2elink_16out3b1_6b1weights_Input16b6i_Accum16b6i_Weight6b1i_Encod3b1i.hdf5'

f = h5py.File(fname,'r+')
conv_k = np.array(f['encoder']['conv2d_0_m']['kernel:0'])
conv_b = np.array(f['encoder']['conv2d_0_m']['bias:0'])
dense_k = np.array(f['encoder']['encoded_vector']['kernel:0'])
dense_b = np.array(f['encoder']['encoded_vector']['bias:0'])

print(conv_k*8)


conv_k = np.array(f['encoder']['conv2d_0_m']['kernel:0'])
