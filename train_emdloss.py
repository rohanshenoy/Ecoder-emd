"""
For training EMD_CNN with different hyperparameters
@author: Rohan
"""
import get_emdloss #Script for training the CNN to approximate EMD
import pandas as pd
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i',"--inputFile", type=str, default='nElinks_5/', dest="inputFile",
                    help="input TSG files")
parser.add_argument("--epochs", type=int, default = 64, dest="num_epochs",
                    help="number of epochs to train")
parser.add_argument("--best", type=int, default = 8, dest="best_num",
                    help="number of emd_models to save")

def main(args):

    data=[]

    if os.path.isdir(args.inputFile):

        for infile in os.listdir(args.inputFile):
            
            data=os.path.join(args.inputFile+infile)
            
    current_directory=os.getcwd()

    #Data to track the performance of various CNN models

    df=[]
    mean_data=[]
    std_data=[]
    nfilt_data=[]
    ksize_data=[]
    neuron_data=[]
    numlayer_data=[]
    convlayer_data=[]
    epoch_data=[]
    z_score=[]

    """
    Hyperparamters listed below are after going through these permutations:
    
    n_filt_hyp=[32,64,128,256]
    kernel_hyp=[1,3,5]
    neuron_hyp=[32,64,128,256]
    num_dense_hyp=[1,2]
    num_conv2d_hyp=[1,2,3,4]
    
    #Training loops: (384 permutations in total)
    
    for num_filt in n_filt_hyp:
        for kernel_size in kernel_hyp:
            for num_dens_neurons in neuron_hyp:
                for num_dens_layers in num_dense_hyp:
                    for num_conv_2d in num_conv2d_hyp:
                           
                        for i in [0,1,2]:
                            obj=get_emdloss.EMD_CNN(True)
                            mean, sd = obj.ittrain(data, num_filt,kernel_size, num_dens_neurons, num_dens_layers, num_conv_2d,num_epochs+i)
                            mean_data.append(mean)
                            std_data.append(sd)
                            nfilt_data.append(num_filt)
                            ksize_data.append(kernel_size)
                            neuron_data.append(num_dens_neurons)
                            numlayer_data.append(num_dens_layers)
                            convlayer_data.append(num_conv_2d)
                            epoch_data.append(num_epochs+i)
                            z=abs(mean)/sd
                            z_score.append(z)

                            #The best 8 models are saved in a list, as [sd, num_filt,kernel_size,num_dens_neurons,num_dens_layers,num_conv_2d,num_epochs+i]
                            #We rank the models per their standard deviation(sd), ie model[0] 

                            max=0;
                            for i in range(0,9):
                                model=best[i]
                                maximum=best[max]
                                if model[0]>maximum[0]:
                                    max=i
                            best[max]=[sd,num_filt,kernel_size, num_dens_neurons, num_dens_layers, num_conv_2d,num_epochs+i]
   
    """
    #List of lists of Hyperparamters <- currently initialized from previous training
    hyp_list=[[32,5,256,1,3],
              [32,5,32,1,4],
              [64,5,32,1,4],
              [128,5,32,1,4],
              [128,5,64,1,3],
              [32,5,128,1,3],
              [128,3,256,1,4],
              [128,5,256,1,4]
              ]

    num_epochs=args.num_epochs
    best_num=args.best_num

    #Best EMD Models Per SD

    best=[[0,0,0,0,0,0,0]]*(best_num)     

    for hyp in hyp_list:
        num_filt=hyp[0]
        kernel_size=hyp[1]
        num_dens_neurons=hyp[2]
        num_dens_layers=hyp[3]
        num_conv_2d=hyp[4]
        
        #Each model per set of hyperparamters is trained thrice to avoid bad initialitazion discarding a good model. (We vary num_epochs by 1 to differentiate between these 3 trainings)
        
        for i in [0,1,2]:
            obj=get_emdloss.EMD_CNN(True)
            mean, sd = obj.ittrain(data, num_filt,kernel_size, num_dens_neurons, num_dens_layers, num_conv_2d,num_epochs+i)
            mean_data.append(mean)
            std_data.append(sd)
            nfilt_data.append(num_filt)
            ksize_data.append(kernel_size)
            neuron_data.append(num_dens_neurons)
            numlayer_data.append(num_dens_layers)
            convlayer_data.append(num_conv_2d)
            epoch_data.append(num_epochs+i)
            z=abs(mean)/sd
            z_score.append(z)
            
            #The best 8 models are saved in a list, as [sd, num_filt,kernel_size,num_dens_neurons,num_dens_layers,num_conv_2d,num_epochs+i]
            #We rank the models per their standard deviation(sd), ie model[0] 

            max=0;
            for j in range(0,best_num):
                model=best[j]
                maximum=best[max]
                if model[0]>maximum[0]:
                    max=j
            best[max]=[sd,num_filt,kernel_size, num_dens_neurons, num_dens_layers, num_conv_2d,num_epochs+i]


    for_pdata=[mean_data,std_data,nfilt_data,ksize_data,neuron_data,numlayer_data,convlayer_data,z_score,epoch_data]

    #Saving data from the entire optimization training 
    
    opt_data_directory=os.path.join(current_directory,r'EMD_Loss CNN Optimization Data.xlsx')
    df=pd.DataFrame(for_pdata)
    df.to_excel(opt_data_directory)
    
    #Saving another .xlsx for the best models    
    best_data_directory=os.path.join(current_directory,r'Best EMD_CNN Models.xlsx')

    df=pd.DataFrame(best)
    df.to_excel(best_data_directory)

    #Preparing Best EMD Loss Models for ECON Training as {i}.h5, which during CNN optimization were saved as: 
    #(num_filt)+(kernel_size)+(num_dens_neurons)+(num_dens_layers)+(num_conv_2d)+(num_epochs)+best.h5 
    # and Renaming them to {i}.h5, i={1,2,...8}

    model_directory=os.path.join(os.getcwd(),r'emd_loss_models')
    i=1
    for models in best:
        mpath=os.path.join(model_directory,str(models[1])+str(models[2])+str(models[3])+str(models[4])+str(models[5])+str(models[6])+'best.h5')
        os.rename(mpath,os.getcwd()+'/'+str(i)+'.h5')
        i+=1

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


