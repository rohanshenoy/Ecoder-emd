# Ecoder
ECON-T autoencoder model

## Setup 

### On VM `klijnsma-gpu3` or any other Linux VM

Get data from EOS at FNAL:
```
/uscms/home/kkwok/eos/ecoder/V11
```

Setup environment using miniconda3
```
source install_miniconda3.sh #if your first time
source setup.sh #also if your first time
conda activate ecoder-env
```

### Setup on LPC 
If you are working on the LPC cluster working node, use the following scripts to setup the environment
```
source LPC_envSetup.sh      ##do this for the first time
source lpc_env.sh           ##do this everytime
```

## Juypter notebook demos
Following files illustrates prototypes of different autoencoder architectures

- `auto.ipynb` - 1D deep NN autoencoder demo
- `auto_CNN.ipynb` - 2D CNN autoencoder demo
- `Auto_qCNN.ipynb` - 2D quantized CNN autoencoder, trained  with qKeras demo

qkeras instructions: https://github.com/google/qkeras

## Training scripts 
Scripts to explore hyperparameters choices:

- `models.py`   - constructs and compile simple model architectures
- `denseCNN.py` - model class for constructing conv2D-dense architectures
- `train.py`    - train(or load weights) and evaluate models

## Example usage:

```
## edit parameters setting inside train.py
## train with 1 epoch to make sure model parameters are OK, output to a trainning folder
python train.py -i /uscms/home/kkwok/eos/ecoder/V11/signal/nElinks_5/ -o ./test/ --epoch 1 --AEonly 1 --nELinks 5
## train the weights with max 150 epoch 
python train.py -i /uscms/home/kkwok/eos/ecoder/V11/signal/nElinks_5/ -o ./test/ --epoch 150 --AEonly 1 --nELinks 5

## After producing a `.hdf5` file from trainning, you can re-run the model skipping the trainning phase.
## Do so by simply setting the model parameter 'ws' to `modelname.hdf5`
```

## Convert to a constant tensorflow graph

Using tensorflow `2.4.0` and keras `2.2.4-tf`.
Can be obtained from `CMSSW_11_1_0`.
```
### convert the decoder model
python3 converttoTF.py -o ./graphs/ -i decoder_Jul24_keras.json --outputGraph decoder --outputLayer decoder_output/Sigmoid 
### convert the encoder model
python3 converttoTF.py -o ./graphs/ -i encoder_Jul24_keras.json --outputGraph encoder --outputLayer encoded_vector/Relu 

```
