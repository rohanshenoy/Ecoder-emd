# Ecoder
Training ECON-T autoencoder model

The original repository is (https://github.com/kakwok/Ecoder).

## Setup

Local setup using miniconda3
```
source install_miniconda3.sh # if conda is not installed in your computer
source setup.sh # if you do not have the ecoder-env environment (i.e. if it is your first time)
conda activate ecoder-env
```

Clone the repository (or your own fork):
```
git clone git@github.com:cmantill/Ecoder.git
cd Ecoder/
``` 

To obtain qkeras (for training with quantized bit constraints), clone the repo locally: 
```
git clone https://github.com/google/qkeras
```

## Input data

Get data from FNAL LPC:
```
mkdir data/
scp -r cmslpc-sl7.fnal.gov:/uscms/home/kkwok/eos/ecoder/V11/ data/
```

or from cernbox: https://cernbox.cern.ch/index.php/s/YpAWu24aw6EaBk7

Electron samples: (neLinks 2-5 with sim-Energy information) `/eos/uscms/store/user/dnoonan/AE_TrainingData/NewData/Skim/ele200PUData_TrainingData_SignalAllocation/`

## Training

Once you have downloaded the data you can do:

```
python3 train.py -i data/V11/SampleSplitting_SignalAllocation/nElinks_5/shuffled/  -o ./test/ --epoch 1 --AEonly 1 --nELinks 5 --noHeader
```

here:
- `-i data/V11/SampleSplitting_SignalAllocation/nElinks_5/shuffled/`: Input directory or input file. Here we have an example of the input training. In this case the .csv files are already shuffled and they do not contain headers. So the `--noHeader` option is needed.
- `-o ./test/`: Output directory. Here we have `test` as an example of the output directory. Change this to something more meaninful for future tests.
- `--epoch 1`: this represents the number of epochs to train the model. We usually train for ~100 epochs.
- `--AEonly`: is an argument to only evaluate the AutoEncoder algorithm (instead of also the other algorithms such as BSC,STC,Threshold..). This is usually an option that we want to include in this stage
- `--nElinks`: this is the number of active elinks for the input dataset (in this case 5 for the signal allocation algorithm.) Find more about the number of elinks and allocation [here](https://github.com/cmantill/ECONAutoencoderStudy/blob/master/fragments/README.MD#number-of-elinks).
- `--noHeader`: this argument is needed only for this shuffled dataset since it has no header. Other datasets (e.g. for SingleElectron) will contain headers.

Other possible arguments are:
- `--loss`: if there is a default loss function to use. We generalize want to use `telescopeMSE8x8` that is defined [here](https://github.com/cmantill/Ecoder/blob/master/telescope.py#L168-L170) and called [here](https://github.com/cmantill/Ecoder/blob/master/networks.py#L4). So we usually want to leave this argument empty.
- `--quantize`: Quantize the model with qKeras.
- `--skipPlot`: Skip the plotting steps.
- `--full`: Run all the algorithms and metrics.
- `--quickTrain`: Train with only 5k events for testing purposes.
- `--retrain`: Re-train models even if weights are already present.
- `--evalOnly`: Use weights of already trained model to evaluate the performance of the network on the input sample.
- `--double`: Double the dataset by combining events from the input dataset.
- `--overrideInput`: Disable safety check on inputs
- `--nCSV`: Number of validation events to write to output .csv files.
- `--maxVal`: Clip outputs to maxVal
- `--rescaleInputToMax`: Rescale the input images so the maximum deposit is 1.
- `--rescaleOutputToMax`: Rescale the output images to match the initial sum of charge.
- `--nrowsPerFile`: Load only this number of rows per file
- `--occReweight`: Train with per-event weight on TC occupancy
- `--maskPartials`: Mask partial modules in the input files
- `--maskEnergies`: Mask input charges with energy fractions <= 0.05 (or some other value defined in train.py)
- `--saveEnergy`: save SimEnergy column from input data
- `--noHeader`: Input data has no header
- `--models`: Models to run.

The default model is:
- 8x8_c8_S2_tele: i.e. a CNN model with 8x8 arranged inputs, 3x3 kernel dimensions and 8 filters, and a stride size of 2x2, using the telescope function as a loss function.

The model parameters are defined as a dictionary in `networks.py`.
Other AE models can also be found in this module.

Other scripts in this repository are:
- denseCNN.py: where the Conv2D model is defined using keras and TF
- qdenseCNN.py: an equivalent version that uses qKeras
- get_flops.py: computes the number of flops (operations by a given network). Needs an model .json file as an input.
- telescope.py: computes the telescope metric that is used as a loss function
- dense2DkernelCNN.py (deprecated)
- ot_tf.py: for sinkhorn metric for qdenseCNN

Useful functions to plot, define or visualize metrics are defined in the `utils/` folder.

## Convert model to a TF graph

This is useful to evaluate the network with other data (e.g. for our physics studies).
The script uses tensorflow 2.4.0 and keras 2.2.4-tf (the versions can be found in CMSSW_11_1_0).
Can also be ran locally - just need to make sure versions are compatible.

```
### convert the decoder model
python3 converttoTF.py -o ./graphs/ -i decoder_Jul24_keras.json --outputGraph decoder --outputLayer decoder_output/Sigmoid 
### convert the encoder model
python3 converttoTF.py -o ./graphs/ -i encoder_Jul24_keras.json --outputGraph encoder --outputLayer encoded_vector/Relu 
```

## How the input data is produced

Training data are generated with the Level 1 Trigger Primitives simulation.
Full documentation can be found [here](https://twiki.cern.ch/twiki/bin/viewauth/CMS/HGCALTriggerPrimitivesSimulation).

For the purposes of training data we run this repository with no threshold applied (Threshold Sum algorithm in the concentrator chip) so that it saves all of the trigger cells (TCs).

The TPG training ntuples can be found in the cmslpc cluster:
- v0: `/eos/uscms/store/user/lpchgcal/ConcentratorNtuples/L1THGCal_Ntuples/TrainingSamples_Sept2020/`

These are taken as input for the ECON-T python simulation [ECONT_Emulator](https://github.com/cmantill/ECONT_Emulator/).
This simulation has 2 steps:
1.  Producing data that goes to the ECON-T input eLinks.

The `getDataFromMC.py` script reads in the ntuples, and produces a .csv file formatted to look as the input data that goes to the ECON-T input eLinks.
This script also computes the total simEnergy alongside the simEnergy sum of each module.

Then, `ECONT_Emulator.py` simulates the behavior of the ECON-T. For the purposes of training data it simulates the channel re-arrangement (through the multiplexer MUX) and the calibration that converts from charge to transverse-charge.
The calibrated charges are saved in a `CALQ.csv` file, which is taken as the input to the AutoEncoder training.

Output data from these intermediate steps can be found here:
- v3: `/eos/uscms/store/user/lpchgcal/ECON_Verification_Data/${sample}Data_v11Geom_layer_${layer}_job${job}.tgz`

Each of these tar files contain subdirectories for each wafer within that layer.
These contain all the actual data needed, but are split up based on wafer, not number of eLinks.

2. Group the outputs based on the link allocations and the number of output eLinks from the ECON-T to the backend - this is the bottleneck of how much data can be transmitted by the algorithms.

This step can be found [in this folder](https://github.com/dnoonan08/ECONT_Emulator/tree/master/MakeTrainingDataSets).

- `getTrainingData.sh`/`submitTrainingData.jdl`: runs the ECONT emulator
- `splitByLinks.sh`/`submitByLink.jdl`: splits the output of the emulator step into csv files based on link allocation
- `runSkim.sh`/`submitSkim.jdl` : skims out partials and modules with 0 sim energy

[comment]: <> (The `SimEnergyPatch` folder contains scripts to include information on the SimEnergy:)
[comment]: <> (- `fixSimEnergyValues.sh` is the main script which calls the others)
[comment]: <> (- `findEventTotal.py` loops through all the SimEnergy csv’s to find total sim energy in each event)
[comment]: <> (- `mergeSimEnergy.py` updates the `CALQ.csv` file for all the wafers with the total sim energy, and simEnergy as a float rather than int)

##### Number of elinks
The number of eLinks sent will vary. A single eLink, running at 1.28 Gbps, can send 32 bits of data per BX, so with:
- 2 eLinks, i.e. 64 bits per BX: we can send 3 bits for each of the 16 encoded layer values (48 total, + leaving room for a 4 bit header and 9 bit sum of the charge in the module - this is the format of the auto-encoder output)
- 3 eLinks, i.e. 96 bits per Bx: we can send 5 bits per encoded value
- 4 eLinks: 7 bits per encoded value
- 5+ eLinks: we send the full 9 bit precision of the encoded values

##### Allocation schemes
The two allocation schemes are:
- PU allocation: assigns output eLinks in a way that areas of the detector with the highest occupancy (which are typically due to PileUp) get the most eLinks.
The distribution of eLinks is roughly a function of the eta angle (since Pileup peaks at high eta).
For this, we just have a [lookup of the links assigned to each module in a csv file](https://github.com/dnoonan08/ECONT_Emulator/blob/master/Utils/ModuleLinkSummary.csv)
Here, `Layer`, `ModU`, `ModV` will uniquely identify a 8 inch wafer (or ECON, since there is one ECON-T per wafer), and ECONT_eTX is the number of links assigned to that ECON-T.

- Signal allocation: assigns more output eLinks to the	modules in layers of the detector that are closest to the shower max.
This means that the [links are assigned by layer](https://github.com/dnoonan08/ECONT_Emulator/blob/master/Utils/linkAllocation.py).
With the shower max (layer 9) being assigned 5 eLinks, the neighboring odd numbered layers (only odd layers are readout for the trigger in EE) get 4 links, then 3, then the rest get 2.
This is found for the shower max of an electron, but would be used no matter what the sample is (so e.g. using a ttbar vs electron sample will not matter here).

Our default training was trained on Signal Allocation, ttbar sample, PU 200, nElinks 5 (e.g. layer 9).

### Output data to use.
This output of the ECONT simulation code will be similar to the `CALQ.csv` file, but with a couple extra columns for book-keeping purposes.
Each row in the file represents a module but several rows can correspond to one MC event or "entry" column.

The output (and training) data from this last step (from the ECON-T simulation), can be found here:
- ttbar: (neLinks 2-5 and no sim-Energy infomation) https://cernbox.cern.ch/index.php/s/YpAWu24aw6EaBk7
- electron samples: (neLinks 2-5 with sim-Energy information) `/eos/uscms/store/user/dnoonan/AE_TrainingData/NewData/Skim/ele200PUData_TrainingData_SignalAllocation/`
