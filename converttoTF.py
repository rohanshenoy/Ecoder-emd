#!/usr/bin/env python

#import imp
#try:
#    imp.find_module('setGPU')
#    import setGPU
#except ImportError:
#    found = False
            
          
from tensorflow.keras.models import model_from_json  
from argparse import ArgumentParser
from keras import backend as K
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

def saveTFgraph(tfsession,pred_node_names,tfoutpath,graphname):
    saver = tfv1.train.Saver()
    
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io

    constant_graph = graph_util.convert_variables_to_constants(
        tfsession, tfsession.graph.as_graph_def(), pred_node_names)

    f = graphname+'_constantgraph.pb.ascii'
    tfv1.train.write_graph(constant_graph, tfoutpath, f, as_text=True)
    print('saved the graph definition in ascii format at: ', os.path.join(tfoutpath, f))

    f = graphname+'_constantgraph.pb'
    tfv1.train.write_graph(constant_graph, tfoutpath, f, as_text=False)
    print('saved the graph definition in pb format at: ', os.path.join(tfoutpath, f))


    #graph_io.write_graph(constant_graph, args.outputDir, output_graph_name, as_text=False)
    #print('saved the constant graph (ready for inference) at: ', os.path.join(args.outputDir, output_graph_name))

    saver.save(tfsession, tfoutpath)

tfback._get_available_gpus = _get_available_gpus

## use tfv1 for conversion
if tf.__version__.startswith("2."):
    tfv1 = tf.compat.v1
tfv1.disable_eager_execution()


parser = ArgumentParser('')
parser.add_argument('-i','--inputModel',dest='inputModel',default='./models/no-weights.json')
parser.add_argument('-o','--outputDir',dest='outputDir',default='./')
args = parser.parse_args()

print args.outputDir


f_model = args.inputModel
with open(f_model,'r') as f:
    model = model_from_json(f.read())

hdf5  = f_model.replace('json','hdf5')
model.load_weights(hdf5)

encoder = model.get_layer("encoder")
decoder = model.get_layer("decoder")
print(model.summary())
print(encoder.summary())
print(decoder.summary())

## get_session is deprecated in tf2
tfsession = tfv1.keras.backend.get_session()

## From output node of (q)DenseCNN model framework + tfsession.graph.as_graph_def()
decoder_node_names = ['decoder_output/Sigmoid']
encoder_node_names = ['encoder/encoded_vector/Relu']

saveTFgraph(tfsession,decoder_node_names,'./testgraphs/','decoder')
saveTFgraph(tfsession,encoder_node_names,'./testgraphs/','encoder')
saveTFgraph(tfsession,decoder_node_names+encoder_node_names,'./testgraphs/','fullmodel')
saveTFgraph(tfsession,encoder_node_names,'./testgraphs/','decoder')

