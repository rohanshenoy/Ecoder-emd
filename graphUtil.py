import tensorflow as tf
import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

## Load qkeras/Keras model from json file
def loadModel(f_model):
    with open(f_model,'r') as f:
        if 'QActivation' in f.read():
            from qkeras import QDense, QConv2D, QActivation,quantized_bits,Clip,QInitializer
            f.seek(0)
            model = model_from_json(f.read(),
                                    custom_objects={'QActivation':QActivation,
                                                    'quantized_bits':quantized_bits,
                                                    'QConv2D':QConv2D,
                                                    'QDense':QDense,
                                                    'Clip':Clip,
                                                    'QInitializer':QInitializer})
            hdf5  = f_model.replace('json','hdf5')
            model.load_weights(hdf5)
        else:
            f.seek(0)
            model = model_from_json(f.read())
            hdf5  = f_model.replace('json','hdf5')
        model.load_weights(hdf5)
    return model

def setQuanitzedWeights(model,f_pkl):
    with open(f_pkl, 'rb') as f:
        #weights as a dictionary
        ws = pickle.load(f)
        for layer_name in ws.keys():
            layer = model.get_layer(layer_name)
            layer.set_weights(ws[layer_name]['weights'])
    return model

## Write model to graph
def outputFrozenGraph(model,outputName="frozen_graph.pb",logdir='./',asText=False):
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=logdir,
                      name=outputName,
                      as_text=asText)

## Load frozen graph
def loadFrozenGraph(graph,printGraph=False):
    with tf.io.gfile.GFile(graph, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    tf.compat.v1.import_graph_def(graph_def, name="")
    
    # Build the tensor from the first and last node of the graph
    #     if isQK:
    #         inputs=["x:0"],
    #         outputs=["Identity:0"]
    #     else:
    #         inputs=["input_1:0"]
    #         outputs=["encoded_vector/Relu:0"]
    #
    inputs = graph_def.node[0].name+":0"
    outputs= graph_def.node[-1].name+":0"

    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=inputs,
                                    outputs=outputs,
                                    print_graph=printGraph)
    return frozen_func


## Helper function to load graph
def wrap_frozen_graph(graph_def, inputs,outputs,print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    if print_graph:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        if print_graph == True:
            for layer in layers:
                print(layer)
        print("-" * 50)
    
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

## Get the output from layer_index of input x from a model
def layerOutput(model,layer_index,x):
    m = tf.keras.models.Model(
        inputs =model.inputs,
        outputs=model.layers[layer_index].output
    )
    return m.predict(x)

## plotAll the weights from model
def plotWeights(model,nBins=20):
    plt.figure(figsize=(8,6))
    for ilayer in range(1,len(model.layers)):
        if len(model.layers[ilayer].get_weights())>0:
            label = model.layers[ilayer].name
            data = np.histogram(model.layers[ilayer].get_weights()[0])
            print(ilayer, label,'unique weights',len(np.unique(model.layers[ilayer].get_weights()[0])))
            hep.histplot(data[0],data[1],label=label)
        else:
            print(ilayer,'no weights')
    plt.xlabel('weights')
    plt.ylabel('Entries')
    plt.yscale('log')            
    plt.legend()
    plt.savefig("%s_weights.pdf"%model.name)
    
#plot outputs from each layers given an input
def plotOutputs(model,x,layer_indices=[],nBins=10):
    plt.figure(figsize=(8,6))
    if len(layer_indices)>0:
        layers = layer_indices
    else:
        layers = range(1,len(model.layers))
    for ilayer in layers:
        label = model.layers[ilayer].name
        output,bins = np.histogram(layerOutput(model,ilayer,x).flatten(),nBins)
        hep.histplot(output,bins,label=label)
    plt.yscale('log')
    plt.tight_layout()
    plt.legend()
    plt.xlabel('Output values')
    plt.ylabel('Entries')
    str_layers = "_".join([str(l) for l in layer_indices])
    plt.savefig("hist_outputs_%s.pdf"%str_layers)
    return
