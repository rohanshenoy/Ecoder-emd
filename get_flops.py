import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from denseCNN import MaskLayer

def get_flops_from_pb_v2(model_json):
    with open(model_json,'r') as fjson:
        model = model_from_json(fjson.read(),custom_objects={'MaskLayer':MaskLayer})
        #hdf5  = model_json.replace('json','hdf5')
        #model.load_weights(hdf5)
        model.summary()
        print(model)
        inputs = [
            tf.TensorSpec([1] + inp.shape[1:], inp.dtype) for inp in model.inputs
        ]
        full_model = tf.function(model).get_concrete_function(inputs)
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        layers = [op.name for op in frozen_func.graph.get_operations()]

        for l in layers: print(l)
        # Calculate FLOPS with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )
        return flops.total_float_ops, model.count_params()
 
def get_flops_from_model(model):
        inputs = [
            tf.TensorSpec([1] + inp.shape[1:], inp.dtype) for inp in model.inputs
        ]
        full_model = tf.function(model).get_concrete_function(inputs)
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        layers = [op.name for op in frozen_func.graph.get_operations()]

        #for l in layers: print(l)
        # Calculate FLOPS with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )
        return flops.total_float_ops

if __name__=='__main__':

    flist = [
        #'./oldModels/4x4_v4/encoder_4x4_v4.json'
        #'./oldModels/4x4_norm_v4/encoder_4x4_norm_v4.json',
        #'./oldModels/4x4_norm_v8/encoder_4x4_norm_v8.json',
        #'./oldModels/4x4_norm_d10/encoder_4x4_norm_d10.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_norm/encoder_Sep1_CNN_keras_norm.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_v12/encoder_Sep1_CNN_keras_v12.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_v13/encoder_Sep1_CNN_keras_v13.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_v14/encoder_Sep1_CNN_keras_v14.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_v8/encoder_Sep1_CNN_keras_v8.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_v9/encoder_Sep1_CNN_keras_v9.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_v10/encoder_Sep1_CNN_keras_v10.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_v1/encoder_Sep1_CNN_keras_v1.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_v2/encoder_Sep1_CNN_keras_v2.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_v3/encoder_Sep1_CNN_keras_v3.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_v4/encoder_Sep1_CNN_keras_v4.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_v7/encoder_Sep1_CNN_keras_v7.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_v5/encoder_Sep1_CNN_keras_v5.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_v6/encoder_Sep1_CNN_keras_v6.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_v15/encoder_Sep1_CNN_keras_v15.json',
         #'./V11/signal/nElinks_5/Sep1_CNN_keras_v16/encoder_Sep1_CNN_keras_v16.json'
         #'./V11/signal/nElinks_5/Sep9_CNN_keras_8x8_v1/encoder_Sep9_CNN_keras_8x8_v1.json',
         #'./V11/signal/nElinks_5/Sep9_CNN_keras_8x8_v5/encoder_Sep9_CNN_keras_8x8_v5.json',
         #'./V11/signal/nElinks_5/Sep9_CNN_keras_8x8_v6/encoder_Sep9_CNN_keras_8x8_v6.json',
         #'./V11/signal/nElinks_5/Sep9_CNN_keras_8x8_v2/encoder_Sep9_CNN_keras_8x8_v2.json',
         #'./V11/signal/nElinks_5/Sep9_CNN_keras_8x8_v3/encoder_Sep9_CNN_keras_8x8_v3.json',
         #'./V11/signal/nElinks_5/Sep9_CNN_keras_8x8_v4/encoder_Sep9_CNN_keras_8x8_v4.json',
         #'./V11/signal/nElinks_5/Sep9_CNN_keras_8x8_v7/encoder_Sep9_CNN_keras_8x8_v7.json',
         #'./V11/signal/nElinks_5/Sep9_CNN_keras_8x8_v8/encoder_Sep9_CNN_keras_8x8_v8.json',
         #'./V11/signal/nElinks_5/Sep9_CNN_keras_8x8_v9/encoder_Sep9_CNN_keras_8x8_v9.json',
         #'./V11/signal/nElinks_5/Sep9_CNN_keras_8x8_v7.2/encoder_Sep9_CNN_keras_8x8_v7.2.json',
         #'./V11/signal/nElinks_5/Sep9_CNN_keras_8x8_v8.2/encoder_Sep9_CNN_keras_8x8_v8.2.json',
         #'./V11/signal/nElinks_5/Sep9_CNN_keras_8x8_v9.2/encoder_Sep9_CNN_keras_8x8_v9.2.json',

         #'./V11/signal/nElinks_5/Sep9_CNN_keras_8x8_v10/encoder_Sep9_CNN_keras_8x8_v10.json',
         #'./V11/signal/nElinks_5/Sep9_CNN_keras_8x8_v11/encoder_Sep9_CNN_keras_8x8_v11.json',

         #'./V11/signal/nElinks_5/Sep21_CNN_keras_SepConv_v1/encoder_Sep21_CNN_keras_SepConv_v1.json',
         #'./V11/signal/nElinks_5/Sep21_CNN_keras_SepConv_v2/encoder_Sep21_CNN_keras_SepConv_v2.json',
         #'./V11/signal/nElinks_5/Sep21_CNN_keras_SepConv_v3/encoder_Sep21_CNN_keras_SepConv_v3.json',

           #'./V11/signal/nElinks_5/Sep26_SepConv_663_c4/encoder_Sep26_SepConv_663_c4.json',
           #'./V11/signal/nElinks_5/Sep26_SepConv_663_c2/encoder_Sep26_SepConv_663_c2.json',
           #'./V11/signal/nElinks_5/Sep26_SepConv_663/encoder_Sep26_SepConv_663.json',
          #'./V11/signal/nElinks_5/Oct8_SepConv_663_pool/encoder_Oct8_SepConv_663_pool.json',
          #'./V11/signal/nElinks_5/Oct8_SepConv_663_c4_pool/encoder_Oct8_SepConv_663_c4_pool.json',
          #'./V11/signal/nElinks_5/Oct8_SepConv_663_c2_pool/encoder_Oct8_SepConv_663_c2_pool.json',
         # './V11/signal/nElinks_5/Oct8_663/encoder_Oct8_663.json',

        #    './V11/signal/nElinks_5/Oct8_SepConv_663_c8_k5_vpad/encoder_Oct8_SepConv_663_c8_k5_vpad.json',
        #    './V11/signal/nElinks_5/Oct8_SepConv663_c10_k5_vpad/encoder_Oct8_SepConv663_c10_k5_vpad.json',
            './V11/signal/nElinks_5/Oct30_8x8_k5/encoder_Oct30_8x8_k5.json',
         ]
    results = {}
    for f in flist:
        results[f.split('/')[-1]]={}
        flops, pams = get_flops_from_pb_v2(f)
        results[f.split('/')[-1]]['flops']= flops
        results[f.split('/')[-1]]['pams']= pams
    for k in results:
        print(k,results[k]['flops'],results[k]['pams'])
