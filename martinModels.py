import numpy as np
edim = 16
arrange443 = np.array([0,16, 32,
                           1,17, 33,
                           2,18, 34,
                           3,19, 35,
                           4,20, 36,
                           5,21, 37,
                           6,22, 38,
                           7,23, 39,
                           8,24, 40,
                           9,25, 41,
                           10,26, 42,
                           11,27, 43,
                           12,28, 44,
                           13,29, 45,
                           14,30, 46,
                           15,31, 47])
arrange8x8 = np.array([
    28,29,30,31,0,4,8,12,
    24,25,26,27,1,5,9,13,
    20,21,22,23,2,6,10,14,
    16,17,18,19,3,7,11,15,
    47,43,39,35,35,34,33,32,
    46,42,38,34,39,38,37,36,
    45,41,37,33,43,42,41,40,
    44,40,36,32,47,46,45,44])
arrMask  =  np.array([
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,0,0,0,0,
    1,1,1,1,0,0,0,0,
    1,1,1,1,0,0,0,0,
    1,1,1,1,0,0,0,0,])
arrMask_full  =  np.array([
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,])
calQMask  =  np.array([
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,0,0,0,0,
    1,1,1,1,0,0,0,0,
    1,1,1,1,0,0,0,0,
    1,1,1,1,0,0,0,0,])

arrange8x8_2 = np.array([
    44,45,46,47,16,20,24,28,
    40,41,42,43,17,21,25,29,
    36,37,38,39,18,22,26,30,
    32,33,34,35,19,23,27,31,
    15,11, 7, 3, 3, 2, 1, 0,
    14,10, 6, 2, 7, 6, 5, 4,
    13,9,  5, 1,11,10, 9, 8,
    12,8,  4, 0,15,14,13,12])

arrMask_martin  =  np.array([
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,0,0,1,1,1,
    1,1,0,0,0,0,1,1,
    1,0,0,0,0,0,0,1,])

defaults = {    'shape':(4,4,3),
                 'channels_first': False,
                 'arrange': arrange443,
                 'encoded_dim': edim,
                 'loss': 'telescopeMSE',
                 'nBits_input'  : {'total': 10,                 'integer': 3,'keep_negative':1},
                 'nBits_accum'  : {'total': 11,                 'integer': 3,'keep_negative':1},
                 'nBits_weight' : {'total':  5,                 'integer': 1,'keep_negative':1},
}
models = [
    #{'name':'Sep1_CNN_keras_norm','label':'norm','pams':{
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #    },
    #},

#    {'name':'Sep1_CNN_keras_v12','label':'dim12','pams':{
#             'CNN_layer_nodes':[8],
#             'CNN_kernel_size':[3],
#             'CNN_pool':[0],
#             'encoded_dim': 12,
#        },
#    },
#    {'name':'Sep1_CNN_keras_v13','label':'dim20','pams':{
#             'CNN_layer_nodes':[8],
#             'CNN_kernel_size':[3],
#             'CNN_pool':[0],
#             'encoded_dim': 20,
#        },
#    },

#    {'name':'Sep1_CNN_keras_v14','label':'den16','pams':{
#             'CNN_layer_nodes':[8],
#             'CNN_kernel_size':[3],
#             'CNN_pool':[0],
#             'Dense_layer_nodes':[16] ,
#        },
#    },
#
#    {'name':'Sep1_CNN_keras_v8','label':'k[5]','pams':{
#             'CNN_layer_nodes':[8],
#             'CNN_kernel_size':[5],
#             'CNN_pool':[0],
#        },
#    },
#    {'name':'Sep1_CNN_keras_v9','label':'c[12]','pams':{
#             'CNN_layer_nodes':[12],
#             'CNN_kernel_size':[3],
#             'CNN_pool':[0],
#        },
#    },
#    {'name':'Sep1_CNN_keras_v10','label':'pool','pams':{
#             'CNN_layer_nodes':[8],
#             'CNN_kernel_size':[3],
#             'CNN_pool':[1],
#        },
#    },



#    {'name':'Sep1_CNN_keras_v1','label':'c[8,8]','pams':{
#             'CNN_layer_nodes':[8,8],
#             'CNN_kernel_size':[3,3],
#             'CNN_pool':[False,False],
#        },
#    },
#    {'name':'Sep1_CNN_keras_v2','label':'c[8,8,8]','pams':{
#             'CNN_layer_nodes':[8,8,8],
#             'CNN_kernel_size':[3,3,3],
#             'CNN_pool':[False,False,False],
#        },
#    },
#    {'name':'Sep1_CNN_keras_v3','label':'c[4,4,4]','pams':{
#             'CNN_layer_nodes':[4,4,4],
#             'CNN_kernel_size':[3,3,3],
#             'CNN_pool':[False,False,False],
#        },
#    },
#    {'name':'Sep1_CNN_keras_v4','label':'c[8,4,2]','pams':{
#             'CNN_layer_nodes':[8,4,2],
#             'CNN_kernel_size':[3,3,3],
#             'CNN_pool':[False,False,False],
#        },
#    },
#   {'name':'Sep1_CNN_keras_v7','label':'c[8,4,4,4,2],','pams':{
#            'CNN_layer_nodes':[8,4,4,4,2],
#            'CNN_kernel_size':[3,3,3,3,3],
#            'CNN_pool':[0,0,0,0,0],
#       },
#   },

#    {'name':'Sep1_CNN_keras_v5','label':'c[8,4,2]_k[5,5,3]','pams':{
#             'CNN_layer_nodes':[8,4,2],
#             'CNN_kernel_size':[5,5,3],
#             'CNN_pool':[False,False,False],
#        },
#    },
#    {'name':'Sep1_CNN_keras_v6','label':'c[4,4,4]_k[5,5,3]','pams':{
#             'CNN_layer_nodes':[4,4,4],
#             'CNN_kernel_size':[5,5,3],
#             'CNN_pool':[False,False,False],
#        },
#    },
#    {'name':'Sep1_CNN_keras_v15','label':'c[4,4,4]_k[5,5,3]_den[16]','pams':{
#             'CNN_layer_nodes':[4,4,4],
#             'CNN_kernel_size':[5,5,3],
#             'CNN_pool':[False,False,False],
#             'Dense_layer_nodes':[16] ,
#        },
#    },
#    {'name':'Sep1_CNN_keras_v16','label':'c[20]_pool','pams':{
#             'CNN_layer_nodes':[20],
#             'CNN_kernel_size':[3],
#             'CNN_pool':[True],
#        },
#    },
    #{'name':'Sep9_CNN_keras_8x8_v1','label':'8x8_c[8]','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v2','label':'8x8_c[2]','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[2],
    #         'CNN_kernel_size':[3],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v3','label':'8x8_c[8]_pool','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v5','label':'8x8_c[4]','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[4],
    #         'CNN_kernel_size':[3],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v6','label':'8x8_c[6]','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[6],
    #         'CNN_kernel_size':[3],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v4','label':'8x8_c[8]_v2','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8_2,'arrMask':arrMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #    },
    #},

    #{'name':'Sep9_CNN_keras_8x8_v7','label':'8x8_c[8]_mask','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'loss':'weightedMSE','maskConvOutput':arrMask,
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v8','label':'8x8_c[6]_mask','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'loss':'weightedMSE','maskConvOutput':arrMask,
    #         'CNN_layer_nodes':[6],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v9','label':'8x8_c[4]_mask','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'loss':'weightedMSE','maskConvOutput':arrMask,
    #         'CNN_layer_nodes':[4],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #    },
    #},

    {'name':'Sep9_CNN_keras_8x8_v10','label':'8x8_c[8]_dup','pams':{
             'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask_full,'calQMask':calQMask,'loss':'weightedMSE',
             'CNN_layer_nodes':[8],
             'CNN_kernel_size':[3],
             'CNN_pool':[False],
        },
    },
    {'name':'Sep9_CNN_keras_8x8_v11','label':'8x8_c[8]_dup_mask','pams':{
             'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask_full,'calQMask':calQMask,'loss':'weightedMSE','maskConvOutput':calQMask,
             'CNN_layer_nodes':[8],
             'CNN_kernel_size':[3],
             'CNN_pool':[False],
        },
    },




    #{'name':'Sep21_CNN_keras_SepConv_v1','label':'SepConv','isDense2D':True,'pams':{
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #    },
    #},
    #{'name':'Sep21_CNN_keras_SepConv_v2','label':'SepConv_NoShareFilter','isDense2D':True,'pams':{
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #         'share_filters'    : False,
    #    },
    #},
    #{'name':'Sep21_CNN_keras_SepConv_v3','label':'SepConv_pool','isDense2D':True,'pams':{
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #    },
    #},


]
for m in models:
   m.update({'isQK':False})
   m.update({'ws':''})
   if not 'isDense2D' in m.keys(): m.update({'isDense2D':False})
   for p,v in defaults.items():
        if not p in m['pams'].keys(): 
            m['pams'].update({p:v})
