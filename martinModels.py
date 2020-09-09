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
    {'name':'Sep1_CNN_keras_v16','label':'c[20]_pool','pams':{
             'CNN_layer_nodes':[20],
             'CNN_kernel_size':[3],
             'CNN_pool':[True],
        },
    },

]
for m in models:
   m.update({'isQK':False})
   m.update({'ws':''})
   for p,v in defaults.items():
        if not p in m['pams'].keys(): 
            m['pams'].update({p:v})
