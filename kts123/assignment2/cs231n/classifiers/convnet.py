#-*- coding: utf-8 -*-
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class Simple_VGG_16(object):
    """
      * 네트워크 구조.
        [conv-relu-conv-relu-pool]x3 -[conv-relu-conv-relu-pool]x2 - [affine]x3 - [softmax] 
        VGG의 D타입과 동일하게 총 15개의 히든 레이어로 구성됨.
        - conv 다음에는 무조건 spatial batch normalization 사용.
        - simple-VGG 이므로 conv 와 pool 은 모두 다음과 같음.
        - conv 필터 크기는  3x3. stride 는 1, padding은 1
        - 풀링 필터 크기는  2x2, stride 는 2, padding은 없음.

        
      ===========================================================================================
      |           :     피처맵      |        필터           | 필터(스케일 표현 )  |  패러미터   |
      |  레이어   :                 |  현재,이전            | 현재,이전           |  갯수       |
      |           : [채널,높이,넓이]| [채널,채널,높이,넓이] |[채널,채널,높이,넓이]| (bias 제외) |
      -------------------------------------------------------------------------------------------
      |  INPUT    : [3,   32, 32]   |                       |                     |             |
      ------------------------------|------------------------------------------------------------
      |  CONV3-64 : [64,  32, 32]   | [64,    3, 3, 3]      | [1*S,   3, 3, 3]    |   27*S      |
      |  CONV3-64 : [64,  32, 32]   | [64,   64, 3, 3]      | [1*S, 1*S, 3, 3]    |    9*S^2    |
      |  POOL2    : [64,  16, 16]   |                       |                     |             |
      ------------------------------|------------------------------------------------------------
      |  CONV3-128: [128, 16, 16]   | [128,  64, 3, 3]      | [2*S, 1*S, 3, 3]    |   18*S^2    |
      |  CONV3-128: [128, 16, 16]   | [128, 128, 3, 3]      | [2*S, 2*S, 3, 3]    |   36*S^2    |
      |  POOL2    : [128,  8,  8]   |                       |                     |             |
      ------------------------------|------------------------------------------------------------
      |  CONV3-256: [256,  8,  8]   | [256, 128, 3, 3]      | [4*S, 2*S, 3, 3]    |   72*S^2    |
      |  CONV3-256: [256,  8,  8]   | [256, 256, 3, 3]      | [4*S, 4*S, 3, 3]    |  144*S^2    |
      |  CONV3-256: [256,  8,  8]   | [256, 256, 3, 3]      | [4*S, 4*S, 3, 3]    |  144*S^2    |
      |  POOL2    : [256,  4,  4]   |                       |                     |             |
      ------------------------------|------------------------------------------------------------
      |  CONV3-512: [512,  4,  4]   | [512, 256, 3, 3]      | [8*S, 4*S, 3, 3]    |  288*S^2    |
      |  CONV3-512: [512,  4,  4]   | [512, 512, 3, 3]      | [8*S, 8*S, 3, 3]    |  576*S^2    |
      |  CONV3-512: [512,  4,  4]   | [512, 512, 3, 3]      | [8*S, 8*S, 3, 3]    |  576*S^2    |
      |  POOL2    : [512,  2,  2]   |                       |                     |             |
      ------------------------------|------------------------------------------------------------
      |  CONV3-512: [512,  2,  2]   | [512, 512, 3, 3]      | [8*S, 8*S, 3, 3]    |  576*S^2    |
      |  CONV3-512: [512,  2,  2]   | [512, 512, 3, 3]      | [8*S, 8*S, 3, 3]    |  576*S^2    |
      |  CONV3-512: [512,  2,  2]   | [512, 512, 3, 3]      | [8*S, 8*S, 3, 3]    |  576*S^2    |
      |  POOL2    : [512,  1,  1]   | [512, 512, 3, 3]      |                     |             |
      ------------------------------|------------------------------------------------------------
      |  FC:        [512]           | [512, 512]            | [8*S, 8*S]          |   64*S^2    |
      |  FC:        [512]           | [512, 512]            | [8*S, 8*S]          |   64*S^2    |
      ------------------------------|------------------------------------------------------------
      |  FC:        [10]            | [ 10, 512]            | [10,  8*S]          |   80*S      |
      ===========================================================================================
 
    위에서 얘기했듯이 대부분 하드 코딩하고 오직 S 만 입력으로 받게 함.
    cnn.py 코드 그대로 복사해 옴.
    """    
    def __init__(self, 
                 scale_factor  = 1, # 위의 표에서 S를 의미함.
                 weight_scale=1e-3, reg=0.0):
        print 'debug 0'
        self.params = {}
        self.reg    = reg
        
        self.conv_param = {'stride': 1, 'pad': 1}
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}   
    
        def add_layer(layer, F, C, H = 1, W = 1, layer_type = 'conv'):
            filter_size =  (F, C, H, W) if layer_type == 'conv' else  (C, F)         
            self.params['W' + `layer`] = np.random.normal(scale = weight_scale,  size = filter_size).astype(np.float32)
            self.params['b' + `layer`] = np.zeros(F).astype(np.float32)
       
        S = scale_factor
        # conv layer
        add_layer(1, 1*S,   3, 3, 3);  # CONV3-64 
        add_layer(2, 1*S, 1*S, 3, 3);  # CONV3-64
           
        add_layer(3, 2*S, 1*S, 3, 3);  # CONV3-128
        add_layer(4, 2*S, 2*S, 3, 3);  # CONV3-128
      
        add_layer(5, 4*S, 2*S, 3, 3);  # CONV3-256
        add_layer(6, 4*S, 4*S, 3, 3);  # CONV3-256
        add_layer(7, 4*S, 4*S, 3, 3);  # CONV3-256
        
        add_layer(8,  8*S, 4*S, 3, 3); # CONV3-512
        add_layer(9,  8*S, 8*S, 3, 3); # CONV3-512
        add_layer(10, 8*S, 8*S, 3, 3); # CONV3-512

        add_layer(11, 8*S, 8*S, 3, 3);  # CONV3-512
        add_layer(12, 8*S, 8*S, 3, 3);  # CONV3-512    
        add_layer(13, 8*S, 8*S, 3, 3);  # CONV3-512    

        # fc layer
        add_layer(14, 8*S, 8*S, layer_type = 'fc') # FC-512
        add_layer(15, 8*S, 8*S, layer_type = 'fc') # FC-512
    
        # score layer
        add_layer(16, 10,  8*S, layer_type = 'score')  # FC-10    

        
    def loss(self, X, y=None):
        ############################################################################
        # the forward pass                                                         #
        ############################################################################
        backward_func = {}
        
        def do_(layer, X, forward, backward):
            Wi, bi   = 'W'+`layer`,  'b'+`layer`
            Y, cache = forward (X, self.params[Wi],  self.params[bi])
            
            def do_back (dY, d):
                dX, d[Wi], d[bi] = backward(dY, cache) # cache = Wi, bi, ..
                return dX            
            backward_func[layer] = do_back
            
            return Y
        
        conv_relu_forward_p      = lambda X, W, b : conv_relu_forward     (X, W, b, self.conv_param)
        conv_relu_pool_forward_p = lambda X, W, b : conv_relu_pool_forward(X, W, b, self.conv_param, self.pool_param)
        
        conv_relu       = lambda layer, X: do_(layer, X, conv_relu_forward_p,      conv_relu_backward)
        conv_relu_pool  = lambda layer, X: do_(layer, X, conv_relu_pool_forward_p, conv_relu_pool_backward)
        affine_relu     = lambda layer, X: do_(layer, X, affine_relu_forward,      affine_relu_backward) 
        affine          = lambda layer, X: do_(layer, X, affine_forward,           affine_backward) 

        # conv layer
        X = conv_relu      (1,  X)
        X = conv_relu_pool (2,  X)

        X = conv_relu      (3,  X)
        X = conv_relu_pool (4,  X)
    
        X = conv_relu      (5,  X)
        X = conv_relu      (6,  X)
        X = conv_relu_pool (7,  X)
       
        X = conv_relu      (8,  X)
        X = conv_relu      (9,  X)
        X = conv_relu_pool (10, X)
        
        X = conv_relu      (11, X)
        X = conv_relu      (12, X)
        X = conv_relu_pool (13, X)
       
        # fc layer
        X = affine_relu    (14, X)
        X = affine_relu    (15, X)
       
        # score layer
        X = affine         (16, X)

        if y is None:
            return X
  
        ############################################################################
        # the loss & backward pass                                                        #
        ############################################################################
        dX = None # gradient of data 
        d  = {}   # gradient of W and b
         
        # loss, 
        loss, dX = softmax_loss(X, y)
        
        # gradient
        for layer in reversed(xrange(1, 17)):
            dX = backward_func[layer](dX, d)

        for layer in xrange(1,17):
            Wi =  self.params['W'+`layer`]
            loss           += 0.5 * self.reg * np.sum(Wi ** 2)  
            d['W'+`layer`] +=       self.reg * Wi
        
        return loss, d
        