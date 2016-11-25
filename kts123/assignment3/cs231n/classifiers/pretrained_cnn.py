#-*- coding: utf-8 -*-

import numpy as np
import h5py

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class PretrainedCNN(object):
    def __init__(self, dtype=np.float32, num_classes=100, input_size=64, h5_file=None):
        self.dtype       = dtype
        self.conv_params = []
        self.input_size  = input_size
        self.num_classes = num_classes
    
        # TODO: 나중에, HDF5 파일에서 아키텍쳐를 로드하게 되면 멋져 질 것이다.
        # 지금은 아래처럼 하드코딩 한다. 
        # For now this will have to do. <= 동구님께 해석 부탁
        self.conv_params.append({'stride': 2, 'pad': 2})
        self.conv_params.append({'stride': 1, 'pad': 1})
        self.conv_params.append({'stride': 2, 'pad': 1})
        self.conv_params.append({'stride': 1, 'pad': 1})
        self.conv_params.append({'stride': 2, 'pad': 1})
        self.conv_params.append({'stride': 1, 'pad': 1})
        self.conv_params.append({'stride': 2, 'pad': 1})
        self.conv_params.append({'stride': 1, 'pad': 1})
        self.conv_params.append({'stride': 2, 'pad': 1})

        self.filter_sizes = [5, 3, 3, 3, 3, 3, 3, 3, 3]
        self.num_filters  = [64, 64, 128, 128, 256, 256, 512, 512, 1024]
        hidden_dim        = 512

        self.bn_params = []
    
        cur_size    = input_size
        prev_dim    = 3
        self.params = {}
        for i, (f, next_dim) in enumerate(zip(self.filter_sizes, self.num_filters)):
            fan_in                           = f * f * prev_dim
            self.params['W%d' % (i + 1)]     = np.sqrt(2.0 / fan_in) * np.random.randn(next_dim, prev_dim, f, f)
            self.params['b%d' % (i + 1)]     = np.zeros(next_dim)
            self.params['gamma%d' % (i + 1)] = np.ones(next_dim)
            self.params['beta%d' % (i + 1)]  = np.zeros(next_dim)
            self.bn_params.append({'mode': 'train'})
            prev_dim                         = next_dim
            if self.conv_params[i]['stride'] == 2: cur_size /= 2
    
        # fully-connected 레이어 추가
        fan_in                           = cur_size * cur_size * self.num_filters[-1]
        self.params['W%d' % (i + 2)]     = np.sqrt(2.0 / fan_in) * np.random.randn(fan_in, hidden_dim)
        self.params['b%d' % (i + 2)]     = np.zeros(hidden_dim)
        self.params['gamma%d' % (i + 2)] = np.ones(hidden_dim)
        self.params['beta%d' % (i + 2)]  = np.zeros(hidden_dim)
        self.bn_params.append({'mode': 'train'})
        self.params['W%d' % (i + 3)]     = np.sqrt(2.0 / hidden_dim) * np.random.randn(hidden_dim, num_classes)
        self.params['b%d' % (i + 3)]     = np.zeros(num_classes)
    
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

        if h5_file is not None:
            self.load_weights(h5_file)
   
        # end of init
        #--------------------------------------------
  
    def load_weights(self, h5_file, verbose=False):
        """
        HDF5 file 에시 기 훈련된 가중치 값 로드
        
        입력:
            - h5_file: HDF5 파일 경로. 기 훈련된 가중치 값이 들어 있다.
            - verbose: 디버깅 정보를 출력할지 결정
         """

        # 가중치 로딩 전에 더미 forward 패스를 이용해 bn_params의 runnng 평균값을 초기화 해야 한다.
        x = np.random.randn(1, 3, self.input_size, self.input_size)
        y = np.random.randint(self.num_classes, size=1)
        loss, grads = self.loss(x, y)

        with h5py.File(h5_file, 'r') as f:
            for k, v in f.iteritems():
                v = np.asarray(v)
                if k in self.params:
                    if verbose: print k, v.shape, self.params[k].shape
                    if v.shape == self.params[k].shape:
                        self.params[k] = v.copy()
                    elif v.T.shape == self.params[k].shape:
                        self.params[k] = v.T.copy()
                    else:
                        raise ValueError('shapes for %s do not match' % k)
                if k.startswith('running_mean'):
                    i = int(k[12:]) - 1
                    assert self.bn_params[i]['running_mean'].shape == v.shape
                    self.bn_params[i]['running_mean'] = v.copy()
                    if verbose: print k, v.shape
                if k.startswith('running_var'):
                    i = int(k[11:]) - 1
                    assert v.shape == self.bn_params[i]['running_var'].shape
                    self.bn_params[i]['running_var'] = v.copy()
                    if verbose: print k, v.shape
        
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(self.dtype)

  
    def forward(self, X, start=None, end=None, mode='test'):
        """
        모델의 일부분에 대해 forward 계산을 수행한다.
        start와 end는 레이어를 뜻한다.
        훈련모드와 테스트 모드 둘 다 사용하는 함수이다.
     
        X는 start 레이어의 입력을 넣는 것이고
        그러면 이 함수는 end 레이어를 출력한다.
        cache도 출력한다.
        이 cache는 동일한 (start, end)레이어로 backward를 계산할 때 이용한다.

        이 함수의 목적에 맞게 "레이어"는 아래와 같이 생겼다.

        [conv - spatial batchnorm - relu] (9개가 있다)
        [affine - batchnorm - relu]       (1개가 있다)
        [affine]                          (1개가 있다)

        입력:
            - X    : start 레이어를 나타내는 입력. 
                     start=0 인 레이어의 경우 입력은 원본 그림 자체여야 하므로 셰이프는 (N, C, 64, 64)이다.
            - start: 시작   레이어 인덱스. 
                     None인 경우 기본값  0 으로 설정.  
                     0 인덱스는 원본 이미지를 입력받는 첫번째 convolution layer 를 뜻함.
            - end  : 마지막 레이어 인덱스. 
                     None인 경우 기본값 10 으로 설정.  (영어 원문에는 10이 아니라 11로 잘못 적혀 있음)
                     10 인덱스는 클래스 스코어를 리턴하는 FC 레이어를 뜻함.
            - mode : 사용할 모드. 'test'나 'train' 값이 가능
                     모드에 따라 배치 노말라이제이션이 달라야 되서 모드 지정이 필요함.

        출력:
            - out  : end 레이어의 출력값.
            - cache: 동일한 (start, end)레이어로 backward를 계산할 때 이용할 캐시값.
        """
        X = X.astype(self.dtype)
        if start is None: start = 0
        if end   is None: end   = len(self.conv_params) + 1
        layer_caches = []
   
        prev_a = X
        for i in xrange(start, end + 1):
            i1 = i + 1
            if 0 <= i < len(self.conv_params):
                # 컨브 레이어
                w, b             = self.params['W%d'     % i1], self.params['b%d'    % i1]
                gamma, beta      = self.params['gamma%d' % i1], self.params['beta%d' % i1]
                conv_param       = self.conv_params[i]
                bn_param         = self.bn_params[i]
                bn_param['mode'] = mode

                next_a, cache = conv_bn_relu_forward(prev_a, w, b, gamma, beta, conv_param, bn_param)
            elif i == len(self.conv_params):
                # 은닉 FC 레이어
                w, b             = self.params['W%d'     % i1], self.params['b%d'    % i1]
                gamma, beta      = self.params['gamma%d' % i1], self.params['beta%d' % i1]
                bn_param         = self.bn_params[i]
                bn_param['mode'] = mode
                next_a, cache = affine_bn_relu_forward(prev_a, w, b, gamma, beta, bn_param)
            elif i == len(self.conv_params) + 1:
                # 마지막 FC 레이어. 스코어 값 계산. 
                w, b             = self.params['W%d' % i1],       self.params['b%d' % i1]
                next_a, cache    = affine_forward(prev_a, w, b)
            else:
                raise ValueError('Invalid layer index %d' % i)

            layer_caches.append(cache)
            prev_a = next_a

        out   = prev_a
        cache = (start, end, layer_caches)
        return out, cache


    def backward(self, dout, cache):
        """
        모델의 일부 레이어에 대해 backward 계산을 수행한다.
        해당 레이어는 self.forward 함수로 이전에 계산한 레이어다. 
        
        입력:
            - dout : 마지막 레이어의 미분값. 마지막 레이어 출력과 같은 셰이프이다.
            - cache: self.forward 에서 리턴한 캐시값

        출력:
        - dX   : start 레이어 입력값에 대한 미분값. self.forward 의 입력 X와 같은 셰이프이다.
        - grads: 모든 레이어의 패러미터들에 대한 미분값.
                 예를 들어 두 개의 컨브 레이어에 대해 수행하는 경우
                 두 레이어 모두에 대해 weights,  biases, spatial batchnorm 다 계산해야 한다.
                 grads 는 self.params 의 부분집합이다. 
                 grads[k] 와 self.params[k] 는 셰이프가 같다.
        """
        start, end, layer_caches = cache
        dnext_a = dout
        grads = {}
        for i in reversed(range(start, end + 1)):
            i1 = i + 1
            if i == len(self.conv_params) + 1:
                # This is the last fully-connected layer
                dprev_a, dw, db = affine_backward(dnext_a, layer_caches.pop())
                grads['W%d' % i1] = dw
                grads['b%d' % i1] = db
            elif i == len(self.conv_params):
                # This is the fully-connected hidden layer
                temp = affine_bn_relu_backward(dnext_a, layer_caches.pop())
                dprev_a, dw, db, dgamma, dbeta = temp
                grads['W%d' % i1]     = dw
                grads['b%d' % i1]     = db
                grads['gamma%d' % i1] = dgamma
                grads['beta%d'  % i1] = dbeta
            elif 0 <= i < len(self.conv_params):
                # This is a conv layer
                temp = conv_bn_relu_backward(dnext_a, layer_caches.pop())
                dprev_a, dw, db, dgamma, dbeta = temp
                grads['W%d' % i1]     = dw
                grads['b%d' % i1]     = db
                grads['gamma%d' % i1] = dgamma
                grads['beta%d'  % i1] = dbeta
            else:
                raise ValueError('Invalid layer index %d' % i)
            dnext_a = dprev_a

        dX = dnext_a
        return dX, grads


    def loss(self, X, y=None):
        """
        네트워크를 훈련시킬 때의 분류 손실값(classificatin loss) 계산

        입력:
            - X (N, 3, 64, 64): 데이터
            - y (N,)          : 라벨
        출력:    
          [y가 None 인 경우 '테스트' 모드 forward 만 하고 score만 리턴]
            - scores: Array of shape (N, 100) giving class scores.
          [y가 None 이 아닌 경우, '훈련' 모드 forward와 backward  수행 후 아래 값도 리턴] 
            - loss  : 손실값. 스칼라
            - grads : 미분값. 딕셔너리. 키는 self.params 에 있는 것이어야 함
        """
        # Note: self.forward 호출과 self.backward 만 호출하면 됨
        mode          = 'test' if y is None else 'train'
        scores, cache = self.forward(X, mode=mode)
        if mode == 'test':
              return scores
        loss, dscores = softmax_loss(scores, y)
        dX, grads     = self.backward(dscores, cache)
        return loss, grads

