#-*- coding: utf-8 -*-
import numpy as np
import itertools


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.affine_relu_forward

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M) D=3072
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """  
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
    
  ''' //  my first code 
  N = x.shape[0]
  D = w.shape[0]
  M = w.shape[1]  
  out = np.zeros((N,M))  
  for i in xrange(N):    
    xi= x[i].ravel()  # reshaep     
    out_i= w.T.dot(xi) # x.dot(w) => (M,)    
    out[i] = out_i + b
  '''  
  N = x.shape[0]
  out = x.reshape(N, np.prod(x.shape[1:])).dot(w) + b    
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################    
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None  
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################

  ''' my first code 
  N = x.shape[0]
  D = w.shape[0]
  M = w.shape[1]

  # db  
  db = np.sum(dout, axis = 0) # (M,)
  
  # dx
  dx = np.zeros((N,D))
  for i in xrange(N):
    xi = x[i] # (D,)
    douti = dout[i]    # (M,)
    dxi = np.zeros((D))
    for j in xrange(M):
        wj = (w.T)[j]        # (D,)
        dxi += (wj*douti[j]) # (D,)
    dx[i] = dxi              # (D,)
  dx = dx.reshape(x.shape)

  # dw 
  dw = np.zeros((D,M))  
  for i in xrange(N):
    xi = x[i].ravel()            # (D,)
    douti = dout[i]              # (M,)
    dwiT = np.zeros((M,D))       # (D, M)
    for j in xrange(M):
        dwiT[j] = (xi*douti[j])  # (D,)
    dwi = dwiT.T
    dw += dwi
  '''
  N = x.shape[0]
  dx = dout.dot(w.T).reshape(x.shape)
  dw = x.reshape(N, np.prod(x.shape[1:])).T.dot(dout)
  db = np.sum(dout, axis=0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape 

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = x*(x>0) # (x.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout*(x>0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var  = bn_param.get('running_var',  np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    # 광현님이 공유한 링크 참조함
    # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    cache = {}
    
    #단계0: 유틸함수 정의
    def f_broadcast (x, shape):
        e = np.tile(x, shape)
        s = shape + np.shape(x)
        return  e.reshape(s)
    
    N, D = x.shape
    
    
    # 단계1:  배치에 포함된 X들의 평균 구하기.
    #    1-1 : reduce sum 구하기 (N,D)=> (D,)
    rs_x = np.sum(x, axis = 0) # rs_x:= reduce sum of x, 
    
    #    1-2 : 1/N 곱하기. (D,)*const_scalar=> (D,)
    mx = rs_x * (1./N)  # mx:= mean of x    
    
    # 단계2: 편차 구하기. (N,D)*(D,)->(N,D)
    #    2.1 broadcast. (D,)=> (N,D)
    bmx = f_broadcast(mx, (N,))
    
    #    2.2 Zero Centered 시키기. (N,D)*(N,D)->(N,D) 
    # 평균 이미지가 zero 이미지가 되게 하기.
    x_m_bmx = x - bmx    # x_m_bmx:= x minius bmx    
    
    # 단계3:  편차 제곱하기. (N,D)=>(N,D)
    sq_x_m_bmx = x_m_bmx ** 2 # sq_x_m_bmx := square of x_m_bmx  
    
    # 단계4: 분산 구하기. ${분산}:= ${편차 제곱}의 평균
    #    4-1 : reduce_sum 구하기. (N,D)=>(D,)
    rs_sq_x_m_bmx = np.sum(sq_x_m_bmx, axis = 0);
    
    #    4-2 : 1/N 곱하기.  (D,)*const_scalar=>(D,)
    var_x = rs_sq_x_m_bmx*(1./N) # var_x := var of x
    
    # 단계5: 입실론 더하기.  (D,)*const_scalar=> (D,)
    var_x_p_e = var_x + eps   # var_x_p_e := var_x plus x
    
    #단계6: 표준편차 구하기.  (D,)=>(D,)
    # 표준편차:= 루트(분산)
    std_dev = np.sqrt(var_x_p_e) # std_dev := standartd deviation    
    
    #단계7: 표준편차로 나눌 준비하기.  (D,)=>(D,)
    # 역수만들기.
    i_std_dev = 1.0/std_dev # i_std_dev := inverse of std_dev
    
    #단계8: 편차를 표준편차로 나누기. (N,D)*(D,)=> (N,D)
    # 즉, 노말라이즈하기. 
    #    8-1 : broadcast  (D,)=>(N,D)
    bi_std_dev = f_broadcast(i_std_dev, (N,))
    
    #   8-2  편차에 표준편차 역수를 곱하기. (N,D)*(N,D)=>(N,D)
    n_x = x_m_bmx*bi_std_dev # n_x:= normalized x.
    
    #단계9: 패러미터값 감마로 Scale 시키기. (D,)*(N,D)=>(N,D)
    #    9-1 broadcast (D,)=>(N,D)
    bgamma = f_broadcast(gamma, (N,))
    
    #   9-2 Scale. (N,D)*(N,D)=>(N,D)
    s_n_x = bgamma * n_x  #  s_n_x : = scaled n_x
    
    #단계10: 패러미터값 베타로 평행이동. (N,D)*(D,)=>(N,D)
    # 배치 노말라이즈 완료.
    #   10-1 broadcast (D,)=>(N,D)
    bbeta = f_broadcast(beta, (N,))
    
    #  10-2 평행이동. (N,D)*(N,D)=>(N,D)
    bn_x = s_n_x + bbeta #  bn_x = batch normalized x    
    
    out= bn_x
    
    #추가 단계.  중간 값 저장.
    running_mean = momentum * running_mean + (1 - momentum) * mx
    running_var  = momentum * running_var  + (1 - momentum) * var_x
   
    # 방만하게 모든 중간 변수 캐시에 저장.
    cache['N'], cache['D'] = N, D
    cache['s_n_x']         = s_n_x
    cache['bbeta']         = bbeta
    cache['beta']          = beta
    cache['n_x']           = n_x
    cache['bgamma']        = bgamma
    cache['gamma']         = gamma
    cache['i_std_dev']     = i_std_dev
    cache['bi_std_dev']    = bi_std_dev
    cache['std_dev']       = std_dev
    cache['var_x']         = var_x
    cache['eps']           = eps
    cache['var_x_p_e']     = var_x_p_e    
    cache['rs_sq_x_m_bmx'] = rs_sq_x_m_bmx    
    cache['sq_x_m_bmx']    = sq_x_m_bmx
    cache['x_m_bmx']       = x_m_bmx
    cache['x']             = x
    cache['bmx']           = bmx
    cache['mx']            = mx
    cache['rs_x']          = rs_x
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    nx = (x-running_mean)/np.sqrt(running_var + eps) # nx := normalized x
    st_nx = gamma* nx + beta # scale and translated nx
    out = st_nx
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var']  = running_var

  return out, cache


def batchnorm_backward_alt(dout, cache):
    """
    Backward pass for batch normalization.
  
    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.
  
    Inputs:
     - dout: Upstream derivatives, of shape (N, D)
     - cache: Variable of intermediates from batchnorm_forward.
  
    Returns a tuple of:
     - dx: Gradient with respect to inputs x, of shape (N, D)
     - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
     - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################
      
    # 단계0. 사전 준비.
    def df_add        (dout, x, y): return dout * 1., dout * 1.           # f x y  = x + y    =>  dx, dy = 1, 1 
    def df_sub        (dout, x, y): return dout * 1., dout * -1.          # f x y  = x - y    =>  dx, dy = 1, -1
    def df_add_const  (dout, x, a): return dout * 1.                      # f x a  = x + a    =>  dx     = 1
    def df_mul        (dout, x, y): return dout * y , dout * x            # f x y  = x * y    =>  dx, dy = y, x
    def df_mul_const  (dout, x, a): return dout * a                       # f a x  = a + x    =>  dx     = a
    def df_square     (dout, x)   : return dout * 2.*x                    # f x    = x^2      =>  dx     = 2 * x
    def df_inv        (dout, x)   : return dout * -1./x**2                # f x    = x^-1     =>  dx     = -1 * (x^-2)
    def df_sqrt       (dout, x)   : return dout * 1./(2.*np.sqrt(x))      # f x    = x^(1/2)   =>  dx     = (1/2) * (x^ (-1/2))
    def df_sum_axis_0 (dout, x)   : return dout *  np.ones_like(x)        # f [x1, x2, x3] = x1+x2+x3  =>  dx = [1,1,1] 
    def df_broadcast  (dout, x, add_shape):                               # f x  (2,3) = [[x,x,x], [x,x,x]]
        for _ in xrange(len(add_shape)):
            dout = np.sum(dout, axis = 0)
        return dout
       
    c = cache
    d = {}
    N,D = c['N'], c['D']
    d['bn_x'] = dout
    
    #단계10: 패러미터값 베타로 평행이동 시키서 배치 노말라이즈 완료. 
    #  10-2 평행이동.     
    #   bn_x = s_n_x + bbeta;                    f x y = x + y
    d['s_n_x'], d['bbeta'] = df_add(d['bn_x'], c['s_n_x'], c['bbeta'])
    
    #  10-1 broadcast
    #   bbeta = f_broadcast(beta, (N,));           f = broadcast    
    d['beta'] = df_broadcast(d['bbeta'], c['beta'], (N,))
    
    #단계9: 패러미터값 감마로 Scale 시키기.    
    #  9-2 Scale. 
    #  s_n_x = bgamma * n_x;                        f x y = x * y    
    d['bgamma'], d['n_x'] = df_mul(d['s_n_x'], c['bgamma'], c['n_x'])
         
    #   9-1 broadcast
    #   bgamma = f_broadcast(gamma, (N,));       f = broadcast    
    d['gamma']            = df_broadcast(d['bgamma'], c['gamma'], (N,))    
    
    #단계8: 편차를 표준편차로 나누기.
    #  8-2  편차에 표준편차 역수를 곱하기.
    #  n_x = x_m_bmx * bi_std_dev;                f x y = x * y        
    d['bi_std_dev'], d['x_m_bmx'] = df_mul(d['n_x'], c['bi_std_dev'], c['x_m_bmx'])
    
    # 8-1 : broadcast 
    #  bi_std_dev = f_broadcast(i_std_dev, N);     f = broadcast    
    d['i_std_dev'] = df_broadcast(d['bi_std_dev'], c['i_std_dev'], (N,))
    
    #단계7: 표준편차로 나눌 준비하기
    # i_std_dev = 1/std_dev;                       f = inverse    
    d['std_dev'] = df_inv(d['i_std_dev'], c['std_dev'])
    
    #단계6: 표준편차 구하기
    # std_dev = np.sqrt(var_x_p_e);                f  = square    
    d['var_x_p_e'] = df_sqrt(d['std_dev'], c['var_x_p_e'])
    
    #단계5: 입실론 더하기.
    # var_x_p_e = var_x + eps;                      f x const = x + const    
    d['var_x'] = df_add_const(d['var_x_p_e'], c['var_x'], c['eps'])
    
    # 단계4: 분산 구하기. 
    #   4-2: 1/N 곱하기.
    #   var_x = rs_sq_x_m_bmx*(1./N);                f x const = x * const    
    d['rs_sq_x_m_mx'] = df_mul_const(d['var_x'], c['rs_sq_x_m_bmx'], 1./N)
    
    #   4-1: reduce_sum 구하기.
    #   rs_sq_x_m_bmx = np.sum(sq_x_m_bmx, axis = 0); f = reduce_sum    
    d['sq_x_m_bmx'] = df_sum_axis_0(d['rs_sq_x_m_mx'], c['sq_x_m_bmx'])

    
    # 단계3:  편차 제곱하기. 
    # sq_x_m_bmx = x_m_bmx ** 2 
    # f(x) = x^2
    d['x_m_bmx'] += df_square(d['sq_x_m_bmx'], c['x_m_bmx']) # 이전에 x_m_bmx 를 구했어서 합치기.  
    
    # 단계2: 편차 구하기. 
    #    2-2 Zero Centered 시키기.
    #    x_m_bmx = x - bmx;                    f x y = x - y   
    d['x'], d['bmx'] = df_sub(d['x_m_bmx'], c['x'], c['bmx'])
    
    #    2-1 broadcast.
    #    bmx = f_broadcast(mx, (N,))             f = broadcast
    d['mx'] = df_broadcast(d['bmx'], c['mx'], (N,))
    
    # 단계1:  배치에 포함된 X들의 평균 구하기.
    #     1-2 : 1/N 곱하기.
    #     mx = rs_x * (1./N);                     f x const = x * const
    d['rs_x'] = df_mul_const(d['mx'], c['rs_x'], 1./N)
    
    #     1-1 : reduce sum 구하기 
    #     rs_x = np.sum(x, axis = 0)     
    d['x'] += df_sum_axis_0(d['rs_x'] , c['x'])  # 이전에 dx 를 구했어서 합치기.     
    
    dx = d['x'];
    dgamma = d['gamma']
    dbeta  = d['beta']
        
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dgamma, dbeta


def batchnorm_backward(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  # 손으로 미분한 결과 및 코드는 아래 링크 참조.
  # http://cthorey.github.io./backpropagation/
  c = cache
  N = c['N']
  
  #dx 는 위 링크에서 코드 카피후 변수 조정.
  dx = (1. / N) * c['gamma'] * c['i_std_dev'] * (N * dout - np.sum(dout, axis=0)
    - c['x_m_bmx'] * c['var_x_p_e']**(-1.0) * np.sum(dout * c['x_m_bmx'], axis=0))

  # dgamma 와 dbeta 는 alt 버전도 기본 버전과 동일함. 기본 버전에서 코드 카피함.
  d = {}
  d['bn_x'] = dout
  def df_add        (dout, x, y): return dout * 1., dout * 1
  def df_mul        (dout, x, y): return dout * y , dout * x 
  def df_broadcast  (dout, x, add_shape):
    for _ in xrange(len(add_shape)):
        dout = np.sum(dout, axis = 0)
    return dout
  d['s_n_x'], d['bbeta'] = df_add(d['bn_x'], c['s_n_x'], c['bbeta'])
  dbeta = df_broadcast(d['bbeta'], c['beta'], (N,))
  d['bgamma'], d['n_x'] = df_mul(d['s_n_x'], c['bgamma'], c['n_x'])
  dgamma  = df_broadcast(d['bgamma'], c['gamma'], (N,))  
  
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################    
    mask      = (np.random.rand(*x.shape) >= p) 
    filtered  = x * mask  
    out       = filtered / p  # invert
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx =  mask * dout
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  # see: https://cloud.githubusercontent.com/assets/1628848/20039453/8e39ae52-a487-11e6-9776-f1d306054535.png
  pad, stride = conv_param['pad'], conv_param['stride']
  N, C, H,  W  = x.shape
  F, _, HH, WW = w.shape
  Hp, Wp       = 1 + (H + 2 * pad - HH) / stride, 1 + (W + 2 * pad - WW) / stride

  # x -> np.pad :: (N, C, H, W) -> (N, C, pad + H + pad, pad + W + pad)
  x = np.pad(x, pad_width=((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
  
  # x w b -> conv_one_filter :: (C,H,W) (C, HH, WW) () -> (Hp, Wp)
  def conv_one_filter (x, w, b):
    out = np.zeros((Hp, Wp))
    for hp, wp, c, hh, ww in itertools.product(xrange(Hp), xrange(Wp), xrange(C), xrange(HH), xrange(WW)):
      h_, w_ = hp*stride + hh, wp*stride + ww
      out[hp][wp] += x[c][h_][w_] * w[c][hh][ww]
    return out + b

  # x -> conv_all_filters :: (C,H,W) -> (F, Hp, Wp)
  def conv_all_filters (x): return [conv_one_filter(x, wi, bi) for wi,bi in zip(w,b)]
      
  # x -> conv_batch :: (N, C, H, W) -> (N, F, Hp, Wp)
  def conv_batch(x) : return [conv_all_filters(xi) for xi in x]
  out = np.array(conv_batch(x))


  cp = conv_param;
  cp['N'], cp['C'],  cp['H'],   cp['W']            = N, C, H, W
  cp['F'], cp['HH'], cp['WW'], cp['Hp'], cp['Wp']  = F, HH, WW, Hp, Wp
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w,  b,  cp               = cache
  N, C,  H,  W,  pad, stride  = cp['N'], cp['C'],  cp['H'],  cp['W'],  cp['pad'], cp['stride']
  F, HH, WW, Hp, Wp           = cp['F'], cp['HH'], cp['WW'], cp['Hp'], cp['Wp']
    
  def zeros_like (*xs)   : return tuple([np.zeros_like(x) for x in list(xs)])
  def df_mul (dout, x, y): return dout * y , dout * x     # f x y  = x * y    =>  dx, dy = y, x

  """
  def conv_one_filter (x, w, b):
    out = np.zeros((Hp, Wp))
    for hp, wp, c, hh, ww in itertools.product(xrange(Hp), xrange(Wp), xrange(C), xrange(HH), xrange(WW)):
      h_, w_ = hp*stride + hh, wp*stride + ww
      out[hp][wp] += x[c][h_][w_] * w[c][hh][ww]
    return out + b
  """
  def df_conv_one_filter(dout, x, w, b):
    dx, dw, db = zeros_like(x, w, b)
    db += np.sum(dout)
    for hp, wp, c, hh, ww in itertools.product(xrange(Hp), xrange(Wp), xrange(C), xrange(HH), xrange(WW)):
      h_, w_ = hp*stride + hh, wp*stride + ww  
      r = df_mul(dout[hp][wp], x[c][h_][w_], w[c][hh][ww])
      dx[c][h_][w_] += r[0]
      dw[c][hh][ww] += r[1]
    return dx, dw, db

  """
  def conv_all_filters (x, w, b): 
    out = np.zeros_like((F, Hp, Wp))
    for i in xrange(F):
      out[i] = conv_one_filter(x, w[i], b[i])
  """
  def df_conv_all_filters(dout, x, w, b):
    dx, dw, db = zeros_like(x, w, b)
    for i in xrange(F):
      r      = df_conv_one_filter(dout[i], x, w[i], b[i])
      dx    += r[0]
      dw[i] += r[1]
      db[i] += r[2] 
    return dx, dw, db

  """    
  def conv_batch(x, w, b):
    out = np.zeros((N,F,Hp,Wp))
    for i in xrange(N):
      out[i] = conv_all_filters(x[i], w, b)
  """
  def df_conv_batch(dout, x, w, b):
    dx, dw, db = zeros_like(x, w, b)
    for i in xrange(N):
      r      = df_conv_all_filters(dout[i], x[i], w, b)
      dx[i] += r[0]
      dw    += r[1]
      db    += r[2]
    return dx, dw, db

  dx, dw, db = df_conv_batch(dout, x, w, b)
  dx = dx[:, :, pad:-pad, pad:-pad]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  # seee https://cloud.githubusercontent.com/assets/1628848/20079079/7a074228-a586-11e6-8144-d529efc1aa7b.png
  N, C, H, W     = x.shape
  PH, PW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  OH, OW         = 1 + (H - PH) / stride, 1 + (W - PW) / stride
       
  def id(x):            return x
  
  def argmax(x, h, w):
    range  = x[h:h+PH, w:w+PW]
    idx    = np.unravel_index(np.argmax(range), (PH,PW))
    return  h + idx[0], w + idx[1]

  # x -> pooling :: (C, H,W) -> (C, OH, OW)
  def pooling (x):
    out = np.zeros((C, OH, OW))
    for c, oh, ow, in itertools.product(xrange(C), xrange(OH), xrange(OW)):      
      h, w           = argmax(x[c], oh*stride, ow*stride)
      out[c][oh][ow] = id(x[c][h][w])
    return out     
  
  # x -> pooling_batch :: (N, C, H, W) -> (N, C, OH, OW)
  def pooling_batch(x) : return [pooling(xi) for xi in x]
  out = np.array(pooling_batch(x))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pp = cache
  N, C, H, W     = x.shape
  PH, PW, stride = pp['pool_height'], pp['pool_width'], pp['stride']
  OH, OW         = 1 + (H - PH) / stride, 1 + (W - PW) / stride
  
  def argmax(x, h, w):
    range  = x[h:h+PH, w:w+PW]
    idx    = np.unravel_index(np.argmax(range), (PH,PW))
    return  h + idx[0], w + idx[1]

  """
  def id(x):           return x
  """
  def df_id(dout, x):  return 1. * dout
  
  """
  def pooling (x):
    out = np.zeros((C, OH, OW))
    for c, oh, ow, in itertools.product(xrange(C), xrange(OH), xrange(OW)):      
      h, w           = argmax(x[c], oh*stride, ow*stride)
      out[c][oh][ow] = id(x[c][h][w])
    return out   
  """
  def df_pooling(dout, x):    
    dx = np.zeros_like(x)
    for c, oh, ow, in itertools.product(xrange(C), xrange(OH), xrange(OW)):
      h, w           = argmax(x[c], oh*stride, ow*stride)
      dx[c][h][w]   += df_id(dout[c][oh][ow], x[c][h][w])
    return dx

  """
  def pooling_batch(x) : 
    out = np.zeros_like(x)
    for i in xrange(N):
      out[i] = pooling(x[i])
  return out
  """
  def df_pooling_batch(dout, x):
    dx = np.zeros_like(x)
    for i in xrange(N):
      dx[i] += df_pooling(dout[i], x[i])
    return dx
 
  dx = df_pooling_batch(dout, x)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # 힌트: vanilla 버전의 배치 노말라이제이션을 이용해 작성할 수 있다.         #  
  # 5줄 이하면 되야 한다                                                      #
  #############################################################################
  # see https://www.reddit.com/r/cs231n/comments/443y2g/hints_for_a2/  의 reshaping 파트.
  N, C, H, W = x.shape
  #                   from  N, C, H, W
  #                     to  N, H, W, C
  x          =  x.transpose(0, 2, 3, 1).reshape((N*H*W, C))
  out, cache =  batchnorm_forward(x, gamma, beta, bn_param)
                  
  #                               from   N, H, W, C
  #                                 to   N, C, H, W
  out = out.reshape((N,H,W,C)).transpose(0, 3, 1, 2)
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  '''
       C0       C1        C2
    [[u, v]   [[a, b]   [[1, 2]
     [w, x]]   [c, d]]   [3, 4]]
     
     [u1, v1, w1, x1, a1, b1, c1, d1, 11, 21, 31, 41]
     [u2, v2, w2, x2, a2, b2, c2, d2, 12, 22, 32, 42]
   
   
        C0  C1 C2
    => [[u, a, 1]
        [v, b, 2]
        [w, c, 3]
        [x, d, 4]]
        
  '''
  N, C, H, W = dout.shape
  #                           from   N, C, H, W
  #                             to   N, H, W, C 
  dout              = dout.transpose(0, 2, 3, 1).reshape((N*H*W, C))
  dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache) # cache : x, gamma, beta
  #                               from   N, H, W, C                       
  #                               from   N, C, H, W                       
  dx = dx.reshape(N, H, W, C).transpose((0, 3, 1, 2))
                  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
