# -*- coding: utf-8 -*-
import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
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
  
  #minibatch_size = x.shape[0]
  #dimension = w.shape[0]
  
  #x.reshape(minibatch_size, dimension)
  #out = np.matmul(x.reshape(x.shape[0], np.prod(x.shape[1:])), w) + b
  out = np.matmul(x.reshape(x.shape[0], w.shape[0]), w) + b
  
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
  
  #minibatch_size = x.shape[0]
  #dimension = w.shape[0]

  db = np.sum(dout, axis=0)
  dw = np.matmul(np.transpose(x.reshape(x.shape[0], w.shape[0])), dout)
  dx = np.matmul(dout, np.transpose(w))
  dx = dx.reshape(x.shape)

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
  
  out = np.maximum(0, x)
  
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
  
  dx = dout*(cache > 0)
  
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
  
  #print x.shape
  N, D = x.shape

  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

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
    
    cache = {}
    
    # batch mean, variance
    x_mean = np.mean(x, axis=0)
    x_var = np.var(x, axis=0)
    
    # normalization
    x_hat = (x - x_mean) / np.sqrt(x_var + eps)
    
    # re-transformation
    out = gamma*x_hat + beta
    
    # running mean, variance
    running_mean = momentum*running_mean + (1 - momentum)*x_mean
    running_var = momentum*running_var + (1 - momentum)*x_var
    
    cache = (x, x_mean, x_var, x_hat, gamma, eps)
    
    '''
    cache['x'] = x
    cache['x_hat'] = x_hat
    cache['gamma'] = gamma
    cache['x_mean'] = x_mean
    cache['x_var'] = x_var
    cache['eps'] = eps
        
    ## batch 평균 : x_mean
    x_mean = np.mean(x, axis=0)
    
    ## 평균 0으로 변환
    x_zero_shift = x - x_mean
    x_zero_shift_2 = np.square(x_zero_shift)
    
    ## batch 분산
    x_var = np.mean(x_zero_shift_2, axis=0)
        
    ## batch 표준편차
    x_std = np.sqrt(x_var + eps)
    x_std_inv = 1. / (x_std)
    
    ## 표준편차 1로 변환
    x_normal = x_zero_shift*x_std_inv
    
    ## gamma scale
    x_rescale = gamma*x_normal
    
    ## beta translation
    out = x_rescale + beta
    
    ## 평균, 표준편차에 대한 이동평균
    running_mean = momentum*running_mean + (1. - momentum)*x_mean
    running_var = momentum*running_var + (1. - momentum)*x_var
    
    cache = (x_mean, x_zero_shift, x_zero_shift_2, x_var, x_std, x_std_inv, x_normal, x_rescale, gamma, beta, x, bn_param)
    '''
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
        
    out = gamma*((x - running_mean) / np.sqrt(running_var + eps)) + beta
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
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
  
  x, x_mean, x_var, x_hat, gamma, eps = cache
  N = float(dout.shape[0])  

  ## out = gamma*x_hat + beta
  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(dout*x_hat, axis=0)
  dx_hat = dout*gamma
  
  dx_var = np.sum((-0.5)*dx_hat*(x - x_mean)*(x_var + eps)**(-1.5), axis=0)
  dx_mean = np.sum(-1*dx_hat / np.sqrt(x_var + eps), axis=0) + dx_var*np.sum(-2*(x - x_mean)) / N
  
  dx = dx_hat/np.sqrt(x_var + eps) + dx_var * 2.0*(x - x_mean) / N + dx_mean / N  
  '''
  x_mean, x_zero_shift, x_zero_shift_2, x_var, x_std, x_std_inv, x_normal, x_rescale, gamma, beta, x, bn_param = cache
    
  N, D = dout.shape
  eps = bn_param.get('eps', 1e-5)
  
  ## dout = gamma*x_normal + beta
  dbeta = np.sum(dout, axis=0)
    
  dgamma = np.sum(dout*x_normal, axis=0)
  dx_normal = dout*gamma

  
  dx_std_inv = np.sum(dx_normal*x_zero_shift, axis=0)
  dx_zero_shift = dx_normal*x_std_inv

  dx_std = -1. /(x_std**2) * dx_std_inv

  dx_var = 0.5 * 1. /np.sqrt(x_var + eps)*dx_std

  dsq = 1. /N * np.ones((N,D))*dx_var

  dx_scale = 2*x_zero_shift*dsq

  dx1 = (dx_zero_shift + dx_scale)
  dx_mean = -1*np.sum(dx_zero_shift + dx_scale, axis=0)

  dx2 = 1. /N*np.ones((N,D)) * dx_mean

  dx = dx1 + dx2
  '''  

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
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
  x, x_mean, x_var, x_hat, gamma, eps = cache
  N = float(dout.shape[0])  
  #x_mean, x_zero_shift, x_zero_shift_2, x_var, x_std, x_std_inv, x_normal, x_rescale, gamma, beta, x, bn_param = cache
  #N, D = dout.shape
  #eps = bn_param.get('eps', 1e-5)
  
  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(dout*x_hat, axis=0)
  
  dx = (1/N)*gamma*(x_var + eps)**(-0.5)*(N*dout - np.sum(dout, axis=0)
    - (x - x_mean)*(x_var + eps)**(-1.0)*np.sum(dout*(x - x_mean), axis=0))

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
    #mask = (np.random.rand(x.shape[0], x.shape[1]) > p) / p
    # x.shape이 tuple이라서 *x.shape으로 써야함
    mask = (np.random.rand(*x.shape) > p) / p
    
    #mask = np.random.rand(*x.shape) > p
    #mask /= p 로 하게되면 mask data type은 bool이고 p는 float이기 때문에
    #mask를 float으로 직접 변환이 되지 않으므로
    #mask = mask / p로 써주어야 새로운 곳에 output memory가 할당됨
    #data type만 동일하다면 mask /= p가 속도가 더 빠름
    out = x*mask
    
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    
    mask = None
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
    
    dx = dout*mask
    
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
  all C channels and has height HH and width HH.

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
  
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  pad, stride = conv_param['pad'], conv_param['stride']
  
  # output size
  conv_H = 1 + (H + 2*pad - HH)/stride
  conv_W = 1 + (W + 2*pad - WW)/stride
  # zero padding : pad_width = pad , 2차원에서 0부터 pad만큼 0으로 붙임
  x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values=0)
  out = np.zeros([N, F, conv_H, conv_W])
  
  for datapt_idx in xrange(N):      # data point
    for filter_idx in xrange(F):    # the number of kernels
      for hpos in xrange(conv_H):   # kernel height
        for wpos in xrange(conv_W): # kernel width
          startw = wpos * stride    # update horizontal positions with stride
          starth = hpos * stride    # update vertical positions with stride
          subarray = x_pad[datapt_idx, :, starth:starth+HH, startw:startw+WW]    # get subimage from padded image
          subfilter = w[filter_idx, :, :]                                        # convolutional kernel
          dotproduct = subarray.flatten().dot(subfilter.flatten())               # their dot product i.e., Convolution
          out[(datapt_idx, filter_idx, hpos, wpos)] = dotproduct + b[filter_idx] # add bias term

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
  
  ##############==by snowyunee==###########
  #Testing conv_backward_fast:
  #Naive: 0.026229s
  #Fast: 0.006315s
  #Speedup: 4.153434x
  #dx difference:  1.28920334946e-11
  #dw difference:  1.76995772764e-13
  #db difference:  7.95337196839e-15
  #########################################
  x, w, b, conv_param = cache
  
  def zero_pad(x, width):
    #N, C, H, W
    return np.lib.pad(x, ((0,), (0,), (width,), (width,)), 'constant', constant_values=(0,))

  stride, pad = conv_param['stride'], conv_param['pad']
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  x_padded = zero_pad(x, pad)

  dx = np.zeros_like(x_padded)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  _, _, HP, WP = x_padded.shape
  
  w2 = np.transpose(w.reshape(F, -1))
  for hi, hp in enumerate(range(HP - HH + 1)[::stride]):
    for wi, wp in enumerate(range(WP - WW + 1)[::stride]):
      dout2 = dout[:, :, hi, wi]
      x2 = x_padded[:, :, hp : hp + HH, wp : wp + WW]
      dx2, dw2, db2 = affine_backward(dout2, (x2, w2, b))
      dx[:, :, hp : hp + HH, wp : wp + WW] += dx2
      dw += dw2.T.reshape(*dw.shape)
      db += db2.reshape(*db.shape)

  dx = dx[:, :, pad:-pad, pad:-pad]
      
  ##############==Origin==#################
  # Testing conv_backward_fast:
  #Naive: 2674.766461s
  #Fast: 0.006588s
  #Speedup: 406006.936378x
  #dx difference:  1.43102991543e-11
  #dw difference:  4.8324564186e-12
  #db difference:  2.64046604536e-15
  #########################################
  '''
  x, w, b, conv_param = cache
  P, S = conv_param['pad'] ,conv_param['stride']
  # zero padding
  x_pad = np.pad(x, ((0,), (0,), (P,), (P,)), 'constant')

  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  N, F, Hh, Hw = dout.shape
      
  dw = np.zeros((F, C, HH, WW))
  
  for fprime in range(F):        # the number of kernels
    for cprime in range(C):      # the number of channels
      for i in range(HH):        # kernel height
        for j in range(WW):      # kernel width
          sub_xpad = x_pad[:, cprime, i:i + Hh * S:S, j:j + Hw * S:S]          # out = X*W + b  >  dout = d(X*W) + db > dW = dout * X + 0
          dw[fprime, cprime, i, j] = np.sum(dout[:, fprime, :, :] * sub_xpad)

  db = np.zeros((F))
  
  for fprime in range(F):
    db[fprime] = np.sum(dout[:, fprime, :, :])   # dout = d(X*W) + db > db = 0 + dout

  dx = np.zeros((N, C, H, W))
  
  for nprime in range(N):       # data point
    for i in range(H):          # image height
      for j in range(W):        # image width
        for f in range(F):      # the number of kernels
          for k in range(Hh):   # conv_out height
            for l in range(Hw): # conv_out width
              mask1 = np.zeros_like(w[f, :, :, :])
              mask2 = np.zeros_like(w[f, :, :, :])
              if (i + P - k * S) < HH and (i + P - k * S) >= 0:
                mask1[:, i + P - k * S, :] = 1.0
              if (j + P - l * S) < WW and (j + P - l * S) >= 0:
                mask2[:, :, j + P - l * S] = 1.0
                w_masked = np.sum(w[f, :, :, :] * mask1 * mask2, axis=(1, 2))
                dx[nprime, :, i, j] += dout[nprime, f, k, l] * w_masked       # dX = dout * W + 0
  '''
  

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
  N, C, H, W = x.shape
  pool_height, pool_width, stride = (pool_param['pool_height'], pool_param['pool_width'], pool_param['stride'])
  # output size
  H_prime = 1 + (H - pool_height) / stride
  W_prime = 1 + (W - pool_width) / stride

  out = np.zeros([N, C, H_prime, W_prime])

  for datapt_idx in xrange(N):
    for c_idx in xrange(C):
      for hpos in xrange(H_prime):
        for wpos in xrange(W_prime):
          startw = wpos * stride
          starth = hpos * stride
          subarray = x[datapt_idx, c_idx, starth:starth+pool_height, startw:startw+pool_width]  # get subimage from pool size
          out[(datapt_idx, c_idx, hpos, wpos)] = subarray.max()                                 # replace its maximum
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
  x, pool_param = cache
  Hp = pool_param['pool_height']
  Wp = pool_param['pool_width']
  S = pool_param['stride']
  N, C, H, W = x.shape
  H1 = (H - Hp) / S + 1
  W1 = (W - Wp) / S + 1

  dx = np.zeros((N, C, H, W))
  for nprime in range(N):
    for cprime in range(C):
      for k in range(H1):
        for l in range(W1):
          x_pooling = x[nprime, cprime, k * S:k * S + Hp, l * S:l * S + Wp]
          maxi = np.max(x_pooling)
          x_mask = x_pooling == maxi
          dx[nprime, cprime, k * S:k * S + Hp, l * S:l * S + Wp] += dout[nprime, cprime, k, l] * x_mask
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
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  
  N, C, H, W = x.shape
  bn_shape = N*H*W
  
  bn_fwd = x.transpose(0, 2, 3, 1).reshape(bn_shape, C)
  out, cache = batchnorm_forward(bn_fwd, gamma, beta, bn_param)
  
  out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2) 

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
  
  N, C, H, W = dout.shape
  bn_shape = N*H*W
    
  bn_bwd = dout.transpose(0, 2, 3, 1).reshape(bn_shape, C)
  dx, dgamma, dbeta = batchnorm_backward(bn_bwd, cache)
  
  dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  
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
