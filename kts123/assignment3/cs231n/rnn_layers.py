#-*- coding: utf-8 -*-

import numpy as np
import itertools


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""

def f_broadcast (x, shape):
    e = np.tile(x, shape)
    s = shape + np.shape(x)
    return  e.reshape(s)

def df_broadcast (dout, x, add_shape):
    for _ in xrange(len(add_shape)):
        dout = np.sum(dout, axis = 0)
    return dout

def  f_add3(a, b, c)     : return a + b + c
def  df_add3(dout, a,b,c): return (dout, dout, dout)
def  df_mul(dout, a, b)  : return (dout*b, dout*a)
def  df_add(dout, a, b)  : return (dout,   dout)

def  df_tanh(dout, x) :    return (1+np.tanh(x))*(1-np.tanh(x))*dout   # see. http://agiantmind.tistory.com/15
def  df_sigmoid(dout, x):  sx = sigmoid(x); return sx*(1.0-sx)*dout
def  df_tanh2(dout, tx) :  return (1+tx)*(1-tx)*dout   # see. http://agiantmind.tistory.com/15
def  df_sigmoid2(dout, sx):return sx*(1.0-sx)*dout
def  df_dot (dout, x, y):  return (np.dot(dout, y.T), np.dot(x.T, dout))

def zeros_like (*xs)   : return tuple([np.zeros_like(x) for x in list(xs)])

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x:      Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx:     Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh:     Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b:      Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  # cs231n 강의 페이지의 https://gist.github.com/karpathy/d4dee566867f8291f086 코드 40번째 라인 참조.
  # next_h = np.tanh(np.dot(x, Wx) + np.dot(prev_h, Wh) + b)
  N  = x.shape[0]
    
  x1     = np.dot      (x,      Wx)
  x2     = np.dot      (prev_h, Wh)
  bb     = f_broadcast (b,      (N,))
  x3     = f_add3      (x1,     x2, bb)
  next_h = np.tanh     (x3)
  
  cache = (N, x3, x2, x1, x, b, bb, Wx, Wh, prev_h)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx:      Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx:     Gradients of input-to-hidden weights, of shape (N, H)
  - dWh:     Gradients of hidden-to-hidden weights, of shape (H, H)
  - db:      Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  N, x3, x2, x1, x, b, bb, Wx, Wh, prev_h = cache
    
  # next_h = np.tanh(         x3)                                     
  dx3      = df_tanh(dnext_h, x3)
                                     
  #          x3 =   f_add3(     x1, x2, bb)
  dx1, dx2, dbb = df_add3(dx3, x1, x2, bb)
    
  # bb =  f_broadcast(     b, (N,)) 
  db   = df_broadcast(dbb, b, (N,))
  
  #          x2 = np.dot(     prev_h, Wh)
  dprev_h,  dWh = df_dot(dx2, prev_h, Wh)
                                     
  #    x1 = np.dot(     x, Wx)   
  dx, dWx = df_dot(dx1, x, Wx) 
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N, T, D = x.shape
  N, H    = h0.shape
  
  cache = {}
  sc    = {} # sub_cache
    
  h = np.zeros((N,T,H))
  prev_h = h0
  for i in xrange(T):
    h[:, i, :], sc[i] = rnn_step_forward(x[:,i,:], prev_h, Wx, Wh, b)    
    prev_h = h[:, i, :]
  
  cache = (N, T, D, H, x, h, h0, Wx, Wh, b, sc)
  
    
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx:  Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db:  Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  N, T, D, H, x, h, h0, Wx, Wh, b, sc =   cache
  
  ''' forward code
  h = np.zeros((N,T,H))
  prev_h = h0
  for i in xrange(T):
    h[:, i, :], sc[i] = rnn_step_forward(x[:,i,:], prev_h, Wx, Wh, b)  # prev_h = h[:, i-1, :]
    prev_h = h[:, i, :]
  '''
  dx, dWx, dWh, db =  zeros_like(x, Wx, Wh, b)
  dprev_h = np.zeros((N,H))
  for i in reversed(xrange(T)):
    dhi = dh[:,i,:] + dprev_h
    dx[:, i, :] , dprev_h, dWxi, dWhi, dbi = rnn_step_backward(dhi, sc[i])   # dx, dprev_h, dWx, dWh, db 
    dWx += dWxi
    dWh += dWhi
    db  += dbi
  dh0 = dprev_h

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  out   = W[x] # (W.T*onehot_x).T
  cache = (x, W)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  x, W = cache
  dW = np.zeros_like(W)
  np.add.at(dW, x, dout)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, 
  the hidden state has dimension H, 
  and we use a minibatch size of N.
  
  Inputs:
      - x      (N, D) : Input data 
      - prev_h (N, H) : Previous hidden state
      - prev_c (N, H) : Previous cell state
      - Wx     (D,4H) : Input-to-hidden weights
      - Wh     (H,4H) : Hidden-to-hidden weights
      - b      (  4H,): Biases, of shape 
  
  Returns a tuple of:
      - next_h (N, H) : Next hidden state, of shape
      - next_c (N, H) : Next cell state, of shape 
      - cache         : Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  N, _ = x.shape

  # affined = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  next_hx = np.dot(x,      Wx)         # (N,4H) = (N,D)*(D,4H)
  next_hh = np.dot(prev_h, Wh)         # (N,4H) = (N,H)*(H,4H)
  bb      = f_broadcast(b, (N,))       # (N,4H) 
  affined = next_hx + next_hh + bb     # (N,4H)
  
  # gates
  gs = np.split(affined, 4, axis = 1)  # [(N, H)]
  i = sigmoid(gs[0])                   #  (N, H)
  f = sigmoid(gs[1])                   #  (N, H)
  o = sigmoid(gs[2])                   #  (N, H)
  g = np.tanh(gs[3])                   #  (N, H)
    
  # next_c
  fp = f*prev_c
  ig = i*g
  next_c = fp + ig                     # (N,H)
  
  # next_h
  next_c2 = np.tanh(next_c)
  next_h  = o * next_c2                # (N,H)

  cache = (N, b, bb, x, Wh, Wx,
           next_c, next_c2, next_h, next_hh, next_hx, 
           prev_c, prev_h, i, f, o, g, fp, ig, gs)
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
      - dnext_h (N, H) : Gradients of next hidden state 
      - dnext_c (N, H) : Gradients of next cell state 
      - cache          : Values from the forward pass
  
  Returns a tuple of:
      - dx      (N, D) : Gradient of input data
      - dprev_h (N, H) : Gradient of previous hidden state
      - dprev_c (N, H) : Gradient of previous cell state
      - dWx     (D,4H) : Gradient of input-to-hidden weights
      - dWh     (H,4H) : Gradient of hidden-to-hidden weights
      - db      (  4H,): Gradient of biases
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  (N, b, bb, x, Wh, Wx,              
  next_c, next_c2, next_h, next_hh, next_hx,  
  prev_c, prev_h, i, f, o, g, fp, ig, gs) = cache

                                                 #===================================  
                                                 #     next_h = o * np.tanh(next_c)
                                                 #-----------------------------------  
  do, dnext_c2  = df_mul  (dnext_h, o, next_c2)  # next_h  = f_mul (o, next_c2) 
  dnext_c      += df_tanh2(dnext_c2,   next_c2)  # next_c2 = np.tanh(next_c)  
                                                 #
                                                 #===================================  
                                                 #     next_c = f * prev_c + i * g 
                                                 #-----------------------------------        
  dfp, dig    = df_add(dnext_c, fp, ig)          # next_c = f_add(fp, ig) 
  df, dprev_c = df_mul(dfp, f, prev_c)           # fp     = f_mul(f, prev_c)
  di, dg      = df_mul(dig, i, g)                # ig     = f_mul(i, g)
                                                 #
                                                 #===================================   
                                                 #      i,f,o,g  
                                                 #-----------------------------------  
  dgs0 = df_sigmoid2(di, i)                      # i = sigmoid(gs[0])
  dgs1 = df_sigmoid2(df, f)                      # f = sigmoid(gs[1])
  dgs2 = df_sigmoid2(do, o)                      # o = sigmoid(gs[2])
  dgs3 = df_tanh2   (dg, g)                      # g = tf.tanh(gs[3])
                                                 #
                                                 #===================================  
                                                 #     gs = np.split(affined, 4, axis = 1) 
                                                 #-----------------------------------  
  daffined = np.concatenate((dgs0, dgs1, dgs2, dgs3), axis=1)
                                                 #
                                                 #===================================  
                                                 #    affined =  f_add3(next_hx, next_hh, bb)
                                                 #-----------------------------------
  dnext_hx, dnext_hh, dbb = df_add3(daffined,    # affined = f_add3(
                         next_hx, next_hh, bb)   #             next_hx, next_hh, bb)
  db           = df_broadcast(dbb, b, (N,))      # bb      = f_broadcast(b, (N,))
  dprev_h, dWh = df_dot(dnext_hh, prev_h, Wh)    # next_hh = np.dot(prev_h, Wh)
  dx, dWx      = df_dot(dnext_hx, x,      Wx)    # next_hx = np.dot(x, Wx) 

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape
  N, H    = h0.shape
  
  cache = {}
  sc    = {} # sub_cache
    
  h = np.zeros((N,T,H))
  prev_c = np.zeros((N, H))
  prev_h = h0
  for i in xrange(T):
    h[:, i, :], prev_c, sc[i] = lstm_step_forward(x[:,i,:], prev_h, prev_c, Wx, Wh, b)    
    prev_h = h[:, i, :]
  
  cache = (N, T, D, H, x, h, h0, Wx, Wh, b, sc)
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  (N, T, D, H, x, h, h0, Wx, Wh, b, sc) = cache
  
  ''' forward code
  h = np.zeros((N,T,H))
  prev_h = h0
  for i in xrange(T):
    h[:, i, :], sc[i] = rnn_step_forward(x[:,i,:], prev_h, Wx, Wh, b)  # prev_h = h[:, i-1, :]
    prev_h = h[:, i, :]
  '''
  dx, dWx, dWh, db =  zeros_like(x, Wx, Wh, b)
  dprev_h = np.zeros((N,H))
  dprev_c = np.zeros((N,H)) 
  for i in reversed(xrange(T)):
    dhi = dh[:,i,:] + dprev_h
    dx[:, i, :] , dprev_h, dprev_c, dWxi, dWhi, dbi = lstm_step_backward(dhi, dprev_c, sc[i])
    dWx += dWxi
    dWh += dWhi
    db  += dbi
  dh0 = dprev_h

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

