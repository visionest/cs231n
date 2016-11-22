import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

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
  #print x.shape, Wx.shape, prev_h.shape, Wh.shape
  next_h = np.tanh(np.dot(prev_h, Wh) + np.dot(x, Wx) + b)
  cache = (x, prev_h, Wx, Wh, b, next_h)
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
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  x, prev_h, Wx, Wh, b, next_h = cache

  # tanh local derivation
  # https://theclevermachine.wordpress.com/2014/09/08/derivation-derivatives-for-common-neural-network-activation-functions/
  dout = dnext_h * (1 - next_h * next_h)
  dx = np.dot(dout, Wx.T)
  dprev_h = np.dot(dout, Wh.T)
  dWx = np.dot(x.T, dout)
  dWh = np.dot(prev_h.T, dout)
  db = np.sum(dout, axis=0)
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
  _, H = h0.shape
  h, prev_h, step_cache = np.zeros((T, N, H)), h0, {}
  for t, v in enumerate(np.transpose(x, (1,0,2))):
    h[t], step_cache[t] = rnn_step_forward(v, prev_h, Wx, Wh, b)
    prev_h = h[t]
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  cache = {}
  cache['main'] = x, h0, Wx, Wh, b
  cache['step'] = step_cache
  return np.transpose(h, (1, 0, 2)), cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  x, h0, Wx, Wh, b = cache['main']
  dx, dh0, dWx, dWh, db = np.zeros_like(x), np.zeros_like(h0), np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)
  N, T, D = x.shape
  _, H = dh0.shape


  dx = np.transpose(dx, (1,0,2))
  step_cache = cache['step']
  step_dprev_h = np.zeros((N,H))
  for t, dout in reversed(zip(range(T),
                              np.transpose(dh, (1, 0, 2)))):
    step_dx, step_dprev_h, step_dWx, step_dWh, step_db = rnn_step_backward(dout + step_dprev_h, step_cache[t])
    dx[t] += step_dx
    dWx += step_dWx
    dWh += step_dWh
    db += step_db
  dh0 += step_dprev_h
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  dx = np.transpose(dx, (1,0,2))
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
  out = W[x]
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
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  XH = np.concatenate((x, prev_h), axis=1)
  WW = np.concatenate((Wx, Wh), axis=0)
  G = np.matmul(XH, WW) + b # gate (N, 4H(i, f, o, g))
  i, f, o, g = np.split(G, 4, axis=1)
  i, f, o, g = sigmoid(i), sigmoid(f), sigmoid(o), np.tanh(g)

  next_c = f * prev_c + i * g
    
  tnext_c = np.tanh(next_c)
  next_h = o * tnext_c
    
  cache = (XH, WW, i, f, o, g, x, prev_h, prev_c, Wx, Wh, b, tnext_c)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  XH, WW, i, f, o, g, x, prev_h, prev_c, Wx, Wh, b, tnext_c = cache
  N, D = x.shape
  
  def d_mul(dout, x, y):
    return dout * y, dout * x
  def d_tanh(dout, tanh_x):   # https://theclevermachine.wordpress.com/2014/09/08/derivation-derivatives-for-common-neural-network-activation-functions/
    return dout * (1 - tanh_x * tanh_x)
  def d_sigmoid(dout, sigmoid_x):   # http://www.ai.mit.edu/courses/6.892/lecture8-html/sld015.htm
    return dout * (sigmoid_x * (1 - sigmoid_x))
  def d_add(dout, x, y):
    return dout, dout
  def d_matmul(dout, x, y):
    """
    dx = np.dot(dout, Wx.T)
    dprev_h = np.dot(dout, Wh.T)
    dWx = np.dot(x.T, dout)
    """
    return np.matmul(dout, y.T), np.matmul(x.T, dout)


  #next_h = o * tnext_c
  dout_o, dout_tnext_c = d_mul(dnext_h, o, tnext_c)
  # tnext_c = np.tanh(next_c)
  dout_next_c = d_tanh(dout_tnext_c, tnext_c)

  dnext_c += dout_next_c
    
  # next_c = sf * prev_c + si * tg
  dout_f, dout_prev_c = d_mul(dnext_c, f, prev_c)
  dout_i, dout_g = d_mul(dnext_c, i, g)

  # si, sf, so, tg = sigmoid(i), sigmoid(f), sigmoid(o), np.tanh(g)
  dout_i = d_sigmoid(dout_i, i)
  dout_f = d_sigmoid(dout_f, f)
  dout_o = d_sigmoid(dout_o, o)
  dout_g = d_tanh   (dout_g, g)

  # i, f, o, g = np.split(G, 4, axis=1)
  dout_G = np.concatenate((dout_i, dout_f, dout_o, dout_g), axis=1)

  # G = np.matmul(H, W) + b # gate (N, 4H(i, f, o, g))
  dout_b = np.sum(dout_G, axis=0)
  #print dout_G.shape, XH.shape, WW.shape
  dout_H, dout_W = d_matmul(dout_G, XH, WW)
  #print dout_H.shape, dout_W.shape

  # WW = np.concatenate((Wx, Wh), axis=0)
  dout_Wx, dout_Wh = np.split(dout_W, [D], axis=0)
  # XH = np.concatenate((x, prev_h), axis=1)
  dout_x, dout_prev_h = np.split(dout_H, [D], axis=1)


  dx, dprev_h, dprev_c, dWx, dWh, db = dout_x, dout_prev_h, dout_prev_c, dout_Wx, dout_Wh, dout_b
  # print dx.shape, dprev_h.shape, dprev_c.shape, dWx.shape, dWh.shape, db.shape
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
  _, H = h0.shape

  h, c = np.zeros((T, N, H)), np.zeros((T, N, H))
  prev_h = h0
  prev_c = np.zeros_like(h0)
  step_cache = []
  for t, v in enumerate(np.transpose(x, (1,0,2))):
    h[t], c[t], step_cache_t = lstm_step_forward(v, prev_h, prev_c, Wx, Wh, b)
    prev_h = h[t]
    prev_c = c[t]
    step_cache.append(step_cache_t)
    
  h = np.transpose(h, (1, 0, 2))
  cache = {}
  cache['main'] = x, h0, Wx, Wh, b
  cache['step'] = step_cache
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
  x, h0, Wx, Wh, b = cache['main']
  step_cache = cache['step']

  dx, dh0, dWx, dWh, db = np.zeros_like(x), np.zeros_like(h0), np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)
  N, T, D = x.shape
  _, H = dh0.shape

  dx = np.transpose(dx, (1,0,2))
  sdprev_h, sdprev_c = np.zeros((N,H)), np.zeros((N,H))
  for dout, dx_t, scache_t in reversed(zip(np.transpose(dh, (1, 0, 2)),
                                           dx,
                                           step_cache)):
    sdx, sdprev_h, sdprev_c, sdWx, sdWh, sdb = lstm_step_backward(dout + sdprev_h, sdprev_c, scache_t)
    dx_t += sdx
    dWx += sdWx
    dWh += sdWh
    db += sdb
    
  dh0 += sdprev_h
  dx = np.transpose(dx, (1,0,2))
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

