import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    F, HH, WW = num_filters, filter_size, filter_size
    
    W1 = np.random.normal(scale=weight_scale, size=(F, C, HH, WW))
    W2 = np.random.normal(scale=weight_scale, size=(F*H*W/4, hidden_dim))
    W3 = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
    b1 = np.zeros(F)
    b2 = np.zeros(hidden_dim)
    b3 = np.zeros(num_classes)

    self.params['W1'], self.params['W2'], self.params['W3'] = W1, W2, W3
    self.params['b1'], self.params['b2'], self.params['b3'] = b1, b2, b3
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    dfs = []
   
    def forward(f, fd, idx, x, conv_param=None, pool_param=None):
      w = self.params['W%d' % idx]
      b = self.params['b%d' % idx]
      out, cache = None, None
      #print 'w:', w.shape
      if (conv_param == None and pool_param == None):
        out, cache = f(x, w, b)
        dfs.append(lambda dout : fd(dout, cache))
      elif (conv_param != None and pool_param != None):
        out, cache = f(x, w, b, conv_param, pool_param)
        dfs.append(lambda dout : fd(dout, cache))
      else:
        print "forward parameter is wrong"
        
      return out

    out = forward(conv_relu_pool_forward, conv_relu_pool_backward, 1, X, conv_param, pool_param)
    out = forward(affine_relu_forward, affine_relu_backward, 2, out)
    out = forward(affine_forward, affine_backward, 3, out)
    
    scores = out
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    
    Ws = [v for k, v in self.params.iteritems() if 'W' in k]
    loss += 0.5 * self.reg * sum(map(lambda w: np.sum(w * w), Ws))

    """
    dout, grads['W3'], grads['b3'] = dfs[2](dout)
    dout, grads['W2'], grads['b2'] = dfs[1](dout)
    dout, grads['W1'], grads['b1'] = dfs[0](dout)
    
    grads['W3'] += self.reg * self.params['W3']
    grads['W2'] += self.reg * self.params['W2']
    grads['W1'] += self.reg * self.params['W1']
    """

    dfs_len = len(dfs)
    for i, v in enumerate(reversed(dfs)):
      idx = dfs_len - i
      dout, grads['W%d'%idx], grads['b%d'%idx] = dfs[idx-1](dout)
      grads['W%d'%idx] += self.reg * self.params['W%d'%idx]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
