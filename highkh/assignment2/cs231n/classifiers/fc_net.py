import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

############################################################################
## for BatchNormalization                                                 ##
############################################################################
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
  fc, fc_cache = affine_forward(x, w, b)
  bn, bn_cache = batchnorm_forward(fc, gamma, beta, bn_param)
  out, relu_cache = relu_forward(bn)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache

def affine_bn_relu_backward(dout, cache):
  fc_cache, bn_cache, relu_cache = cache
  drelu = relu_backward(dout, relu_cache)
  dbn, dgamma, dbeta = batchnorm_backward(drelu, bn_cache)
  dx, dw, db = affine_backward(dbn, fc_cache)

  return dx, dw, db, dgamma, dbeta
############################################################################
############################################################################

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
   
    self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
    self.params['b1'] = np.zeros(hidden_dim)
    
    self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b2'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    
    W1 = self.params['W1']
    b1 = self.params['b1']
    W2 = self.params['W2']
    b2 = self.params['b2']
    
    ##fc1, ReLU2
    #out, cache = affine_relu_forward(x, w, b)
    hid_out, hid_cache = affine_relu_forward(X, W1, b1)
    
    ##fc3
    #out, cache = affine_forward(x, w, b)
    scores, scores_cache = affine_forward(hid_out, W2, b2)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    
    ##loss, dx = softmax_loss(x, y)
    data_loss, data_scores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg *(np.sum(W1**2) + np.sum(W2**2))
    loss = data_loss + reg_loss

    ##dx, dw, db = affine_backward(dout, cache)
    dx1, dW2, db2 = affine_backward(data_scores, scores_cache)
    dW2 += self.reg * W2

    ##dx, dw, db = affine_relu_backward(dout, cache)
    dx, dW1, db1 = affine_relu_backward(dx1, hid_cache)
    dW1 += self.reg * W1

    #grads.update({'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2})
    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    
    ##listed dim's [img_size=D=32*32*3, H1=20, H2=30, C=10]
    list_dimension = [input_dim] + hidden_dims + [num_classes]
    '''Two layer case
    ## W1.shape = (D, H1), b1,shape = (H1, )    
    self.params['W1'] = weight_scale*np.random.randn(list_dimension[0], list_dimension[1])
    self.params['b1'] = np.zeros(list_dimension[1])
    ## W2.shape = (H1, H2), b1,shape = (H2, )    
    self.params['W2'] = weight_scale*np.random.randn(list_dimension[1], list_dimension[2])
    self.params['b2'] = np.zeros(list_dimension[2])
    '''
    ## Multi(>2) layer case
    for i in xrange(self.num_layers):
        self.params['b%d' % (i+1)] = np.zeros(list_dimension[i+1])
        #self.params['W%d' % (i+1)] = np.random.normal(scale=weight_scale, size=(list_dimension[i], list_dimension[i+1]))
        self.params['W%d' % (i+1)] = weight_scale*np.random.randn(list_dimension[i], list_dimension[i+1])
        
        if use_batchnorm and i != (self.num_layers - 1):
            self.params['gamma%d' % (i+1)] = np.ones(list_dimension[i+1])
            self.params['beta%d' % (i+1)] = np.zeros(list_dimension[i+1])
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    
    ''' 2 layer w/o batchnorm
    ##out, cache = affine_relu_forward(x, w, b)
    layer1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
    layer2, cache2 = affine_relu_forward(layer1, self.params['W2'], self.params['b2'])
    ##out, cache = affine_forward(x, w, b)
    weight3 = 'W3'
    bias3 = 'b3'
    scores, cache_scores = affine_forward(layer2, self.params[weight3], self.params[bias3])
    '''
    
    ''' multilayer w/o batchnorm
    multilayer = {}
    multilayer[0] = X
    ## >>>>>>>>>>>>>> layer = {0:X}    :    dictionary
    ## >>>>>>>>>>>>>> {3,4,5,5} O      :    set
    ## >>>>>>>>>>>>>> {[1,2],3} X      :    set does NOT have list as element!!
    
    cache_multilayer = {}

    for i in xrange(1, self.num_layers):
      multilayer[i], cache_multilayer[i] = affine_relu_forward(multilayer[i - 1],
                                                     self.params['W%d' % i],
                                                     self.params['b%d' % i])

        
    
    WLast = 'W%d' % self.num_layers
    bLast = 'b%d' % self.num_layers
    scores, cache_scores = affine_forward(multilayer[self.num_layers - 1],
                                          self.params[WLast],
                                          self.params[bLast])
    '''
    
    ## add batch normalization
    scores = X
    caches = {}
    
    
    for i in xrange(1, self.num_layers + 1):
      W_layer = 'W%d' % (i)
      b_layer = 'b%d' % (i)
      gamma_layer = 'gamma%d' % (i)
      beta_layer = 'beta%d' % (i)
      cache_layer = 'cache%d' % (i)
        
      if i ==  self.num_layers:
        scores, cache = affine_forward(scores, self.params[W_layer], self.params[b_layer])
      else:
        if self.use_batchnorm:
          #scores, self.bn_cache[bn_layer] = affine_bn_relu_forward(scores, self.params[gamma_layer], self.params[beta_layer], self.bn_params[i-1])
          scores, cache = affine_bn_relu_forward(scores, self.params[W_layer], self.params[b_layer], self.params[gamma_layer], self.params[beta_layer], self.bn_params[i-1])
        else:
          scores, cache = affine_relu_forward(scores, self.params[W_layer], self.params[b_layer])
                
      caches[cache_layer] = cache
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}  ### grads : dictionary
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    
    #dx = {}
    loss, diff_scores = softmax_loss(scores, y)
    ''' 2 layer w/o batch norm
    data_loss += 0.5*self.reg*np.sum(self.params['W1']*self.params['W1'])
    data_loss += 0.5*self.reg*np.sum(self.params['W2']*self.params['W2'])
    data_loss += 0.5*self.reg*np.sum(self.params[weight3]*self.params[weight3])
    ##dx, dw, db = affine_backward(dout, cache)
    dx[self.num_layers], grads[weight3], grads[bias3] = affine_backward(data_scores, cache_scores)
    grads[weight3] += self.reg * self.params[weight3]
    
    dx[2], grads['W2'], grads['b2'] = affine_relu_backward(dx[3], cache2)
    dx[1], grads['W1'], grads['b1'] = affine_relu_backward(dx[2], cache1)
    
    grads['W2'] += self.reg*self.params['W2']
    grads['W1'] += self.reg*self.params['W1']
    '''
    
    ''' multilayer w/o batchnorm
    for i in xrange(1, self.num_layers + 1):
        loss += 0.5*self.reg*np.sum(self.params['W%d' % i]*self.params['W%d' % i])
    
    ## grads of the last layer
    dx[self.num_layers], grads[WLast], grads[bLast] = affine_backward(diff_scores, cache_scores)
    grads[WLast] += self.reg+self.params[WLast]
    
    for i in reversed(xrange(1, self.num_layers)):
        dx[i], grads['W%d' % i], grads['b%d' % i] = affine_relu_backward(dx[i + 1],
                                                                         cache_multilayer[i])
        grads['W%d' % i] += self.reg * self.params['W%d' % i]
    '''
    
    ### add batch normalization 
    for i in xrange(self.num_layers, 0, -1):
        W_layer = 'W%d' % (i)
        b_layer = 'b%d' % (i)
        gamma_layer = 'gamma%d' % (i)
        beta_layer = 'beta%d' % (i)
        cache_layer = 'cache%d' % (i)
        
        loss += 0.5*self.reg*np.sum(self.params[W_layer]**2)

        if i ==  self.num_layers:
            diff_scores, grads[W_layer], grads[b_layer] = affine_backward(diff_scores, caches[cache_layer])
        else:
            if self.use_batchnorm:
                diff_scores, grads[W_layer], grads[b_layer], grads[gamma_layer], grads[beta_layer] = affine_bn_relu_backward(diff_scores, caches[cache_layer])
            else:
                diff_scores, grads[W_layer], grads[b_layer] = affine_relu_backward(diff_scores, caches[cache_layer])
            
        
        grads[W_layer] += self.reg*self.params[W_layer]
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
