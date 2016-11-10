import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
  
  def __init__(self, input_dim=(3, 32, 32), num_classes=10,
               weight_scale=1e-3, reg=0.0, dtype=np.float32):
    self.params = {}
    self.input_dim = input_dim
    self.num_classes = num_classes
    self.weight_scale = weight_scale
    self.reg = reg
    self.dtype = dtype
    
    self.output_dim = input_dim
    self.forwards = []
    self.backwards = []
  
  def add_conv_relu(self, num_filters, filter_size):
    weight_scale = np.sqrt(self.output_dim[0] * filter_size * filter_size / 2.) ** -1
    W = np.random.normal(scale = self.weight_scale,
                         size = (num_filters, self.output_dim[0], filter_size, filter_size))
    b = np.zeros(num_filters)
    W = W.astype(self.dtype)
    b = b.astype(self.dtype)
    
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    
    idx = len(self.params) / 2 + 1
    kW = 'W{}'.format(idx)
    kb = 'b{}'.format(idx)
    self.params[kW] = W
    self.params[kb] = b
    
    self.forwards.append(lambda X, _: conv_relu_forward(X, self.params[kW], self.params[kb], conv_param))
    
    def backward(dout, cache):
        dout, dw, db = conv_relu_backward(dout, cache)
        return dout, { kW: dw, kb: db }
    
    self.backwards.insert(0, backward)
    
    self.output_dim = (num_filters, self.output_dim[1], self.output_dim[2])
    
  def add_max_pool(self):
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    self.forwards.append(lambda X, _: max_pool_forward_fast(X, pool_param))
    self.backwards.insert(0, lambda dout, cache: (max_pool_backward_fast(dout, cache), {}))
    
    self.output_dim = (self.output_dim[0],
                       self.output_dim[1] / 2,
                       self.output_dim[2] / 2)

  def add_affine(self, output_dim):
    input_dim = self.output_dim
    if type(input_dim) == tuple:
        input_dim = np.prod(list(input_dim))
    
    weight_scale = np.sqrt(input_dim / 2.) ** -1
    W = np.random.normal(scale = weight_scale,
                         size = (input_dim, output_dim))
    b = np.zeros(output_dim)
    W = W.astype(self.dtype)
    b = b.astype(self.dtype)
    
    idx = len(self.params) / 2 + 1
    kW = 'W{}'.format(idx)
    kb = 'b{}'.format(idx)
    self.params[kW] = W
    self.params[kb] = b
    
    self.forwards.append(lambda X, _: affine_forward(X, self.params[kW], self.params[kb]))
        
    def backward(dout, cache):
        dout, dw, db = affine_backward(dout, cache)
        return dout, { kW: dw, kb: db }
    
    self.backwards.insert(0, backward)
    
    self.output_dim = output_dim
    
  def add_relu(self):
    self.forwards.append(lambda X, _: relu_forward(X))
    self.backwards.insert(0, lambda dout, cache: (relu_backward(dout, cache), {}))
    
  def add_affine_relu(self, output_dim):
    self.add_affine(output_dim)
    self.add_relu()
    
  def add_spatial_batchnorm(self, bn_param = {}):
    idx = len(self.params) / 2 + 1
    kgamma = 'bn_gamma{}'.format(idx)
    kbeta = 'bn_beta{}'.format(idx)
    self.params[kgamma] = np.ones(self.output_dim[0]).astype(self.dtype)
    self.params[kbeta] = np.zeros(self.output_dim[0]).astype(self.dtype)
    
    def forward(X, mode):
        param = bn_param
        param['mode'] = mode
        return spatial_batchnorm_forward(X, self.params[kgamma], self.params[kbeta], param)
    
    def backward(dout, cache):
      dout, dgamma, dbeta = spatial_batchnorm_backward(dout, cache)
      return dout, { kgamma: dgamma, kbeta: dbeta }
    
    self.forwards.append(forward)
    self.backwards.insert(0, backward)
  
  def loss(self, X, y=None):

    scores = None
    mode = 'test' if y is None else 'train'
    X = X.astype(self.dtype)
    
    #############################################################################
    # forward pass
    #############################################################################
    out = X
    caches = []
    for f in self.forwards:
        out, cache = f(out, mode)
        caches.append(cache)
    caches = reversed(caches)
    
    scores = out
    
    if y is None:
      return scores
    
    #############################################################################
    # backwards pass
    #############################################################################
    reg = self.reg
    loss, grads = 0, {}
    
    loss, dout = softmax_loss(scores, y)
    Ws = [v for k, v in self.params.iteritems() if 'W' in k]
    loss += 0.5 * reg * sum(map(lambda w: np.sum(w * w), Ws))
    
    for f, cache in zip(self.backwards, caches):
      dout, grad = f(dout, cache)
      grads.update(grad)

    for k, v in self.params.iteritems():
      if 'W' in k:
        grads[k] += self.reg * v
    
    return loss, grads
    