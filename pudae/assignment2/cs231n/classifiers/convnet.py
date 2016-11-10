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
  
  def add_conv(self, num_filters, filter_size):
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
    
    self.forwards.append(lambda X: conv_forward_fast(X, self.params[kW], self.params[kb], conv_param))
    self.backwards.insert(0, conv_backward_fast)
    
    self.output_dim = (num_filters, self.output_dim[1], self.output_dim[2])
    
  def add_max_pool(self):
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    self.forwards.append(lambda X: max_pool_forward_fast(X, pool_param))
    self.backwards.insert(0, lambda dout, cache: (max_pool_backward_fast(dout, cache), None, None))
    
    self.output_dim = (self.output_dim[0],
                       self.output_dim[1] / 2,
                       self.output_dim[2] / 2)

  def add_affine(self, output_dim):
    input_dim = self.output_dim
    if type(input_dim) == tuple:
        input_dim = np.prod(list(input_dim))
        
    W = np.random.normal(scale = self.weight_scale,
                         size = (input_dim, output_dim))
    b = np.zeros(output_dim)
    W = W.astype(self.dtype)
    b = b.astype(self.dtype)
    
    idx = len(self.params) / 2 + 1
    kW = 'W{}'.format(idx)
    kb = 'b{}'.format(idx)
    self.params[kW] = W
    self.params[kb] = b
    
    self.forwards.append(lambda X: affine_forward(X, self.params[kW], self.params[kb]))
    self.backwards.insert(0, affine_backward)
    
    self.output_dim = output_dim
    
  def add_relu(self):
    self.forwards.append(lambda X: relu_forward(X))
    self.backwards.insert(0, lambda dout, cache: (relu_backward(dout, cache), None, None))
    
  def add_affine_relu(self, output_dim):
    self.add_affine(output_dim)
    self.add_relu()
  
  def loss(self, X, y=None):

    scores = None
    
    #############################################################################
    # forward pass
    #############################################################################
    out = X
    caches = []
    for f in self.forwards:
        out, cache = f(out)
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
    
    dWbs = []
    for f, cache in zip(self.backwards, caches):
      dout, dW, db = f(dout, cache)
      if dW is not None:
        dWbs.append((dW, db))

    for i, (dW, db) in enumerate(reversed(dWbs)):
      idx = i + 1
      kW = 'W{}'.format(i + 1)
      kb = 'b{}'.format(i + 1)
      grads[kW] = dW + self.reg * self.params[kW]
      grads[kb] = db
    
    return loss, grads
    