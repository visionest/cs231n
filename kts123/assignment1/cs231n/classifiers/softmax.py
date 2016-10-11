import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)  # dw.shape = (D,C)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train   = X.shape[0]
  dW = np.transpose(dW)
  for i in xrange(num_train):
    scores = X[i].dot(W)                               # scores.shape = (C,)
    scores -= np.max(scores)                    
    correct_class_score = scores[y[i]]
    softmaxs = np.exp(scores) / np.sum(np.exp(scores)) # softmaxs.shape = (C,)
    correct_class_softmax = softmaxs[y[i]]
    loss_i = -1*np.log(correct_class_softmax)
    loss += loss_i    
    for j in xrange(num_classes):
      dW[j] += (softmaxs[j]*X[i])
      if (j == y[i]):
        dW[j] += (-1)*X[i]
  dW = np.transpose(dW)
        
  loss /= num_train
  dW   /= num_train  
  dW   += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  num_train   = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores   = X.dot(W)                             # score.shape   = (N, C)  
  scores  -= np.max(scores)
  exp_scores = np.exp(scores) 
  softmaxs = np.exp(scores) / np.sum(exp_scores, axis = 1)[:,np.newaxis]  # softmaxs.shape = (N,C)
  correct_class_softmax = softmaxs[xrange(num_train), y]
  loss = np.sum(-1*np.log(correct_class_softmax))
  loss /= num_train    

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

