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
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # shape of scores: (N, C)
  scores = np.matmul(X, W)
  scores -= np.expand_dims(np.amax(scores, axis=1), 1)
  
  # shape of softmax: (N, C)
  scores_exp = np.exp(scores)
  scores_exp_corr = scores_exp[np.arange(y.shape[0]), y]
  scores_exp_sum = np.sum(scores_exp, axis=1)
  softmax = scores_exp_corr / scores_exp_sum
  
  # shape of cross_entropy: (N, )
  cross_entropy = -1 * np.log(softmax)
  loss = np.mean(cross_entropy)
  loss += 0.5 * reg * np.sum(W * W)


  # calculate gradient
  # shape of p: (N, C)
  p = scores_exp / np.expand_dims(scores_exp_sum, 1)
  yi = np.zeros(p.shape)
  yi[range(num_train), y] = 1
  
  dW = np.mean(np.expand_dims(X, 1) * np.expand_dims(p-yi, 2), axis=0).T
  dW += reg * W

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

  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # shape of scores: (N, C)
  scores = np.matmul(X, W)
  scores -= np.expand_dims(np.amax(scores, axis=1), 1)
  
  # shape of softmax: (N, C)
  scores_exp = np.exp(scores)
  scores_exp_corr = scores_exp[np.arange(y.shape[0]), y]
  scores_exp_sum = np.sum(scores_exp, axis=1)
  softmax = scores_exp_corr / scores_exp_sum
  
  # shape of cross_entropy: (N, )
  cross_entropy = -1 * np.log(softmax)
  loss = np.mean(cross_entropy)
  loss += 0.5 * reg * np.sum(W * W)
    
  # calculate gradient
  # shape of p: (N, C)
  p = scores_exp / np.expand_dims(scores_exp_sum, 1)
  yi = np.zeros(p.shape)
  yi[range(num_train), y] = 1
  
  dW = np.mean(np.expand_dims(X, 1) * np.expand_dims(p-yi, 2), axis=0).T
  dW += reg * W
  
  # calculate gradient

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

