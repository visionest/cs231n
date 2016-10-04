# -*- coding: utf-8 -*-
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_train = X.shape[0]
  num_class = W.shape[1]
  
  for i in xrange(num_train):
    train_score = X[i].dot(W)
    label_score = train_score[y[i]]
    
    score_exp = np.exp(train_score)
    deno_sum = np.sum(score_exp)
    # Softmax
    frac_exp = score_exp / deno_sum
    # Cross-entropy
    ce = -np.log(frac_exp[y[i]])
    loss += ce
    # Gradient
    for c in xrange(num_class):
      dW[:, c] += X[i] * (frac_exp[c] - (c == y[i]))

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  dW /= num_train
  dW += reg*W   
    
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_train = X.shape[0]
  train_score = X.dot(W)
  score_exp = np.exp(train_score)
  deno_sum = np.sum(score_exp, axis=1, keepdims=True)
  
  # Softmax
  frac_exp = score_exp / deno_sum
  # Cross-entropy
  ce = -np.log(frac_exp[np.arange(num_train), y])
  loss = np.sum(ce)

  label_indx = np.zeros_like(frac_exp)
  label_indx[np.arange(num_train), y] = 1
  # Gradient
  dW = X.T.dot(frac_exp - label_indx)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

