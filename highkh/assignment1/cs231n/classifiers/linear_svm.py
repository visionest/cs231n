# -*- coding: utf-8 -*-
import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  # compute the loss and the gradient
  num_classes = W.shape[1] # C
  num_train = X.shape[0] # D
  loss = 0.0

  for i in xrange(num_train):
    scores = X[i].dot(W)  # 예측 점수
    correct_class_score = scores[y[i]]  # 정답
    num_incorrect = 0 ## (margin > 0) = # of incorrect classes
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      # margin = w_j * x_i - w_{y[i]}*x_i + 1
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0: # incorrect class = increase loss
        loss += margin
        dW[:, j] += X[i]  # dW_j L_i = -(\sum_{j!=y_i} indicator(margin>0))*x_i
        num_incorrect += 1  ##
    dW[:, y[i]] -= num_incorrect*X[i] ## indicator output은 0 or 1이기 때문에 incorretc 개수만 세어주고 곱하면 된다

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train ##

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W ##

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  num_train = X.shape[0]
      
  scores = X.dot(W)
  #print 'scr', scores , scores.shape[0], scores.shape[1]
  #print 'y' , y , y.shape[0]
  correct_class_score = scores[np.arange(num_train), y]
  #print 'ccs', correct_class_score , correct_class_score.shape[0]
  margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + 1) # delta = 1 # [:, np.newaxis] = like transpose
  #print 'marg', margins , margins.shape[0] , margins.shape[1]
  # incorrect만 loss증가에 영향. 따라서 correct는 0
  margins[np.arange(num_train), y] = 0
  loss = np.sum(margins)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  X_mask = np.zeros(margins.shape)
  # incorrect인 경우만 개수를 세어주기 위해 1로 변경
  X_mask[margins > 0] = 1
  incorrect_counts = np.sum(X_mask, axis=1)
  X_mask[np.arange(num_train), y] = -incorrect_counts
  dW = X.T.dot(X_mask)

  dW /= num_train
  dW += reg*W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
