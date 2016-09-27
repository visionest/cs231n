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
  dW = np.transpose(dW)

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):        
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]    
    for j in xrange(num_classes):            
      if j == y[i]:
        dwyi = 0
        for k in xrange(num_classes):
          margin = scores[k] - correct_class_score + 1 # note delta = 1 
          dwyi += (margin > 0)
        dwyi -= 1
        dW[y[i]] +=  (-1 * dwyi*X[i])
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[j]+= X[i]    
  dW = np.transpose(dW)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW /= num_train
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  
  """
  for c in xrange(num_classes):
    X_yc  = X[np.flatnonzero(y == c)]
    score_yc = np.transpose(X_yc.dot(W)) # score.shape = (10, ?)    
    margin = np.maximum(0, score_yc - score_yc[c] + 1)
    margin_sum = np.sum(margin) - margin.shape[1]
    loss += margin_sum
  """  
  score = X.dot(W)                              # score.shape = (500, 10)    
  score_y = score[range(score.shape[0]), y]     # score_y .shape = (500,)
  score   = np.transpose(score)
  score_y = np.transpose(score_y)   
  margin = np.maximum(0, score - score_y + 1)   # margin = (500, 10)
  margin_sum = np.sum(margin) - margin.shape[1] 
  loss += margin_sum
  loss /= num_train 

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
  score = X.dot(W)                              # score.shape = (500, 10)    
  score_y = score[range(score.shape[0]), y]     # score_y .shape = (500,)
  score   = np.transpose(score)
  score_y = np.transpose(score_y)   
  margin = np.maximum(0, score - score_y + 1)   # margin = (500, 10)
  margin_cnt = np.sum(margin > 0) - num_train
  print 'margin_cnt', margin_cnt

  # to be continue...
  
    
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
