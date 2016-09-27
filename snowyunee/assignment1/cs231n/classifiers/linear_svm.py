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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,y[i]] -= X[i]
        dW[:,j] += X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  # 1/2 * reg * W^2 =>  reg * W 
  dW += reg * W

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  #for i in xrange(num_train):
  #scores = X[i].dot(W)
  scores = np.tensordot(X,W,([1],[0]))
  #print scores.shape
  #(500,10)

  correct_class_score = scores[range(y.shape[0]), y]
  #print correct_class_score  # (500,)
  margin = scores - np.expand_dims(correct_class_score,1) + 1
  #print 'margin:', margin[0]
  margin_mask = (margin > 0)
  margin_mask[range(y.shape[0]), y] = 0
  margin = margin * margin_mask
  #print 'margin2:', margin[0]
  loss += margin.sum()

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
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
# dW[:,y[i]] -= X[i]
# dW[:,j] += X[i]
  #print 'margin_mask1:', margin_mask[0]
  true_cnts = np.sum(margin_mask, axis=1)
  #print 'true_cnts1:', true_cnts[0]
  #print 'ground truth:', y[0]
  margin_mask = margin_mask.astype(float)
  margin_mask[range(y.shape[0]), y] = -1 * true_cnts
  #print 'margin_mask2:', margin_mask[0]
  X_product_margin = np.expand_dims(X,2) * np.expand_dims(margin_mask,1)
  #print X_product_margin.shape
  dW = np.sum(np.expand_dims(X,2) * np.expand_dims(margin_mask,1), axis=0)

  dW /= num_train

  # 1/2 * reg * W^2 =>  reg * W 
  dW += reg * W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
