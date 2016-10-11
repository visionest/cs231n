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
  num_classes = W.shape[1]
  num_train = X.shape[0]

  data_loss = 0
  dscores = np.zeros_like(dW)
  for i in xrange(num_train):
    # evaluate class scores, [K]
    scores = X[i].dot(W)

    # compute class probablities
    exp_scores = np.exp(scores)
    prob = exp_scores / np.sum(exp_scores)
    correct_logprob = -np.log(prob[y[i]])
    
    # data_loss
    data_loss += correct_logprob

    # compute the gradient on scores
    dscore = prob             # [K,]
    dscore[y[i]] -= 1
    dscore /= num_train
  
    # backpropate the gradient to the parameters (W,b)
    dW += np.dot(np.expand_dims(X[i], 1), np.expand_dims(dscore, 0))

  # compute the loss: average cross-entropy loss and regularization
  data_loss /= num_train
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  
  dW += reg*W # regularization gradient

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  # evaluate class scores, [K]
  scores = np.dot(X, W)
  scores -= np.amax(scores, axis=1, keepdims=True)

  # compute class probablities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=1)
  correct_logprobs = -np.log(probs[range(num_train), y] + 10**-10)

  # compute the loss: average cross-entropy loss and regularization
  data_loss = np.mean(correct_logprobs)
  #data_loss /= num_train
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss

  # compute the gradient on scores
  dscores = probs             # [K,C]
  dscores[range(num_train), y] -= 1
  dscores /= num_train
  
  # backpropate the gradient to the parameters (W,b)
  dW += np.dot(X.T, dscores)
  dW += reg*W # regularization gradient

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

