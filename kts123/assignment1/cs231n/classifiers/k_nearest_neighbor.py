#-*- coding: utf-8 -*-
import numpy as np
from scipy import stats

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    분류기를 훈련시키기.
    k-nearest neighbors 에서의 훈련은 단순히 기억만 하고 있는 것임.
    
    입력:
    - X: (num_train, D) shape의 numpy 배열. 
         num_train 개수의 D차원 샘플로 구성된 훈련 데이터.
    - y: (N,) shape의 numpy 배열.
         훈련 라벨(ground truth). X[i]에 대한 라벨은 y[i]
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    X 의 각 테스트 포인트에 대해 self.X_train 모든 훈련 포인트와의 거리를 계산.
    훈련 데이터와 테스트 데이터 모두 훓는 이중 루프 이용하시오.

    Inputs:
    - X: (num_test, D) shape 의 numpy 배열. 테스트 데이터가 담겨 있음

    Returns:
    - dists: (num_test, num_train) shape의 numpy 배열.
      dists[i, j] := i번째 테스트 포인트와 j번째 훈련 포인트 사이의 유클리디안 거리  
    """
    num_test  = X.shape[0]
    num_train = self.X_train.shape[0]
    dists     = np.zeros((num_test, num_train))
    for i in xrange(num_test):
        for j in xrange(num_train):
        #####################################################################
        # i번째 테스트 포인트와 j번째 훈련 포인트간의 유클리디안 거리 계산  #
        # 후 그 값을 dists[i, j] 에 저장하시오.                             #
        # 디맨션을 훓는 루프를 사용하면 안됨                                #
        #####################################################################
            A  = X[i]
            B  = self.X_train[j]
        
            A_minus_B        = (A - B)
            A_minus_B_square = np.square(A_minus_B)        
            dists[i][j]      = np.sqrt(np.sum(A_minus_B_square))
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test  = X.shape[0]
    num_train = self.X_train.shape[0]
    dists     = np.zeros((num_test, num_train))
   
    B   = self.X_train
    B_square = np.sum(np.square(B), axis = 1)
        
    for i in xrange(num_test):
        ####################################################################
        # i번째 테스트 포인트와 모든 훈련 포인트와의 유클리드 거리 계산 후 #
        # 그 값을 dists[i, :] 에 저장하시오                                #
        ####################################################################
        A   = X[i]
        #B   = self.X_train
    
        A_square = np.sum(np.square(A))
        #B_square = np.sum(np.square(B), axis = 1)
    
        AB = np.dot(A, np.transpose(B))
      
        squares = A_square - 2*AB + B_square

        dists[i] = np.sqrt(squares) 
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    A  = X
    B  = self.X_train
     
    A_square = np.sum(np.square(A), axis = 1)
    B_square = np.sum(np.square(B), axis = 1)
    A_square_plus_B_square = A_square.reshape(A_square.shape[0],1) + B_square
    AB = np.dot(A, np.transpose(B))
    squaress = A_square_plus_B_square - 2*AB
    
    dists     = np.sqrt(squaress)
    
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      closest_y = np.argsort(dists[i])[0:k]
     
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      labels = [self.y_train[index] for index in closest_y]
      y_pred[i] = stats.mode(labels)[0]
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

