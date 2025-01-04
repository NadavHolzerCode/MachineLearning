import numpy as np
def ID1():
    '''
        Write your personal ID here.
    '''
    # Insert your ID here
    return 205810963
def ID2():
    '''
        Only If you were allowed to work in a pair will you fill this section and place the personal id of your partner otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

def LeastSquares(X,y):
  '''
    Calculates the Least squares solution to the problem
    X*theta=y using the least squares method
    :param X: input matrix
    :param y: input vector
    :return: theta = (Xt*X)^(-1) * Xt * y 
  '''
  size = X.shape
  X = np.c_[X, np.ones(size[0])]
  print(X)
  return np.dot(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)

def classification_accuracy(model,X,s):
  '''
    calculate the accuracy for the classification problem
    :param model: the classification model class
    :param X: input matrix
    :param s: input ground truth label
    :return: accuracy of the model
  '''
  test_pred = model.predict(X)
  acc = test_pred - s
  return 100*np.sum(acc == 0)/len(s)

def linear_regression_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the linear regression problem.  
  '''
  return [-0.05815511,  0.00370502, -0.03286861, -0.00778299, -0.02244268, -0.01029301,
  0.08848966, -0.01828506,  0.06681716, -0.03677032,  0.02622848,  0.01483321,
  0.07075521,  0.15158242,  0.78889119,  0.0170496,   0.02292242, -0.02476731,
 -0.00271557, -0.01844768,  0.0079023,   0.03596398,  0.0201349,  -0.04256833,
 -0.03422087,  0.00615061, -0.0037684,  -0.01784053]

def linear_regression_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return -1.6986470058791918e-16

def classification_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the classification problem.  
  '''
  return [ 0.01490529,  0.13365614, -0.03762084, -0.22083703, -0.05900128, -0.4314777,
  -0.28287207, -0.0224343,   0.01777695,  0.0717757,  -0.56620682,  0.18845946,
   0.00480852,  1.20600109,  2.8025857,  -0.40424394,  0.17481964, -0.23747771,
  -0.26726405, -0.26837457,  0.10053936,  0.05842024,  0.38333368, -0.42969027,
  -0.10837668, -0.20253965,  0.09074067,  0.34739936]

def classification_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return [0.41152437]

def classification_classes_submission():
  '''
    copy the classes values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of classes for the classification problem.  
  '''
  return [0, 1]