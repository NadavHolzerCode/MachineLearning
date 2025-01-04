import numpy as np
# Dont add any more imports here!

# Make Sure you fill your ID here. Without this ID you will not get a grade!
def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 205810963
def ID2():
    '''
        Personal ID of the second student.
    '''
    # Insert your ID here
    return 000000000

def sigmoid(z):
  return 1 / (1 + np.exp(-z))
    
def cross_entropy(t, y):
  return -t * np.log(y) - (1 - t) * np.log(1 - y)


def get_accuracy(y, t):
  acc = 0
  N = 0
  for i in range(len(y)):
    N += 1
    if (y[i] >= 0.5 and t[i] == 1) or (y[i] < 0.5 and t[i] == 0):
      acc += 1
  return acc / N

def pred(w, b, X):
  """
  Returns the prediction `y` of the target based on the weights `w` and scalar bias `b`.

  Preconditions: np.shape(w) == (90,)
                 type(b) == float
                 np.shape(X) = (N, 90) for some N
  Postconditions: np.shape(y)==(N,)

  >>> pred(np.zeros(90), 1, np.ones([2, 90]))
  array([0.73105858, 0.73105858]) # It's okay if your output differs in the last decimals
  """
  # Preconditions
#   print(f'Shape of w is okay (90,): {np.shape(w) == (90,)}')
#   print(f'Type if b is okay (float): {type(b) == float or type(b) == int}')
#   print(f'Shape of X is okay (N,90): {np.shape(X)[1] == 90}')
  
  #compute
  z = np.dot(X,w) + b
  y = sigmoid(z)
  
  #postconditions:
#   print(f'y size is {np.shape(y)}, and it equal to N: {np.shape(y)[0] == np.shape(X)[0]}')
  return y

def cost(y, t):
  """
  Returns the cost(risk function) `L` of the prediction 'y' and the ground truth 't'.

  - parameter y: prediction
  - parameter t: ground truth
  - return L: cost/risk
  Preconditions: np.shape(y) == (N,) for some N
                 np.shape(t) == (N,)
  
  Postconditions: type(L) == float
  >>> cost(0.5*np.ones(90), np.ones(90))
  0.69314718 # It's okay if your output differs in the last decimals
  """
  #preconditions
#   print(f'Shape of y is okay (N,): {np.shape(y)}')
#   print(f'Shape of t is okay (N,): {np.shape(t)}')
  
  #compute
  L = np.mean(cross_entropy(t, y))
  
  #postconditions
#   print(f'type of L supposed to be float: {type(L)}')
  return L

def derivative_cost(X, y, t):
  """
  Returns a tuple containing the gradients dLdw and dLdb.

  Precondition: np.shape(X) == (N, 90) for some N
                np.shape(y) == (N,)
                np.shape(t) == (N,)

  Postcondition: np.shape(dLdw) = (90,)
           type(dLdb) = float
           return dLdw,dldb
  """
  #preconditions
#   print(f'Shape of X is okay (N,90): {np.shape(X)}')
#   print(f'Shape of y is okay (N,): {np.shape(y)}')
#   print(f'Shape of t is okay (N,): {np.shape(t)}')
  
  #compute
  dLdw = -1/np.shape(X)[0]*(np.dot((t-y), X))
  dLdb = -np.mean(t-y)
  
  #postconditions
#   print(f'dLdw size is {np.shape(dLdw)}, and it equal to (90,): {np.shape(dLdw) == (90,)}')
#   print(f'type of dLdb supposed to be float: {type(dLdb)}')

  return (dLdw,dLdb)