import numpy as np


def gradient(f, x, delta=1e-5):
    """
    Returns the gradient of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method


    Returns:
        ret (numpy.array): gradient of f at the point x
    """
    #TODO
    n, = x.shape
    gradient = np.zeros(n).astype('float64')

    return gradient


def jacobian(f, x, delta=1e-5):
    """
    Returns the Jacobian of function f at the point x
    Parameters:
        f (numpy.array -> numpy.array): A function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (f(x).shape[0], x.shape[0])
                            which is the jacobian of f at the point x
    """
    #TODO
    n, = x.shape
    m, = f(x).shape
    x = x.astype('float64') #Need to ensure dtype=np.float64 and also copy input. 
    gradient = np.zeros((m, n)).astype('float64')
    
    return gradient



def hessian(f, x, delta=1e-5):
    """
    Returns the Hessian of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (x.shape[0], x.shape[0])
                            which is the Hessian of f at the point x
    """
    #TODO
    pass
    


