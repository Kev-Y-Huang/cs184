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
    n, = x.shape
    gradient = np.zeros(n).astype('float64')

    for i in range(n):
        onehot = np.zeros(n).astype('float64')
        onehot[i] = delta
        gradient[i] = (f(x + onehot) - f(x - onehot)) / (2 * delta)

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
    n, = x.shape
    m, = f(x).shape
    # Need to ensure dtype=np.float64 and also copy input.
    x = x.astype('float64')
    jacobian = np.zeros((m, n)).astype('float64')

    for i in range(n):
        for j in range(m):
            onehot = np.zeros(n).astype('float64')
            onehot[i] = delta
            jacobian[j][i] = (f(x + onehot)[j] -
                              f(x - onehot)[j]) / (2 * delta)

    return jacobian


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
    n, = x.shape
    hessian = np.zeros((n, n)).astype('float64')

    for i in range(n):
        onehot_first = np.zeros(n).astype('float64')
        onehot_first[i] = delta
        gradient_pos = gradient(f, x + onehot_first, delta)
        gradient_neg = gradient(f, x - onehot_first, delta)
        for j in range(n):
            hessian[j][i] = (gradient_pos[j] - gradient_neg[j]) / (2 * delta)

    return hessian
