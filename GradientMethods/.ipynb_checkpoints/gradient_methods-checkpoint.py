# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
Parker Carlson
MTH 420
5/17/22
"""
import numpy as np
import scipy.optimize


# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    min = np.zeros_like(x0, dtype=np.float32)
    x_k = x0.copy()
    p_k = x0.copy()
    alpha_k = 1.
    for i in range(maxiter):
        # update step weight
        def alpha_fun(alpha):
            return f(x_k - alpha*Df(x_k))

        alpha_k = scipy.optimize.minimize_scalar(alpha_fun).x

        # find direction of descent
        p_k = -Df(x_k)

        # update x_k
        x_k = x_k + (alpha_k * p_k)

        # check stopping criteria: ||grad f(x_k)||_inf < tol
        if np.amax(Df(x_k)) < tol:
            return x_k, True, i+1

    # did not converge
    return x_k, False, maxiter


# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    x_k = x0.copy()
    r_k = Q @ x0 - b
    d_k = r_k.copy()
    d_k *= -1
    for i in range(Q.shape[0]):
        alpha_k = (r_k.T @ r_k)/(d_k.T @ Q @ d_k)
        x_k = x_k + (alpha_k * d_k)
        r_k1 = r_k + alpha_k * (Q @ d_k)
        beta_k = (r_k1.T @ r_k1)/(r_k.T @ r_k)
        r_k = r_k1
        d_k = -r_k + (beta_k*d_k)

        # check stopping criteria: ||r_k|| < tol
        if ((r_k @ r_k.T) ** (0.5)) < tol:
            return (x_k, True, i+1)

    # did not converge
    return (x_k, False, Q.shape[0])


# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    r_k = -1*df(x0).T
    d_k = r_k.copy()
    x_k = x0.copy()
    def alpha_fun(alpha):
        return f(x_k - alpha*d_k)

    alpha_k = scipy.optimize.minimize_scalar(alpha_fun).x
    x_k = x0 + alpha_k*d_k
    for i in range(1,maxiter):
        r_k1 = -1*df(x_k).T
        #print(r_k1)
        beta_k = (r_k1.T @ r_k1) / (r_k.T @ r_k)
        #print(beta_k)
        r_k = r_k1
        d_k = r_k + (beta_k * d_k)
        #print(d_k)
        # update step weight
        def alpha_fun(alpha):
            return f(x_k - alpha*d_k)

        alpha_k = scipy.optimize.minimize_scalar(alpha_fun).x

        # update x_k
        x_k = x_k + (alpha_k * d_k)

        # check stopping criteria: ||grad f(x_k)||_inf < tol
        print((r_k.T @ r_k) ** (0.5))
        if ((r_k.T @ r_k) ** (0.5)) < tol:
            return (f(x_k), True, i+1)

    # did not converge
    return (f(x_k), False, maxiter)

# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    data = np.loadtxt(filename)
    A = data.copy()
    A[:,0] = 1.
    b = data[:,0]
    res = conjugate_gradient(A.T @ A, A.T @ b, x0, tol=1e-4)
    return res[0]


# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        raise NotImplementedError("Problem 5 Incomplete")

    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    raise NotImplementedError("Problem 6 Incomplete")
