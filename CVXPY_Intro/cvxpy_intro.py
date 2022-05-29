# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
Parker Carlson
MTH 420
May 20, 2022
"""
import cvxpy as cp
import numpy as np


def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # set up problem
    x = cp.Variable(3, nonneg=True)
    c = np.array([2, 1, 3])
    objective = cp.Minimize(c.T @ x)

    # constraints
    A = np.array([1, 2, 0])
    B = np.array([0, 1, -4])
    C = np.array([2, 10, 3])
    P = np.eye(3)
    constraints = [A @ x <= 3, B @ x <= 1, C @ x >= 12, P @ x >= 0]

    # assemble and solve the problem
    problem = cp.Problem(objective, constraints)
    min_val = problem.solve()
    return x.value, min_val


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # set up problem
    x = cp.Variable(A.shape[1], nonneg=True)
    objective = cp.Minimize(cp.norm(x, 1))

    # constraints
    constraints = [A @ x == b]

    # assemble and solve the problem
    problem = cp.Problem(objective, constraints)
    min_val = problem.solve()
    return x.value, min_val


# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # set up problem
    x = cp.Variable(6, nonneg=True)
    c = np.array([4, 7, 6, 8, 8, 9])         # from table 15.3
    objective = cp.Minimize(c.T @ x)

    # constraints
    A = np.array([[1, 1, 0, 0, 0, 0],[0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]]) # num supplied <= num_avail
    a = np.array([7, 2, 4]).T
    B = np.array([[1, 0, 1, 0, 1, 0],[0, 1, 0, 1, 0, 1]])                     # num recv'd >= num_needed
    b = np.array([5, 8])

    P = np.eye(6)
    constraints = [A @ x <= a, B @ x >= b, P @ x >= 0]

    # assemble and solve the problem
    problem = cp.Problem(objective, constraints)
    min_val = problem.solve()
    return x.value, min_val


# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    Q = np.array([[3,2,1],[2,2,2],[1,2,3]])
    r = np.array([3, 0, 1])
    x = cp.Variable(3, nonneg=True)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q) + r.T @ x))
    res = prob.solve()
    return x.value, res



# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # set up problem
    x = cp.Variable(A.shape[1], nonneg=True)
    objective = cp.Minimize(cp.norm(A @ x - b, 2))

    # constraints
    P = np.eye(A.shape[1])
    C = np.ones(A.shape[1])
    constraints = [C @ x == 1,P @ x >= 0 ]

    # assemble and solve the problem
    problem = cp.Problem(objective, constraints)
    min_val = problem.solve()
    return x.value, min_val


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    raise NotImplementedError("Problem 6 Incomplete")

