# python_intro.py
"""Python Essentials: Introduction to Python.
Parker Carlson
MTH 420
April 15, 2022
"""

import numpy as np

#Problem 1
def isolate(a, b, c, d, e):
    print(a,b,c,sep="     ", end=" ")
    print(d,e)
    

#Problem 2
def first_half(string):
    return string[:int(len(string)/2)]



def backward(first_string):
    return first_string[::-1]

#Problem 3
def list_ops():
    l = ["bear", "ant", "cat", "dog"]

    l.append("eagle")
    l[2] = "fox"
    l.pop(1)
    l.sort()
    l = l[::-1]
    l[l.index("eagle")] = "hawk"
    l.append("hunter")

    return l

    raise NotImplementedError("Problem 3 Incomplete")

#Problem 4
def alt_harmonic(n):
    """Return the partial sum of the first n terms of the alternating
    harmonic series. Use this function to approximate ln(2).
    """
    return sum([((-1)**(i+1))/i for i  in range(1,n+1)])



def prob5(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    new_A = A.copy()
    new_A[new_A < 0] = 0
    return new_A
    

def prob6():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A = np.array([[0,2,4],[1,3,5]])
    B = np.array([[3,0,0],[3,3,0],[3,3,3]])
    C = -2 * np.eye(3)

    col1 = np.vstack((np.zeros((3,3)),A,B))
    col2 = np.vstack((A.T, np.zeros((2,2)), np.zeros((3,2))))
    col3 = np.vstack((np.eye(3), np.zeros((2,3)), C))

    return np.hstack((col1, col2, col3))

def prob7(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    raise NotImplementedError("Problem 7 Incomplete")


def prob8():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    grid = np.load("grid.npy")
    horizontal = grid[:,:-3] * grid[:, 1:-2] * grid[:, 2:-1] * grid[:,3:]
    vertical = grid[:-3,:] * grid[1:-2,:] * grid[2:-1,:] * grid[3:,:]
    left_diag = grid[:-3,:-3] * grid[1:-2, 1:-2] * grid[2:-1, 2:-1] * grid[3:,3:]
    right_diag = grid[3:,:-3] * grid[2:-1, 1:-2] * grid[1:-2,2:-1] * grid[:-3,3:]
    res = [np.max(horizontal), np.max(vertical), np.max(left_diag), np.max(right_diag)]

    return max(res)

