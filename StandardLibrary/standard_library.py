# standard_library.py
"""Python Essentials: The Standard Library.
<Parker Carlson>
<MTH 420>
<4/8/22>
"""

import calculator as calc
from itertools import combinations

# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """
    return (min(L), max(L), sum(L)/len(L))


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    print("Lists and Sets are mutable;\nInts, Strings, and Tuples are inmutable.")


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt that are 
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    return calc.sqrt(calc.sum(calc.product(a,a), calc.product(b,b)))


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    my_sets = []
    for set_len in range(len(A)+1):
      for combos in combinations(A, set_len):
        my_sets.append(set(combos))

    return my_sets


# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
