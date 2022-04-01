# python_intro.py
"""Python Essentials: Introduction to Python.
<Parker Carlson>
<MTH 420>
<04/01/2022>
"""
import numpy as np


def sphere_volume(r):
  """computes the volume of a sphere with radius r"""
  PI = 3.14159
  return (4./3.) * PI * (r ** 3)


def prob4():
  """computes matmul of two predefined matricies"""
  A = np.array([[3, -1, 4],[1, 5, 9]])
  B = np.array([[2, 6, -5, 3], [5, -8, 9, 7], [9, -3, -2, -3]])

  return np.dot(A,B)


def tax_liability(income):
  """computes tax liability for first 3 tax brackets in US"""
  total_owed = 0.0
  income_left = income
  
  # bracket one
  if income <= 9875:
    return 0.1 * income
  else:
    total_owed = 0.1 * 9875
    income_left -= 9875

  # bracket two
  if income_left <= (40125-9875):
    total_owed += 0.12 * income_left
    return total_owed
  else:
    total_owed += 0.12 * (40125-9875)
    income_left -= (40125-9875)

  # final bracket
  total_owed += 0.22 * income_left
  return total_owed


def prob6a():
  """compares numpy arrays and python lists. python style"""
  A = [_ for _ in range(1,8)]
  B = [5 for _ in range(7)]

  # A dot B
  dot = 0.
  # A + B
  sm = []
  # 5A
  five_a = []

  for a, b in zip(A,B):
    dot += a*b
    sm.append(a+b)
    five_a.append(5*a)

  return (dot, sm, five_a)

def prob6b():
  """compares numpy arrays and python lists. numpy style"""
  A = np.array([1,2,3,4,5,6,7])
  B = np.array([5,5,5,5,5,5,5])

  return (np.dot(A,B), A + B, 5*A)


if __name__ == "__main__":
  print(sphere_volume(10))
  print(prob4())
  print(tax_liability(63000))
  print(prob6a())
  print(prob6b())
