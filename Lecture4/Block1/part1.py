#------------------------------------------------------------------------------
#
#   Lecture 4 - Deep Dive & Implementation
#   Block 1 / Part 1 - Hello Numpy
#
#   Author : Igor Janos
#            FIIT - Neural Networks Lectures
#
#------------------------------------------------------------------------------
import numpy as np

# 1. List -> NP.Array
a = [1, 2, 3, 4, 5]
print(a)

na = np.array(a, dtype=np.float)


print(na)
print('Shape is: ', na.shape)

na=na.T
print(na)
print('Shape is: ', na.shape)
















