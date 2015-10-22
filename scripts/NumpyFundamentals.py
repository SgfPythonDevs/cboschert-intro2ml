__author__ = 'Chad'
import numpy as np

data = [1, 2, 3, 4, 5]

arr = np.array(data)

arr2d = np.arange(40).reshape(5, 8)

arr3d = np.arange(30).reshape(2, 3, 5)

print(arr2d)
print(arr2d[2])
print(arr2d[2, 3])
print(arr2d[2, 3:])
print(arr2d[0::, 3])

bool_index = np.array([False, False, True, False, True])
print(arr2d[bool_index, 0])