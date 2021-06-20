# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np

data = np.array([9, 2, 3, 4, 10, 6, 7, 8, 1, 5])
data[::-1].sort()
data

data = np.array([9, 2, 3, 4, 10, 6, 7, 8, 1, 5])
data[::-2].sort()
data

data = np.array([9, 2, 3, 4, 10, 6, 7, 8, 1, 5])
data[::-3].sort()
data

data = np.array([9, 2, 3, 4, 10, 6, 7, 8, 1, 5])
data[::-4].sort()
data

data = np.array([9, 2, 3, 4, 10, 6, 7, 8, 1, 5])
data[::-5].sort()
data

data = np.array([9, 2, 3, 4, 10, 6, 7, 8, 1, 5])
data[::-6].sort()
data

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
data[::1].sort()
data

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
data[::2].sort()
data

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
data[::3].sort()
data

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
data[::4].sort()
data

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
data[::5].sort()
data

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
data[::6].sort()
data

data[::3]

np.sort(data)

data = np.array([0,1,2,3,4,5,6,7,8,9])
data[1:6:2].sort()
data

data = np.array([0,1,2,3,4,5,6,7,8,9])
data[1:6:-2].sort()
data

data = np.array([9, 2, 3, 6, 10, 4, 7, 8, 1, 5])
data[1:6:2].sort()
data

data[1:6:2]

data = np.array([9, 2, 3, 6, 10, 4, 7, 8, 1, 5])
data[1:6:-2].sort()
data

data = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
data[1:6:2].sort()
data

data = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
data[1:6:2]

data = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
data[1:6:-2]

data = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
data[1:6:-2].sort()
data

data = np.array(['b', 'e', 'a', 'i' ,'c', 'f', 'j', 'g', 'd', 'h'])
data[::1].sort()
data

data = np.array(['b', 'e', 'a', 'i' ,'c', 'f', 'j', 'g', 'd', 'h'])
data[::-1].sort()
data

data = np.array(['b', 'e', 'a', 'i' ,'c', 'f', 'j', 'g', 'd', 'h'])
data[::-2].sort()
data
