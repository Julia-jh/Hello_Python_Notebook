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
import numpy.random as random

# +
# 리스트 생성하기
list_kind = int(input('리스트 종류를 선택해 주십시오 \n1. 랜덤 \n2. 순서대로 \n'))
list_num = int(input('리스트의 갯수를 정해주십시오: '))

data = np.arange(1, list_num + 1)
if list_kind == 1:
    random.shuffle(data)
elif list_kind == 2:
    data = data
print('생성된 리스트입니다', data)

# +
# 인덱싱이랑 정렬 바꾸는 함수 만들기....언젠간....
# -

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
