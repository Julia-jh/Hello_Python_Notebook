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

# ## 2. 1. 1 Library Import

import numpy as np

from numpy import random

# ## 2. 1. 2. magic Command

# magic command 전체 리스트 확인하기
# %quickref

# # 2.2 넘파이

import numpy as np

# %precision 3

# ## 2.2.2. 배열 생성과 조작, 가공

# 배열
data = np.array([9, 2, 3, 4, 10, 6, 7, 8, 1, 5])
data

# 데이터 형
data.dtype

# dimension
print('차원수: ', data.ndim)
print('원소수: ', data.size)

# calculation
print(data * 2)
print('곱셈', np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]))
print('거듭제곱', np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) ** 2)
print('나눗셈', np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) / np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]))

# sort
print('현재 순서대로: ', data)
data.sort()
print('정렬 후: ', data)
data[::-1].sort()
print('역정렬 후: ', data)

# 통계값
print('Min: ', data.min())
print('Max: ', data.max())
print('Sum: ', data.sum())
print('Cum: ', data.cumsum())
print('Ratio: ', data.cumsum() / data.sum())

# ## 2.2.3 난수

# 난수
import numpy.random as random

random.seed(0)
rnd_data = random.rand(10)
print('10개의 난수 배열: ', rnd_data)
print('복원 추출: ', random.choice(data, 10))
print('비복원 추출: ', random.choice(data, 10, replace = False))

# - random.seed(): 괄호 안의 숫자를 0 이상의 정수를 넣으면 고정, 비워두면 비고정, 음수 혹은 소수점 아래를 넣으면 에러
# - %timeit np.sum(원하는 변수): 그 변수를 계산하는데 걸린 시간을 알 수 있다

# ## 2. 2. 4. 행렬

np.arange(9)

array1 = np.arange(9).reshape(3, 3)
print(array1, "\n")
print("첫 번째 행", array1[0,:], "\n")
print("첫 번째 열", array1[:,0])

# 행렬 연산
array2 = np.arange(9, 18).reshape(3,3)
print(array2, "\n")
print("행렬 곱셈", np.dot(array1, array2), "\n")
print("행렬 각 원소 간 곱", array1 * array2)

# 원소가 0 혹은 1인 행렬
print(np.zeros((2,3), dtype = np.int64), "\n")
print(np.ones((2,3), dtype = np.float64))

# 연습문제 2-1
array1 = np.arange(1, 51)
print(array1)
sum_array = np.sum(array1)
print("1부터 50까지 다 더한 값: ", sum_array)

# 연습문제 2-2
rnd_array = random.rand(10)
print(rnd_array)
print('Min: ', rnd_array.min())
print('Max: ', rnd_array.max())
print('Sum: ', rnd_array.sum())

# 연습문제 2-3
array3 = (np.ones((5, 5), dtype = np.float64)) * 3
print("행렬: ", array3, "\n")
print("행렬의 제곱: ", np.dot(array3, array3))

# # 2. 3. Scipy

import scipy.linalg as linalg
from scipy.optimize import minimize_scalar

# ## 2. 3. 2. 행렬연산

matrix = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
print('행렬식: ', linalg.det(matrix))
print('역행렬: \n', linalg.inv(matrix))
print('\n')
eig_value, eig_vector = linalg.eig(matrix)
print('고윳값: ', eig_value)
print('고유벡터: \n', eig_vector)


# ## 2. 3. 3. 뉴턴법

# 방정식 해 구하기
def my_function(x):
    return (x ** 2 + 2 * x + 1)


from scipy.optimize import newton
print(newton(my_function, 0))

# 최솟값 구하기
print(minimize_scalar(my_function, method = 'Brent'))


def my_function(x):
    return (x ** 4)
print("뉴턴법 해", newton(my_function, 0))
print('최솟값  구하기 \n', minimize_scalar(my_function, method = 'Brent'))

# - 최솟값 구하기는, 아래로 볼록한 모양의 함수에서만 가능하다. 그럼 구간을 정해서 최솟값 찾는 거는 어떻게 할 수 있는가?

# 연습문제 2-4
A = np.array([[1, 2, 3], [1, 3, 2], [3, 1, 2]])
print(A)
print("행렬식", linalg.det(A))

# 연습문제 2-5
print("역행렬", linalg.inv(A))
eig_value, eig_vector = linalg.eig(A)
print("고윳값: ", eig_value)
print("고유벡터 \n", eig_vector)


# 연습문제 2-6
def function_a(x):
    return(x ** 3 + 2*x + 1)
print(newton(function_a, 0))

# # 2.4 판다스 기초

import pandas as pd
from pandas import Series, DataFrame

# ## 2. 4. 2. Series 사용법

sample_pandas_data = pd.Series([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
print(sample_pandas_data)

sample_pandas_data = pd.Series([0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                              index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
print(sample_pandas_data)

print("데이터 값: ", sample_pandas_data.values)
print("인덱스 값: ", sample_pandas_data.index)

# ## 2. 4. 3. DataFrame 사용법

attri_data1 = {'ID': ['100', '101', '102', '103', '104'],
               'City': ['Seoul', 'Pusan', 'Daegu', 'Gangneung', 'Seoul'],
               'Birth_year': [1900, 1989, 1992, 1997, 1982],
               'Name': ['Junho', 'Heejin', 'Mijung', 'Minho', 'Steve']}
attri_data_frame1 = DataFrame(attri_data1)
print(attri_data_frame1)

attri_data_frame_index1 = DataFrame(attri_data1, index = ['a', 'b', 'c', 'd', 'e'])
print(attri_data_frame_index1)

# 주피터 환경에서 데이터 출력
attri_data_frame_index1

# ## 2. 4. 4. 행렬 다루기

# 전치
attri_data_frame1.T

# 특정 열 추출 (한 줄)
attri_data_frame1.Birth_year

# 특정 열 추출 (복수 줄)
attri_data_frame1[['ID', 'Birth_year']]

# ## 2. 4. 5. 데이터 추출

attri_data_frame1[attri_data_frame1['City'] == 'Seoul']

attri_data_frame1['City'] == 'Seoul'

attri_data_frame1[attri_data_frame1['City'].isin(['Seoul', 'Pusan'])]

attri_data_frame1[attri_data_frame1['Birth_year'] < 1990]

# ## 2. 4. 6. 데이터 삭제와 결합


















