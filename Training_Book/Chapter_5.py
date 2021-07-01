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

# # 5. 1. 개요와 사전준비

# ## 5. 1. 2. 라이브러리 임포트

# +
# 필요한 라이브러리 임포트
import numpy as np
import numpy.random as random
import scipy as sp

# 시각화 라이브러리
import matplotlib.pyplot as plt
import matplotlib as mpl
# %matplotlib inline

# 소수점 세 번째 자리까지 표시
# %precision 3
# -

# # 5. 2. 넘파이를 이용한 계산 방법

# ## 5. 2. 1. 인덱스 참조

# +
sample_array = np.arange(10)

# 원본 데이터
print('sample_array: ', sample_array)

# 앞에서 숫자 5개를 추출해 sample_array_slice에 저장(슴라이싱)
sample_array_slice = sample_array[0:5]
print('sample_array_slice: ', sample_array_slice)

# sample_array_slice 처음 3개 원소를 10으로 변경
sample_array_slice[0:3] = 10
print('변경한 sample_array_slice: ', sample_array_slice)

# 슬라이싱은 원본 리스트의 원소 값도 변경한다는 점에 주의
print('변경 후 sample_array: ', sample_array)

# 데이터 복사
print('\n데이터 복사')

# copy하고 다른 object를 생성
sample_array_copy = np.copy(sample_array)
print('sample_array_copy: ', sample_array_copy)

sample_array_copy[0:3] = 20
print('변경한 sample_array_copy: ', sample_array_copy)

# 원본 리스트의 원소는 변경되지 않음
print('sample_array: ', sample_array)

# +
# 불 인덱스 참조

# 데이터 생성
sample_names = np.array(['a', 'b', 'c', 'd', 'a'])
print('\n')
random.seed(0)
data = random.randn(5, 5)

print(sample_names)
print(data)

print('\n')

print('불값 출력', sample_names == 'a')
print('\n')
print('불 인덱스 참조\n', data[sample_names == 'a'])
print('\n')

# +
# 조건 제어
# 조건을 제어하기 위해 불 배열 생성
cond_data = np.array([True, True, False, False, True])

# 배열 x_array 생성
x_array = np.array([1,2,3,4,5])

# 배열 y_array 생성
y_array = np.array([100,200,300,400,500])

# 조건 제어 실행
print(np.where(cond_data, x_array, y_array))

# +
# 연습문제 5-1
# 불 인덱스를 이용해 b에 해당하는 데이터를 추출하시오
# 데이터 생성
sample_names = np.array(['a', 'b', 'c', 'd', 'a'])
print('\n')
random.seed(0)
data = random.randn(5, 5)

print(sample_names)
print(data)

print('\n')

print('b 불 인덱스 참조\n', data[sample_names == 'b'])

# +
# 연습문제 5-2

print('c 빼고 불 인덱스 참조\n', data[sample_names != 'c'])

# +
# 연습문제 5-3
# x: 3,4 , y: 1,2,5

x_array = np.array([1, 2, 3, 4, 5])
y_array = np.array([6, 7, 8, 9, 10])

cond_data = np.array([True, True, False, False, True])
print(np.where(cond_data, y_array, x_array))
# -

# ## 5. 2. 2. 넘파이를 이용한 연산 작업

# +
# 중복 삭제

cond_data = np.array([True, True, False, False, True])

# cond_data를 출력
print(cond_data)

# 중복 원소 삭제
print(np.unique(cond_data))
# -

# 범용함수
sample_data = np.arange(10)
print('원본 데이터: ', sample_data)
print('모든 원소의 제곱근: ', np.sqrt(sample_data))
print('모든 원소의 자연상수 지수함수: ', np.exp(sample_data))

# +
# 최소, 최대 ,평균, 합계 계산

# arange로 9개의 원소를 갖는 배열을 생성하고 reshape로 3*3 행렬로 재배열
sample_multi_array_data1 = np.arange(9).reshape(3,3)

print(sample_multi_array_data1)
print('\n')
print('최솟값: ', sample_multi_array_data1.min())
print('최댓값: ', sample_multi_array_data1.max())
print('평균: ', sample_multi_array_data1.mean())
print('합계: ', sample_multi_array_data1.sum())
print('\n')

# 행렬을 지정해 합계를 구함
print('행 합계: ', sample_multi_array_data1.sum(axis = 1))
print('열 합계: ', sample_multi_array_data1.sum(axis = 0))

# +
# 진릿값 판정

# 진릿값 판정 배열함수
cond_data = np.array([True, True, False, False, True])

print('True가 적어도 하나라도 있는가? ', cond_data.any())
print('모두 True인가? ', cond_data.all())
print('\n')

sample_multi_array_data1 = np.arange(9).reshape(3, 3)
print(sample_multi_array_data1)
print('5보다 큰 숫자가 몇 개인가? ', (sample_multi_array_data1 > 5).sum())

# +
# 대각성분 계산

# 행렬 연산
sample_multi_array_data1 = np.arange(9).reshape(3,3)
print(sample_multi_array_data1)

print('대각성분: ', np.diag(sample_multi_array_data1))
print('대각성분의 합: ', np.trace(sample_multi_array_data1))

# +
# 연습문제 5-4
# 다음 데이터에 대해 모든 원소의 제곱근 행렬을 출력하세요

sample_multi_array_data2 = np.arange(16).reshape(4,4)
print(sample_multi_array_data2)
print('\n')

print('모든 원소의 제곱근 행렬\n', np.sqrt(sample_multi_array_data2))
# -

# 연습문제 5-5
print('\n')
print('최솟값: ', sample_multi_array_data2.min())
print('최댓값: ', sample_multi_array_data2.max())
print('합계: ', sample_multi_array_data2.sum())
print('평균: ', sample_multi_array_data2.mean())
print('\n')

# 연습문제 5-6
print('대각성분의 합: ', np.trace(sample_multi_array_data2))

# ## 5. 2. 3. 배열의 조작과 브로드캐스트

# +
# 재배열

# 데이터 생성
sample_array = np.arange(10)
print(sample_array)
print('\n')

# 재배열
sample_array2 = sample_array.reshape(2,5)
print(sample_array2)
print('\n')

sample_array3 = sample_array.reshape(5,2)
print(sample_array3)

# +
# 데이터 결합

# 수직 방향 결합

# 데이터 생성
sample_array3 = np.array([[1, 2, 3], [4, 5, 6]])
sample_array4 = np.array([[7, 8, 9], [10, 11, 12]])
print(sample_array3)
print('\n')
print(sample_array4)

# 수직 방향으로 결합, 파라미터 axis에 0 지정
np.concatenate([sample_array3, sample_array4], axis = 0)
# -

# vstack을 이용한 수직 방향 결합
np.vstack((sample_array3, sample_array4))

# 수평 방향으로 결합, 파라미터 axis에 1 지정
np.concatenate([sample_array3, sample_array4], axis = 1)

# hstack을 이용한 수직 방향 결합
np.hstack((sample_array3, sample_array4))

# +
# 배열 분할

# 데이터 생성
sample_array3 = np.array([[1, 2, 3], [4, 5, 6]])
sample_array4 = np.array([[7, 8, 9], [10, 11, 12]])
sample_array_vstack = np.vstack((sample_array3, sample_array4))

# 생성한 데이터 sample_array_vstack 출력
sample_array_vstack
# -










# # 5. 3. 사이파이 응용

# ## 5. 3. 1. 보간법


# +
# x는 linspace를 이용해 0에서 10까지 11개의 원소를 갖는 등차수열
x = np.linsplace(1, 10, num = 11, endpoint = True)

# y값 생성
y = np.cos(x, y, )

# +
from scipy import linalg

A = np.identity(5)
A[0, :] = 1
A[:, 0] = 1
A[0, 0] = 5
b = np.ones(5)

# 정방행렬을 LU 분해함
(LU, piv) = sp.linalg.lu_factor(A)
L = np.identity(5) + np.tril(LU, -1)
U = np.triu(LU)
P = np.identity(5)[piv]

# 해를 구함
x = sp.linalg.lu_solve((LU, piv), b)
x
# -

help(np.tril)

help(np.conj)





from scipy import integrate
sp.integrate.quad(lambda x: 4/(1+x**2), 1, 5)








