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

data.shape

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

data = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
data

data = np.array([9, 2, 3, 4, 10, 6, 7, 8, 1, 5])
data[::-1].sort()
data

# 통계값
print('Min: ', data.min())
print('Max: ', data.max())
print('Sum: ', data.sum())
print('Cum: ', data.cumsum())
print('Ratio: ', data.cumsum() / data.sum())

np.min(data)

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

print(array1[:,:])

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

array1.sum()

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

sample_pandas_data.shape

sample_pandas_data = pd.Series([0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                              index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
print(sample_pandas_data)

sample_pandas_data.shape

print("데이터 값: ", sample_pandas_data.values)
print("인덱스 값: ", sample_pandas_data.index)

print(sample_pandas_data.values[2])
print(sample_pandas_data.index[5])

# ## 2. 4. 3. DataFrame 사용법

attri_data1 = {'ID': ['100', '101', '102', '103', '104'],
               'City': ['Seoul', 'Pusan', 'Daegu', 'Gangneung', 'Seoul'],
               'Birth_year': [1900, 1989, 1992, 1997, 1982],
               'Name': ['Junho', 'Heejin', 'Mijung', 'Minho', 'Steve']}
attri_data_frame1 = DataFrame(attri_data1)
print(attri_data_frame1)

attri_data_frame_index1 = DataFrame(attri_data1, index = ['a', 'b', 'c', 'd', 'e'])
print(attri_data_frame_index1)

attri_data_frame_index1.shape
# dict는 안됨...

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

a = attri_data_frame1[attri_data_frame1['City'] == 'Seoul']['ID']
a

attri_data_frame1['City'] == 'Seoul'

attri_data_frame1[attri_data_frame1['City'].isin(['Seoul', 'Pusan'])]

attri_data_frame1[attri_data_frame1['Birth_year'] < 1990]

# ## 2. 4. 6. 데이터 삭제와 결합

# 열과 행 삭제
attri_data_frame1.drop(['Birth_year'], axis = 1)
# inplace = TRUE: 대체하게 하는 것

attri_data_frame1.drop([0], axis = 0)

attri_data_frame1.drop([2])

# 데이터 결합
attri_data2 = {'ID': ['100', '101', '102', '105', '107'],
               'Math': [50, 43, 33, 76, 98],
               'English': [90, 30, 20, 50, 30],
               'Sex': ['M', 'F', 'F', 'M', 'M']}
attri_data_frame2 = DataFrame(attri_data2)
attri_data_frame2

pd.merge(attri_data_frame1, attri_data_frame2)

# ## 2. 4. 7. 집계

attri_data_frame2.groupby('Sex')['Math'].mean()

a = attri_data_frame2.groupby('English')['Math'].mean()
print(a)
b = attri_data_frame2.groupby('Sex')['English'].max()
print(b)

# ## 2. 4. 8. 정렬

attri_data2 = {'ID': ['100', '101', '102', '103', '104'],
               'City': ['Seoul', 'Pusan', 'Daegu', 'Gangneung', 'Seoul'],
               'Birth_year': [1900, 1989, 1992, 1997, 1982],
               'Name': ['Junho', 'Heejin', 'Mijung', 'Minho', 'Steve']}
attri_data_frame2 = DataFrame(attri_data2)
attri_data_frame_index2 = DataFrame(attri_data2, index = ['e', 'b', 'a', 'd', 'c'])
attri_data_frame_index2

# index 기준으로 정렬
attri_data_frame_index2.sort_index()

# 값을 기준으로 정렬, 필터는 오름차순ㄴ
attri_data_frame_index2.Birth_year.sort_values()

# ## 2. 4. 9. nan(null) 판정

# 값이 있는지 확인
attri_data_frame_index2.isin(['Seoul'])

# 결측값 처리 방법
# name을 모두 nan으로 변경
attri_data_frame_index2['Name'] = np.nan
attri_data_frame_index2.isnull()

pd.isnull(attri_data_frame_index2)

attri_data_frame_index2.isnull().sum()

# +
# 연습문제 2-7
from pandas import Series, DataFrame
import pandas as pd

attri_data1 = {'ID': ['1', '2', '3', '4', '5'],
               'Sex': ['F', 'F', 'M', 'M', 'F'],
               'Money': [1000, 2000, 500, 300, 700],
               'Name': ['Suzy', 'Minji', 'Taeho', 'Jinsung', 'Suyoung']}
attri_data_frame1 = DataFrame(attri_data1)
# -

attri_data_frame1[attri_data_frame1['Money'] > 500]['Name']

# 연습문제 2-8
attri_data_frame1.groupby('Sex').Money.mean()

# 연습문제 2-9
attri_data2 = {'ID': ['3', '4', '7'],
               'Math': [60, 30, 40],
               'English': [80, 20, 30]}
attri_data_frame2 = DataFrame(attri_data2)

attri_data_frame = attri_data_frame1.merge(attri_data_frame2)
attri_data_frame

print('합친 데이터프레임에서 Money 평균: ', attri_data_frame.Money.mean())
print('합친 데이터프레임에서 Math 평균: ', attri_data_frame.Math.mean())
print('합친 데이터프레임에서 English 평균: ', attri_data_frame.English.mean())

# # 2. 5. 매트플롯립 기초

# +
# 매트플롯립과 씨본 불러오기
# 씨본은 그래프를 더 보기 좋게 만듦
import matplotlib as mpl
import seaborn as sns

# pyplot은 plt이라는 이름으로 실행 가능
import matplotlib.pyplot as plt

# 주피터 노트북에서 그래프를 표시하기 위해 필요한 매직 명령어
# %matplotlib inline
# -

# ## 2. 5. 2. 산점도

# +
# 산점도

# 시드 값 선정
random.seed(0)

# x축 데이터
x = np.random.randn(30)

# y축 데이터
y = np.sin(x) + np.random.randn(30)

# 그래프 크기 지정
plt.figure(figsize=(20, 6))

# 그래프 생성
plt.plot(x, y, 'o')

# 타이틀
plt.title('Title Name')

# X 좌표 이름
plt.xlabel('X')

# Y 좌표 이름
plt.ylabel('Y')

# grid 표시
plt.grid(True)

# +
# 산점도

# 시드 값 선정
random.seed(0)

# x축 데이터
x = np.random.randn(30)

# y축 데이터
y = np.sin(x) + np.random.randn(30)

# 그래프 크기 지정
plt.figure(figsize=(20, 6))

# 산점도 다른 방식
plt.scatter(x, y)

# 타이틀
plt.title('Title Name')

# X 좌표 이름
plt.xlabel('X')

# Y 좌표 이름
plt.ylabel('Y')

# grid 표시
plt.grid(True)

# +
# 연속형 곡선

# 시드 값 선정
random.seed(0)

# 데이터 범위
numpy_data_x = np.arange(1000)

# 난수 발생과 누적 합계
numpy_random_data_y = np.random.randn(1000).cumsum()

# 그래프 크기 지정
plt.figure(figsize=(20, 6))

# label = 과 legend로 레이블 표시 가능하다
plt.plot(numpy_data_x, numpy_random_data_y, label = 'Label')
plt.legend()

# X 좌표 이름
plt.xlabel('X')

# Y 좌표 이름
plt.ylabel('Y')

# grid 표시
plt.grid(True)
# -

# ## 2. 5. 3. 그래프 분할

# +
# 그래프 크기 지정
plt.figure(figsize = (20, 6))

# 2행 1열 그래프의 첫 번째
plt.subplot(2, 1, 1)

x = np.linspace(-10, 10, 100)
plt.plot(x, np.sin(x))

# 2행 1열 그래프의 두 번째
plt.subplot(2, 1, 2)

y = np.linspace(-10, 10, 100)
plt.plot(y, np.sin(2*y))

plt.grid(True)


# -

# ## 2. 5. 4. 함수 그래프 그리기

# 함수 정의
def my_function(x):
    return x ** 2 + 2 * x +1
x = np.arange(-10, 10)
plt.figure(figsize = (20, 6))
plt.plot(x, my_function(x))
plt.grid(True)

# ## 2. 5. 5. 히스토그램

# +
# 시드 지정
random.seed(0)

# 그래프 크기 지정
plt.figure(figsize = (20, 6))

# 히스토그램 생성
plt.hist(np.random.randn(10 ** 5) * 10 + 50,
         bins = 60, range = (20, 80))

plt.grid(True)


# -

# ?plt.hist

# +
# 연습문제 2-10

def my_function(x):
    return 5 * x + 3
x = np.linspace(-10, 10, 100)
plt.figure(figsize = (20, 6))
plt.plot(x, my_function(x))
plt.grid(True)


# +
# 연습문제 2-11

def my_function1(x):
    return np.sin(x)

def my_function2(x):
    return np.cos(x)


x = np.linspace(-10, 10, 100)
plt.figure(figsize = (20, 6))
plt.plot(x, my_function1(x), color = 'darkgreen')
plt.plot(x, my_function2(x), color = 'yellowgreen')
plt.grid(True)

# +
# 연습문제 2-12

random.seed(0)
data_x = np.random.uniform(0.0, 1.0, 1000)
data_y = np.random.uniform(0.0, 1.0, 1000)

plt.figure(figsize=(20, 6))

plt.subplot(2, 1, 1)
plt.hist(data_x,
         bins = 100, range = (0, 1),
         color = 'purple')
plt.title('LALALA')
plt.xlabel('X')
plt.ylabel('score')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.hist(data_y,
         bins = 1000, range = (0, 1),
         color = 'pink')
plt.title('HAHAHA')
plt.xlabel('Y')
plt.ylabel('score')
plt.grid(True)

plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
# -

# 2장 종합문제
# 1. 구간 [0, 1]에서 균등분포 따르는 난수를 두 번 발생 시켜, 각 10000개씩 생성해보자
data_1 = np.random.uniform(0.0, 1.0, 10000)
data_2 = np.random.uniform(0.0, 1.0, 10000)

# +
# 2장 종합문제
# 2. x-y축에서 중심이 (0, 0)이고 반경 1인 원과 길이가 1인 정사각형을 생각해보자, 이때 x, y 조합으로 만든 난수 10000개 중
# 원 내부에 들어가는 점은 몇 개인가?
# 유클리드 놈: sqrt(x^2 + y^2)을 사용하여 길이를 구한다
# 파이썬에서는 math.hypot(x, y)로 구할 수 있다
# 원 안과 밖에 있는 점을 함께 그래프로 그려보자
import math

x = np.random.uniform(0.0, 1.0, 10000)
y = np.random.uniform(0.0, 1.0, 10000)

x_in = []
y_in = []
x_out = []
y_out = []

radius = 1
count = 0

for i in range(0, 10000):
    dist = math.hypot(x[i], y[i])
    if dist < radius:
        count += 1
        x_in.append(x[i])
        y_in.append( y[i])
    else:
        x_out.append(x[i])
        y_out.append(y[i])        


plt.scatter(x_in, y_in, color = 'green')
plt.scatter(x_out, y_out, color = 'blue')

x_cir = np.arange(0, 1, 0.001)
y_cir = np.sqrt(1-x_cir**2)
plt.plot(x_cir, y_cir, color = 'red')

plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

print("원 안의 점의 개수: ", count)
print("원 밖의 점의 개수: ", 1000 - count)
# -

# 2장 종합문제
# 3. 반경 1인 원의 면적 1/4과 길이 1인 정사각형의 면적 비율은 pi/4 : 1이 되는데,
# 이 값과 위에서 구한 결과를 이용해 원주율을 구하라
print("원주율: ", 4 * count / 10000)
