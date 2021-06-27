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

# # 4.1 확률과 통계 학습을 위한 사전준비

# +
import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
# %matplotlib inline

# %precision 3

np.random.seed(0)
# -

# # 4. 2. 확률

# ## 4. 2. 1. 수학적 확률

# +
# 주사위 던지기 결과값을 배열로 저장
dice_data = np.array([1, 2, 3, 4, 5, 6])

print('숫자 하나만 무작위로 추출: ', np.random.choice(dice_data, 1))
# -

# ![image.png](attachment:image.png)

# ## 4. 2. 2. 통계적 확률

# +
# 주사위를 1000회 던짐
calc_steps = 1000

# 1 ~ 6의 숫자 중에서 1000회 추출 시행
dice_rolls = np.random.choice(dice_data, calc_steps)

# 각 숫자가 추출되는 횟수의 비율을 계산
for i in range(1, 7):
    p = len(dice_rolls[dice_rolls == i]) / calc_steps
    print(i, '가 나올 확률', p)
# -

# ## 4. 2. 5. 베이즈 정리
# ![image.png](attachment:image.png)

# +
# 연습문제 4-1
coin_data = np.array([0, 1])
calc_steps = 1000

coin_toss = np.random.choice(coin_data, calc_steps)

for i in range(0, 2):
    p = len(coin_toss[coin_toss == i]) / calc_steps
    print(i, '가 나올 확률', p)

# +
#연습문제 4-2
lottery_data = np.array([0, 1])
lottery_steps = 1000
lottery_proportion = np.array([0.9, 0.1])

lottery = np.random.choice(lottery_data, lottery_steps, p = lottery_proportion)

for i in range(0, 2):
    p = len(lottery[lottery == i]) / lottery_steps
    print(i, '가 나올 확률', p)
    
# A가 당첨되고 B가 당첨될 확률은?
A_lottery_proportion = np.array([(1000-100)/1000, 100/1000])
B_lottery_proportion = np.array([((1000-1)-(100-1))/(1000-1), (100-1)/(1000-1)])

print('A가 당첨되고 B가 당첨될 확률은 ', A_lottery_proportion[1] * B_lottery_proportion[1])
# -

# ![image.png](attachment:image.png)

# +
#연습문제 4-3
# 0: 음성, 1: 양성
p_x = np.array([0.999, 0.001])
p_x_test_x = np.array([0.01, 0.99])
p_not_x_test_x = np.array([0.97, 0.03])

p_test_x_x = (p_x_test_x[1] * p_x[1]) / ((p_x_test_x[1] * p_x[1]) + (p_not_x_test_x[1] * p_x[0]))
print(p_test_x_x)
# -

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # 4. 3. 확률변수와 확률분포

# ## 4. 3. 1. 확률변수, 확률함수, 분포함수, 기댓값
# ![image.png](attachment:image.png)

# ## 4. 3. 2. 다양한 분포함수

# +
# 균등분포
# 주사위를 1000회 던짐
calc_steps = 1000

dice_rolls = np.random.choice(dice_data, calc_steps)

prob_data = np.array([])
for i in range(1, 7):
    p = len(dice_rolls[dice_rolls == i]) / calc_steps
    prob_data = np.append(prob_data, len(dice_rolls[dice_rolls == i]) / calc_steps)

plt.bar(dice_data, prob_data, alpha = 0.2, facecolor = 'green', edgecolor = 'red')
plt.grid(True)

# +
# list와 array의 차이
x = np.array([1, 2, 3, 4, 5])
y = [6, 7, 8, 9, 10]

print(x * 2, y * 2)

# +
# 내장형 append
a = []

for i in range(1, 11):
    a.append(i)
print('내장형', a)

# np.append는 차원이 같아야만 붙일 수 있다
b = []

for i in range(1, 11):
    b = np.append(b, i)
print('numpy', b)

# +
# 베르누이분포
# head: 0, tail: 1
prob_be_data = np.array([])
coin_data = np.array([0, 0, 0, 0, 0, 1, 1, 1])

# unique로 유일한 값 추출
for i in np.unique(coin_data):
    p = len(coin_data[coin_data == i]) / len(coin_data)
    print(i, '가 나올 확률', p)
    prob_be_data = np.append(prob_be_data, p)

plt.bar([0, 1], prob_be_data, align = 'center')
plt.xticks([0, 1], ['head', 'tail'])
plt.grid(True)
# -

# align은 edge도 가능하다
# xticks는 x 좌표의 눈금을 꾸미는 것
plt.bar([0, 1], prob_be_data, align = 'edge', fc = 'yellow', ec = 'purple')  
plt.xticks([0, 1], ['head', 'tail'], rotation=20)
y_step = np.arange(0, 1, 0.1)
plt.yticks(y_step, y_step)
plt.grid(True)

help(plt.xticks)

help(plt.bar)

# 이항분포
np.random.seed(0)
# 시행횟수, 확률, 샘플수
# n회 시행하는 동안 확률 p로 발생하는 사건의 횟수를 반환
x = np.random.binomial(30, 0.5, 1000)
plt.hist(x)
plt.grid(True)

# https://angeloyeo.github.io/2021/04/23/binomial_distribution.html
# - 이항분포 그래프 설명 보기 좋음

# 푸아송분포
# 사건이 발생하길 기대되는 횟수, 샘플수
x = np.random.poisson(7, 1000)
plt.hist(x)
plt.grid(True)

# 정규분포
# 평균, 표준편차, 샘플 수
x = np.random.normal(5, 10, 10000)
plt.hist(x)
plt.grid(True)

# 로그 정규분포
# 평균, 표준편차, 샘플 수
x = np.random.lognormal(30, 0.4, 1000)
plt.hist(x)
plt.grid(True)

# ## 4. 3. 3. 커널 밀도함수

student_data_math = pd.read_csv('./chap3/student-mat.csv', sep = ';')
student_data_math

# 커널 밀도함수
student_data_math.absences.plot(kind = 'kde', style = 'r--')

# 단순 히스토그램 density = True로 지정시, 확률로 표시
student_data_math.absences.hist(density = True)
plt.grid(True)

# 커널 밀도함수
student_data_math.absences.plot(kind = 'kde', style = 'g--')
# 단순 히스토그램 density = True로 지정시, 확률로 표시
student_data_math.absences.hist(density = True, color = 'yellow')
plt.grid(True)

# 연습문제 4-4
x_sampling = [np.random.normal(0, 1, 100).mean() for i in range(10000)]
plt.hist(x_sampling)
plt.grid(True)

# 연습문제 4-5
x_sampling = [np.random.lognormal(0, 1, 100).mean() for i in range(10000)]
plt.hist(x_sampling)
plt.grid(True)

# +
# 연습문제 4-6

# 커널 밀도함수
student_data_math.G1.plot(kind = 'kde', style = 'k--')
# 단순 히스토그램 density = True로 지정시, 확률로 표시
student_data_math.G1.hist(density = True)
plt.grid(True)
# -
# # 4. 4. 심화학습: 다차원확률분포

# ## 4. 4. 1. 결합확률분포와 주변확률분포

# ## 4. 4. 2. 조건부 확률 함수와 조건부 기댓값

# ## 4. 4. 3. 독립과 연속분포

# +
# 2차원 정규분포 시각화
import scipy.stats as st
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# 데이터 설정
x, y = np.mgrid[10:100:2, 10:100:2]

# 메모리 초기화 하지 않으면 쓰레기값이 들어가있다
pos = np.empty(x.shape + (2, ))

pos[:, :, 0] = x
pos[:, :, 1] = y
# -

# 찾아보니 x와 y의 차원을 똑같이 맞춰주고 싶을 때, 격자를 만들 때 유용하다고 한다.
# 알겠는데, 그래서 왜 좋은지 잘 모르겠다.. 나중에 더 공부해야지.
# 근데 세로가 45개인건 오케이, 가로는 왜 45야? 정방행렬 맞춰줄 필요는 없는데.. 왜지?
help(np.mgrid)

# 메모리 초기화 하지 않으면 쓰레기값이 들어가있다
p = np.empty(x.shape + (2, ))
p

# +
# 다차원정규분포
# 각 변수의 평균과 분산공분산행렬 설정
# x와 y의 각 평균이 50과 0이고, [[100, 0], [0, 100]]은 x와 y의 공분산행렬입니다
rv = multivariate_normal([50, 50], [[100, 0], [0, 100]])

# 확률밀도함수
z = rv.pdf(pos)

# +
fig = plt.figure(dpi = 100)

ax = Axes3D(fig)
ax.plot_wireframe(x, y, z, color = 'red')

# x, y, z 레이블 설정
ax.set_xlabel('x')
ax.set_xlabel('y')
ax.set_zlabel('f(x, y)')

# z축 단위 변경, sci는 지수 표시를 의미, axis로 축 설정
# scilimits = (n, m)는 n부터 m 사이 밖의 값은 지수 표기 의미
# scilimits = (0, 0): 모두 지수로 표기한다는 의미
ax.ticklabel_format(style = 'sci', axis = 'z', scilimits = (0, 0))
# -

# # 4. 5. 추론 통계학

# ## 4. 5. 1. 대수 법칙

# +
# 대수의 법칙
# 계산(주사위 던지기) 횟수
calc_times = 1000

# 주사위
sample_array = np.array([1, 2, 3, 4, 5, 6])
number_cnt = np.arange(1, calc_times + 1)

# 4개의 선 생성
for i in range(4):
    p = np.random.choice(sample_array, calc_times).cumsum()
    plt.plot(p / number_cnt)
    plt.grid(True)


# -

# ## 4. 5. 2. 중심 극한 정리

# 중심 극한 정리
def function_central_theory(N):
    
    sample_array = np.array([1, 2, 3, 4, 5, 6])
#     numaber_cnt = np.arange(1, N + 1) * 1.0
    
    mean_array = np.array([])
    
    for i in range(1000):
        cum_variables = np.random.choice(sample_array, N).cumsum() * 1.0
        mean_array = np.append(mean_array, cum_variables[N-1] / N)
    
    plt.hist(mean_array)
    plt.grid(True)


function_central_theory(3)

function_central_theory(6)

function_central_theory(10 ** 3)

# ## 4. 5. 3. 표본분포

# 카이 제곱 분포
# 자유도 2, 10, 60을 따르는 카이제곱분포가 생성하는 난수의 히스토그램
for df, c in zip([2, 10, 60], 'bgr'):
    x = np.random.chisquare(df, 1000)
    plt.hist(x, 20, color = c)
    plt.grid(True)

# 카이 제곱 분포
# 자유도 2, 10, 60을 따르는 카이제곱분포가 생성하는 난수의 히스토그램
# plt.hist안에 있는 10은 bins인가?
for df, c in zip([2, 10, 60], 'bgr'):
    x = np.random.chisquare(df, 1000)
    plt.hist(x, 10, color = c)
    plt.grid(True)

help(plt.hist)

# 스튜던트 t 분포
x = np.random.standard_t(5, 1000)
plt.hist(x)
plt.grid(True)

help(np.random.standard_t)

# F 분포
for df, c in zip([(6, 7), (10, 10), (20, 25)], 'bgr'):
    x = np.random.f(df[0], df[1], 1000)
    plt.hist(x, 100, color = c, alpha = 0.7)
    plt.grid(True)

help(np.random.f)

# +
# 연습문제 4-7
# 자유도 5, 25, 50인 카이제곱분포를 따르는 난수를 각 1000개씩 생성하고 히스토그램을 그리세요

for df, c in zip([5, 25, 50], 'ymc'):
    x = np.random.chisquare(df, 1000)
    plt.hist(x, 20, color = c)
    plt.grid(True)

# +
# 연습문제 4-8
# 자유도 100인 t 분포를 따르는 난수를 각 1000개씩 생성하고 히스토그램을 그리세요

x = np.random.standard_t(100, 1000)
plt.hist(x)
plt.grid(True)

# +
# 연습문제 4-9
# 자유도 (10, 30), (20, 25)인 F 분포를 따르는 난수를 각 1000개씩 생성하고 히스토그램을 그리세요

for df, c in zip([(10, 30), (20, 25)], 'cr'):
    x = np.random.f(df[0], df[1], 1000)
    plt.hist(x, 100, color = c)
    plt.grid(True)
# -

# # 4. 6. 통계적 추정

# ### 연습문제 4-10
# ![image.png](attachment:image.png)

# ### 연습문제 4-11
# ![image.png](attachment:image.png)

# +
# 연습문제 4-12
# -

# # 4. 7. 통계적 검정

# +
# 수학 성적 데이터 읽어 오기
student_data_math = pd.read_csv('./chap3/student-mat.csv', sep = ';')

# 포루투갈어 성적 데이터 읽어 오기
student_data_por = pd.read_csv('./chap3/student-por.csv', sep = ';')

# 결합
student_data_merge = pd.merge(student_data_math,
                              student_data_por,
                              on = ['school', 'sex', 'age', 'address', 'famsize',
                                    'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
                                    'nursery', 'internet'],
                              suffixes = ('_math', '_por'))

print('G1 수학 성적 평균: ', student_data_merge.G1_math.mean())
print('G1 포르투갈어 성적 평균: ', student_data_merge.G1_por.mean())
# -

# ## 4. 7. 1. 검정

from scipy import stats
t, p = stats.ttest_rel(student_data_merge.G1_math, student_data_merge.G1_por)
print('p값 = ', p)

# ## 4. 7. 3. 빅데이터 검정

# +
# 연습문제 4-13
# 3장에서 이용한 데이터에서 수학과 포르투갈어 성적(G2) 평균에 차이가 있다고 할 수 있을까요? G3는 어떤가요?

# G2 평균 비교
print('G2 수학 성적 평균: ', student_data_merge.G2_math.mean())
print('G2 포르투갈어 성적 평균: ', student_data_merge.G2_por.mean())

t2, p2 = stats.ttest_rel(student_data_merge.G2_math, student_data_merge.G2_por)
print('G2 p값: ', p2)

print('\n')

# G3 평균 비교
print('G3 수학 성적 평균: ', student_data_merge.G3_math.mean())
print('G3 포르투갈어 성적 평균: ', student_data_merge.G3_por.mean())

t3, p3 = stats.ttest_rel(student_data_merge.G3_math, student_data_merge.G3_por)
print('G3 p값: ', p3)

print('p2와 p3 모두 매우 작으므로, 수학과 포르투갈어 성적은 G2와 G3 모두 유의한 차이가 있습니다')

# +
# 4장 종합문제
# stduent_data_merge를 이용해 아래의 질문에 답하세요
# 1. 각 결석일 수는 차이가 있다고 할 수 있을까요?

# +
# 4장 종합문제
# stduent_data_merge를 이용해 아래의 질문에 답하세요
# 2. 각 공부시간은 어떨까요?