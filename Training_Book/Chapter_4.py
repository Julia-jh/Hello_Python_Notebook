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

# ## 4. 2. 2. 통계적 확률

# +
# 주사위를 1000회 던짐
calc_steps = 1000

# 1 ~ 6의 숫자 중에서 1000회  추출 시행
dice_rolls = np.random.choice(dice_data, calc_steps)

# 각 숫자가 추출되는 횟수의 비율을 계산
for i in range(1, 7):
    p = len(dice_rolls[dice_rolls == i]) / calc_steps
    print(i, '가 나올 확률', p)

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

# +
#연습문제 4-3
p_x = np.array([0.999, 0.001])
p_x_test_x = np.array([0.01, 0.99])
p_not_x_test_x = np.array([0.97, 0.03])

p_test_x_x = (p_x_test_x[1] * p_x[1]) / ((p_x_test_x[1] * p_x[1]) + (p_not_x_test_x[1] * p_x[0]))
print(p_test_x_x)
# -

# # 4. 3. 확률변수와 확률분포

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

plt.bar(dice_data, prob_data)
plt.grid(True)

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

# 이항분포
np.random.seed(0)
# 시행횟수, 확률, 샘플수
x = np.random.binomial(30, 0.5, 1000)
plt.hist(x)
plt.grid(True)

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
student_data_math.absences.plot(kind = 'kde', style = 'k--')

# 단순 히스토그램 density = True로 지정시, 확률로 표시
student_data_math.absences.hist(density = True)
plt.grid(True)

# 커널 밀도함수
student_data_math.absences.plot(kind = 'kde', style = 'k--')
# 단순 히스토그램 density = True로 지정시, 확률로 표시
student_data_math.absences.hist(density = True)
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









