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

# # 3.1. 통계의 종류

# +
import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

# 시각화 라이브러리
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set()
# %matplotlib inline

# 소수점 세 번째 자리 숫자까지 표시
# %precision 3
# -

from sklearn import linear_model

# # 3. 2. 데이터 입력과 기본 분석

# ## 3. 2. 1. 인터넷 등에 올라 있는 데이터를 읽어 들이기

pwd

# mkdir chap3

# cd ./chap3

# zip 파일과 파일을 다운로드 하기 위한 라이브러리
import requests, zipfile
from io import StringIO
import io

# +
# 데이터 url 지정
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00356/student.zip'

# 데이터를 url에서 받기
r = requests.get(url, stream = True)

# zipfile을 읽어들여 압축풀기
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()
# -

# ls

student_data_math = pd.read_csv('student-mat.csv')

student_data_math.head()

DataFrame.head(student_data_math)

student_data_math = pd.read_csv('student-mat.csv', sep = ';')

student_data_math.head()

# ?pd.read_csv

student_data_math.info()

student_data_math.shape

# ## 3. 2. 4. 정량 데이터와 정성 데이터

student_data_math['sex'].head()

student_data_math['absences'].head()

student_data_math.groupby('sex')['age'].mean()

# # 3.3. 기술통계

# ## 3. 3. 1. 히스토그램

# +
# histogram, 변수 데이터 지정
plt.hist(student_data_math['absences'], color = 'purple')

# x축과 y축의 레이블
plt.xlabel('absences')
plt.ylabel('count')

# 그리드 추가
plt.grid(True)
# -



# ## 3. 3. 2. 평균, 중앙값, 최빈값
#

# +
# 평균값
print('평균값: ', student_data_math['absences'].mean())

# 중앙값: 중앙값으로 데이터를 나누면 중앙값 앞뒤 데이터 수가 동일하고(데이터의 중앙에 위치한 값), 이상값의 영향을 덜 받는다.
print('중앙값: ', student_data_math['absences'].median())

# 최빈값: 가장 빈도가 많은 값
print('최빈값: ', student_data_math['absences'].mode())

# +
# 분산
print('분산: ', student_data_math['absences'].var(ddof = 0))

# 표준편차
print('표준편차: ', student_data_math['absences'].std(ddof = 0))

# 표준편차
print('표준편차: ', np.sqrt(student_data_math['absences'].var(ddof = 0)))
# -

# ## 3. 3. 4. 요약 통계량과 백분위수

# 요약 통계량
print('요약 통계량:\n', student_data_math['absences'].describe())
# 요약 통계량 [1]
print('\n요약 통계량 [1]: ', student_data_math['absences'].describe()[1])
# 사분위범위(75% 백분위수 값 - 25% 백분위수 값)
print('\n사분위범위: ', student_data_math['absences'].describe()[6] - student_data_math['absences'].describe()[4])

student_data_math['absences'].describe().shape

a, b, c, d, e, f, g, h = student_data_math['absences'].describe()

# 한꺼번에 요약 통계량 계산
student_data_math.describe()

# ## 3. 3. 5. 박스플롯 그래프

# 박스플롯: G1
plt.boxplot(student_data_math['G1'])
plt.grid(True)

# 박스플롯: 결석일 수
plt.boxplot(student_data_math['absences'])
plt.grid(True)

# 박스플롯: G1, G2, G3
plt.boxplot([student_data_math['G1'], student_data_math['G2'], student_data_math['G3']])
plt.grid(True)

# ## 3. 3. 6. 변동계수

# +
# 변동계수: 결석일 수
print('변동계수: ', student_data_math['absences'].std() / student_data_math['absences'].mean())

# 변동계수: 전체
print('\n변동계수:\n', student_data_math.std() / student_data_math.mean())
# -

# ## 3. 3. 7. 산점도와 상관계수

# +
# 산점도
plt.plot(student_data_math['G1'], student_data_math['G3'], 'o', color = 'green')

# 레이블
plt.ylabel('G3 grade')
plt.xlabel('G1 grade')
plt.grid(True)
# -

# 공분산
np.cov(student_data_math['G1'], student_data_math['G3'])

# 분산
print('G1의 분산: ', student_data_math['G1'].var())
print('G3의 분산: ', student_data_math['G3'].var())

# 피어슨 상관계수, p값
sp.stats.pearsonr(student_data_math['G1'], student_data_math['G3'])

# 스피어만 상관계수, p값
sp.stats.spearmanr(student_data_math['G1'], student_data_math['G3'])

# 켄달 상관계수, p값
sp.stats.kendalltau(student_data_math['G1'], student_data_math['G3'])

# 상관행렬
np.corrcoef([student_data_math['G1'], student_data_math['G3']])

# ## 3. 3. 8. 모든 변수의 히스토그램과 산점도 그리기

sns.pairplot(student_data_math[['Dalc', 'Walc', 'G1', 'G3']])
plt.grid(True)

# 주말에 술을 마시는 사람의 1학기 성적 평균값
student_data_math.groupby('Walc')['G1'].mean()

# 연습문제 3-1
student_data_por = pd.read_csv('student-por.csv', sep = ';')
student_data_por.describe()

# 연습문제 3-2
student_data = pd.merge(student_data_math, student_data_por,
                        on = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
                              'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
                              'nursery', 'internet'],
                        suffixes = ('_math', '_por'))
student_data.describe()

student_data

# 연습문제 3-3
sns.pairplot(student_data[['Medu', 'Fedu', 'G3_math']])
plt.grid(True)

# # 3. 4. 단순회귀분석

# +
# 산점도
plt.plot(student_data_math['G1'], student_data_math['G3'], 'o', color = 'green')

# 레이블
plt.ylabel('G3 grade')
plt.xlabel('G1 grade')
plt.grid(True)
# -

# ## 3. 4. 1. 선형회귀분석

# +
from sklearn import linear_model

# 선형회귀 인스턴스 생성
reg = linear_model.LinearRegression()

# +
# 설명변수는 1학기 수학 성적
X = student_data_math.loc[:, ['G1']].values

# 목표변수는 최종 수학 성적
Y = student_data_math['G3'].values

# 예측 모델 계산, a, b 산출
reg.fit(X, Y)

# 회귀계수
print('회귀계수: ', reg.coef_)

# 절편
print('절편: ', reg.intercept_)

# +
plt.scatter(X, Y)
plt.xlabel('G1 grade')
plt.ylabel('G3 grade')

# 위의 그래프에 선형 회귀선을 추가
plt.plot(X, reg.predict(X), color = 'red')
plt.grid(True)
# -

# ## 3. 4. 2. 결정계수

# 결정계수, 기여율, 교과서는 0.9이상, 실무에서는 0.64도 사용가능한 수준
print('결정계수: ', reg.score(X, Y))

# +
# 연습문제 3-4

from sklearn import linear_model

# 선형회귀 인스턴스 생성
reg = linear_model.LinearRegression()

X = student_data_por.loc[:, ['G1']].values
Y = student_data_por['G3'].values

# 예측 모델 계산, a, b 산출
reg.fit(X, Y)

# 회귀계수, 절편, 결정계수
print('회귀계수: ', reg.coef_)
print('절편: ', reg.intercept_)
print('결정계수: ', reg.score(X, Y))

# +
# 연습문제 3-5

plt.scatter(X, Y)
plt.plot(X, reg.predict(X), color = 'red')

plt.xlabel('G1 grade')
plt.ylabel('G3 grade')
plt.grid(True)

# +
# 연습문제 3-6


from sklearn import linear_model

# 선형회귀 인스턴스 생성
reg = linear_model.LinearRegression()

X = student_data_por.loc[:, ['absences']].values
Y = student_data_por['G3'].values

# 예측 모델 계산, a, b 산출
reg.fit(X, Y)

# 회귀계수, 절편, 결정계수
print('회귀계수: ', reg.coef_)
print('절편: ', reg.intercept_)
print('결정계수: ', reg.score(X, Y))

plt.scatter(X, Y)
plt.plot(X, reg.predict(X), color = 'red')

plt.xlabel('G1 grade')
plt.ylabel('G3 grade')
plt.grid(True)

# +
# 종합문제 3-1

# 데이터 url 지정
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

# 데이터를 url에서 받기
wine = pd.read_csv(url, sep = ';')

wine.head()
# -

# 종합문제 3-1-1
wine_describe = wine.describe()
wine_describe.to_csv('wine_describe.csv')

wine_describe

# ls

wine_col = wine.columns
wine_col

# 종합문제 3-1-2
sns.pairplot(wine[wine.columns])
plt.grid(True)

# 종합문제 3-1-2
sns.pairplot(wine[wine_col])
plt.grid(True)

# +
# 종합문제 3-2-1
# 1학기 수학 성적 데이터를 성별 오름차순으로 정렬하라
# 가로축은 인원의 누적 비율
# 세로축은 1학기 성적 누적 비율

# 위의 곡선을 로렌츠 곡선이라고 한다
# 로렌츠 곡선으로 성별로 나누어 1학기 수학성적으로 시각화하라
# -
student_data_math.head()

# +
s_d_m_f = student_data_math[student_data_math.sex == 'F']
s_d_m_m = student_data_math[student_data_math.sex == 'M']

s_d_m_f_G1_sorted = s_d_m_f.G1.sort_values()
s_d_m_m_G1_sorted = s_d_m_m.G1.sort_values()

F_G1_sorted_cumratio = s_d_m_f_G1_sorted.cumsum() / s_d_m_f_G1_sorted.sum()
M_G1_sorted_cumratio = s_d_m_m_G1_sorted.cumsum() / s_d_m_m_G1_sorted.sum()

F_num = range((1, len(s_d_m_f_G1_sorted) + 1) / len(s_d_m_f_G1_sorted) + 1)
M_num = range((1, len(s_d_m_m_G1_sorted) + 1) / len(s_d_m_m_G1_sorted) + 1)


plt.plot(F_num, F_G1_sorted_cumratio, label = 'F', color = 'green')
plt.plot(M_num, M_G1_sorted_cumratio, label = 'M', color = 'yellow')

plt.legend()
plt.grid(True)
# -







