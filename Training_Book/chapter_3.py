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
# -

# ## 3. 3. 3. 분산과 표준편차

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

# 공분산
np.cov([student_data_math['G1'], student_data_math['G2'], student_data_math['G3']])

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

# 상관행렬
np.corrcoef([student_data_math['G1'], student_data_math['G2'], student_data_math['G3']])

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

# 연습문제 3-3
sns.pairplot(student_data[['Medu', 'Fedu']])
plt.grid(True)

help(sns.pairplot)

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

# 종합문제 3-1-2
sns.pairplot(wine[wine.columns])
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
# 학생성적을 성별 오름차순으로 정렬하기
student_data_math_F = student_data_math[student_data_math.sex == 'F']
student_data_math_M = student_data_math[student_data_math.sex == 'M']

F_G1_sorted = student_data_math_F.G1.sort_values().reset_index(drop=True)
M_G1_sorted = student_data_math_M.G1.sort_values().reset_index(drop=True)

# 가로축의 인원의 누적 비율
F_num = np.arange(len(F_G1_sorted)) / len(F_G1_sorted)
M_num = np.arange(len(M_G1_sorted)) / len(M_G1_sorted)

# 세로축의 1학기 성적 누적 비율
F_G1_sorted_cumratio = F_G1_sorted.cumsum() / F_G1_sorted.sum()
M_G1_sorted_cumratio = M_G1_sorted.cumsum() / M_G1_sorted.sum()

# 각 그래프 그리기
plt.plot(F_num, F_G1_sorted_cumratio, label = 'F', color = 'green')
plt.plot(M_num, M_G1_sorted_cumratio, label = 'M', color = 'yellow')

plt.xlabel('cumulative population ratio')
plt.ylabel('GI cumulative ratio')
plt.legend()
plt.grid(True)
# +
# 종합문제 3-2-2
# 불평등 정도를 수치로 나타낸 것을 지니 계수라고 한다
# 지니계수값은 로렌츠 곡선과 45도 선으로 둘러싸인 부분의 면적의 2배로 정의 되며, 0에서 1사이의 값이다
# 값이 클수록 불평등 정도가 커진다
# GI = sum of i(sum of j(abs((xi - xj)/(2*n**2*mean(x)))))
# 남녀의 1학기 성적에 대한 지니 계수를 구하라

# 분모
den_F = 2 * (len(F_G1_sorted) ** 2) * np.mean(F_G1_sorted)
den_M = 2 * (len(M_G1_sorted) ** 2) * np.mean(M_G1_sorted)

def num_x(x):
    num_x = []
    for i in range(0, len(x)):
        for j in range(0, len(x)):
            num_x.append(abs(x[i] - x[j]))
    return sum(num_x)
        
# 분자
num_F = []
for i in range(0, len(F_G1_sorted)):
    for j in range(0, len(F_G1_sorted)):
        num_F.append(abs(F_G1_sorted[i] - F_G1_sorted[j]))

num_M = []
for i in range(0, len(M_G1_sorted)):
    for j in range(0, len(M_G1_sorted)):
        num_M.append(abs(M_G1_sorted[i] - M_G1_sorted[j]))

GI_F = sum(num_F) / den_F
GI_M = sum(num_M) / den_M

print('여학생의 지니계수: ',GI_F)
print('남학생의 지니계수: ',GI_M)
# -

# ## 나중에 더 해본 문제들!

# 심심해서 해봄.. 되나 싶어가지고...
x = student_data_math.index
y = student_data_math['sex'].isin(['F'])
plt.scatter(x[y == True], y[y == True], color = 'green')
plt.scatter(x[y == False], y[y == False], color = 'blue')

# +
# 학생성적을 성별 오름차순으로 정렬하기
# 모양이 예쁘지는 않지만, 어디까지 되나 확인해보려고 해봄
# groupby(x)로 나눈 다음에, get_group(y)을 하면 x에서 y인 값을 추출할 수 있다
# 마찬가지로 그 중에서, ['G1']을 따로 추출할 수 있는데(이때 시리즈로 추출됨)
# ['G1']을 groupby뒤에 넣어도 되고, get_group 뒤에 넣어도 실행이 된다
# 값을 기준으로 정렬하고 싶을 때는 sort_values()를 사용한다
# 인덱스를 리셋해주는 것이, 나중에 인덱스를 활용하기에 좋다
student_data_math_F = student_data_math.groupby('sex')['G1'].get_group('F').sort_values().reset_index(drop=True)
student_data_math_M = student_data_math.groupby('sex')['G1'].get_group('M').sort_values().reset_index(drop=True)

# 가로축의 인원의 누적 비율
# 인덱스를 활용해서 만들어봄
# 너무 길어짐..
num_F = []
for i in range(0, len(student_data_math_F)):
    num_F.append((student_data_math_F.index[i] + 1) / (student_data_math_F.index.max() + 1))

num_M = []
for i in range(0, len(student_data_math_M)):
    num_M.append((student_data_math_M.index[i] + 1) / (student_data_math_M.index.max() + 1))

# 세로축의 1학기 성적 누적 비율
F_G1_sorted_cumratio = F_G1_sorted.cumsum() / F_G1_sorted.sum()
M_G1_sorted_cumratio = M_G1_sorted.cumsum() / M_G1_sorted.sum()

# 각 그래프 그리기
plt.plot(F_num, F_G1_sorted_cumratio, label = 'F', color = 'green')
plt.plot(M_num, M_G1_sorted_cumratio, label = 'M', color = 'yellow')

plt.xlabel('cumulative population ratio')
plt.ylabel('GI cumulative ratio')
plt.legend()
plt.grid(True)


# +
# 종합문제 3-2-2
# 불평등 정도를 수치로 나타낸 것을 지니 계수라고 한다
# 지니계수값은 로렌츠 곡선과 45도 선으로 둘러싸인 부분의 면적의 2배로 정의 되며, 0에서 1사이의 값이다
# 값이 클수록 불평등 정도가 커진다
# GI = sum of i(sum of j(abs((xi - xj)/(2*n**2*mean(x)))))
# 남녀의 1학기 성적에 대한 지니 계수를 구하라

# 위에서 하다보니, 두 번 쓰지 않고 활용하 수 있을 것이라고 생각함
# 분모와 분자를 따로 구했다
# 추출한 DF는 활용에 좋기 위해 인덱스를 꼭 리셋하자!!!

def GI(x):

    # 분모
    def den_x(x):
        return 2 * (len(x) ** 2) * np.mean(x)

    # 분자
    def num_x(x):
        num_x = []
        for i in range(0, len(x)):
            for j in range(0, len(x)):
                num_x.append(abs(x[i] - x[j]))
        return sum(num_x)
    return num_x(x) / den_x(x)

print('여학생의 지니계수: ', GI(F_G1_sorted))
print('남학생의 지니계수: ', GI(M_G1_sorted))
