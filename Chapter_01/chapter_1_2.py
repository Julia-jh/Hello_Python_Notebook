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

# # 1.2.1 How to use Jupyter Notebook

print('Hello, world!')

print(1+1)
print(2*5)
print(10**3)

# - press esc: to get out from editing mode
# - press H: to see the list of Keyboard shortcuts

# # 1.2.2 Python Basic

msg = 'test'
print(msg)
msg[0] # msg변수의 첫 번째 문자 추출하는 코드

data = 1
print(data)
data = data +10
print(data)

# 예약어 표시
__import__('keyword').kwlist

# 내장형 함수 표시
dir(__builtins__)

# # 1.2.3 List and Dictionary

# ## - List

data_list = [1, 2, 3, 4, 5 ,6, 7, 8, 9, 10]
print(data_list)
print('변수 형: ', type(data_list))
print('세 번째 원소: ', data_list[2])
print('원소의 개수: ', len(data_list))

# 리스트에 곱하기를 하면 그 수만큼 반복할 뿐
data_list * 2

# 리스트에 원소 추가하기
data_list.append(100)
print(data_list)

# 리스트 원소 삭제하기
data_list.remove(2)
print(data_list)

# 리스트 마지막 원소만 남기기
data_list.pop()

# 리스트의 n번째 요소 지우기
del data_list[2]
print(data_list)

# ## - Dictionary

dic_data = {'apple' : 100, 'banana' : 200,
            'orange' : 300, 'mango' : 400,
            'melon' : 500}
print(dic_data['melon'])

print(dic_data['orange'])

print(dic_data.keys())

print(dic_data.values())

print(dic_data.get('orange'))

print('apple' in dic_data)
print('pineapple' in dic_data)

print(dic_data['orange'] + dic_data['apple'])

# 추가
dic_data['pineapple'] = 600
print(dic_data)

# 수정
dic_data['orange'] = 350
print(dic_data)

# 삭제
del dic_data['orange']
print(dic_data)

# # 1.2.4 Conditional Statements and iteration



# # 1.2.5 Function

# # 1.2.6 Class and Instance
