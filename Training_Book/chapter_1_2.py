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

# if문
findvalue = 5
if findvalue in data_list:
    print('{0}은 들어 있습니다.'.format(findvalue))
else:
    pnt('{0}은 들어 있지 않습니다.'.format(findvalue))
print('여기서부터는 if 문과 관계없이, 반드시 표시됩니다.')

# 1. {0}은 무슨 뜻인가?
# 2. .format()는 무엇인가?
#     - {0}은 format() method의 첫 번째 인수를 의미한다
# 3. 추리해보자면, 저 안에 뒤에있는 것이 들어간다는 것으로 보이는데, 왜 저런 방법을 사용했는가?
#     - 서식을 정해서 출력하고 싶을 때 사용한다고 한다
# 4. 만약 변수가 여러개면 어떻게 되는가?
#     - {0}, {1}, ... 와 format(findvalue, findvalue, ...) 등으로 여러 개 표현이 가능하다
#     - 혹은 {} {} 으로 표시하면, 순서대로 나온다.
#     - format(findvalue = 5) 등 method 내에서 변수 선언이 가능하다
# 5. 다른 방식으로 표현하는 방법은 없는가?
#     - {:6s}, {07d:0}, {0:^09d}, 등 공백 혹은 문자열을 앞 뒤로 넣을 수 있다.
#     - 세 번째는 0번째의 변수를 가운데 정렬한다는 의미가 추가되었다.
#     - %d 방식은 오래된 방식이라 없어질 가능성이 농후하다...!

findvalue = 3
if findvalue in data_list:
    print('{0}은 들어 있습니다.'.format(findvalue))
else:
    pirnt('{0}은 들어 있지 않습니다.'.format(findvalue))
print('여기서부터는 if 문과 관계없이, 반드시 표시됩니다.')

print('{0}와 {1}을 더하면 {2}입니다.'.format(2, 3, 5))

# for문
total = 0
for num in [1, 2, 3]:
    print('num: ', num)
    total = total + num
print('total: ', total)


for dic_key in dic_data:
    print(dic_key, dic_data[dic_key])

for i in range(11):
    print(i)

for i in range(1, 11, 2):
    print(i)

for key, value in dic_data.items():
    print(key, value)

data_list1 = []
data_list1 = [i * 2 for i in data_list]
print(data_list1)

data_list2 = []
data_list2 = [i * 2 for i in data_list if i % 2 == 0]
print(data_list2)

for one, two in zip([1, 2, 3], [11, 12, 13]):
    print(one, '과', two)

# 1. one, two 대신 다른 숫자를 사용할 수 있는가?
#     - three를 사용해도 두번째 것이 나온다
# 2. 리스트를 여러 개 사용해도 되는가?
#     - 세 번째 리스트를 삽입하자, 2개까지 expected 값이라며 에러가 뜬다

# while문
num = 1
while num <= 10:
    print(num)
    num = num + 1
print('마지막 값은 {0}입니다.'.format(num))


# # 1.2.5 Function

def calc_multi(a, b):
    return a * b


calc_multi(3, 10)


def calc_fib(n):
    if n == 1 or n == 2:
        return 1
    else:
        return calc_fib(n - 1) + calc_fib(n - 2)


print('피보나치 수: ', calc_fib(10))


def calc_fib2(n):
    if n > 2:
        return calc_fib2(n - 2) + calc_fib2(n - 1)
    else:
        return 1


calc_fib2(10)

(lambda a ,b: a * b)(3, 10)


def calc_double(x):
    return x * 2


for num in [1, 2, 3, 4]:
    print(calc_double(num))

list(map(calc_double, [1, 2, 3, 4]))

list(map(lambda x: x * 2, [1, 2, 3, 4]))

from functools import reduce

reduce(lambda sum, x: sum + x, [1, 2, 3, 4], 0)

list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4]))

# 연습 문제 1-1
string = 'Hello, Data Science!'
for i in range(len(string)):
    print(string[i])

# 연습 문제 1-2
sum = 0
for i in range(1, 51):
    sum = sum + i
    i += 1
print("1부터 50까지의 합은 ", sum)


# # 1.2.6 Class and Instance

# +
class PrintClass:
    def print_me(self):
        print(self.x, self.y)

p1 = PrintClass()
p1.x = 10
p1.y = 100
p1.z = 1000

p1.print_me()


# -

class MyCalcClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def calc_add1(self, a, b):
        return a + b
    def calc_add2(self):
        return self.x + self.y
    def calc_multi(self, a, b):
        return a * b
    def calc_print(self, a):
        print('data : {0} : y의 값 {1}'.format(a, self.y))


instance_1 = MyCalcClass(1, 2)
instance_2 = MyCalcClass(5, 10)

print('2개의 수를 더함(새로운 숫자를 인수로 지정):', instance_1.calc_add1(5, 3))
print('2개의 수를 더함(인스턴스화될 때 값):', instance_1.calc_add2())
print('2개의 수를 곱함:', instance_1.calc_multi(5, 3))
instance_1.calc_print(5)

print('2개의 수를 더함(새로운 숫자를 인수로 지정):', instance_2.calc_add1(10, 3))
print('2개의 수를 더함(인스턴스화될 때 값):', instance_2.calc_add2())
print('2개의 수를 곱함:', instance_2.calc_multi(4, 3))
instance_2.calc_print(20)


class MyCalcClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def calc_add1(self, a, b):
        return a + b
    def calc_add2(self):
        return self.x + self.y
    def calc_minus1(self, a, b):
        return a + b
    def calc_minus2(self):
        return self.x - self.y
    def calc_multi(self, a, b):
        return a * b
    def calc_print(self, a):
        print('data : {0} : y의 값 {1}'.format(a, self.y))


instance_3 = MyCalcClass(3, 8)
print('2개의 수를 더함(새로운 숫자를 인수로 지정):', instance_3.calc_add1(10, 7))
print('2개의 수를 더함(인스턴스화될 때 값):', instance_3.calc_add2())
print('2개의 수를 뺌(새로운 숫자를 인수로 지정):', instance_3.calc_minus1(10, 7))
print('2개의 수를 뺌(인스턴스화될 때 값):', instance_3.calc_minus2())
print('2개의 수를 곱함:', instance_3.calc_multi(4, 3))
instance_3.calc_print(13)

# 종합문제
# 1. 10까지의 소수를 출력하는 프로그램을 작성하세요. 여기서 소수는 약수가 1과 자기자신뿐인 양의 정수를 말합니다.
prime = range(2, 11)
for i in range(2, 10 // 2 + 1):
    prime = [j for j in prime if (j == i or j % i != 0)]
print(prime)

# +
a = range(2, 11)
b = []
for i in range(2, 10 // 2 + 1):
    b = [j for j in a if (j == i or j % i != 0)]
print(b)

'''
바꾸면 왜 안되는가? 더 알아보기..
'''


# -

# 종합문제
# 2. 위의 예제를 일반화해서, 자연수 N까지의 소수를 출력하는 함수를 작성하세요.
def calc_prime(n):
    prime = range(2, n + 1)
    for i in range(2, n // 2 + 1):
        prime = [j for j in prime if (j == i or j % i != 0)]
    print(prime)


calc_prime(100)
