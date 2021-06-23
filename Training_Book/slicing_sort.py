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
import pandas as pd
import numpy.random as random


def list_indexing():
    
    # 리스트 생성하기
    def listing():
        list_kind = int(input('리스트 종류를 선택해 주십시오 \n1. 랜덤 \n2. 순서대로 \n3. 그만두기\n'))
        list_num = int(input('리스트의 갯수를 정해주십시오: '))

        # 다른 함수에서 사용해야 한다
        global data, data2
        data = list(np.arange(1, list_num + 1))
        # 재사용할 경우 사용할 임시 저장소
        data2 = data

        if list_kind == 1:
            random.shuffle(data)
        elif list_kind == 3:
            quit()
            print('종료합니다.')
        print('생성된 리스트입니다', data)
        

    # 인덱싱이랑 정렬 바꾸는 함수
    def indexing_sorting():
        indexings = list(map(int, input('인덱싱 a:b:c에 들어갈 숫자를 ,로 구분하며 차례대로 입력하시오\n이때 비워두고 싶은 경우 0을 사용하시오\n').split(',')))

        for i in range(3):
            if indexings[i] == 0:
                indexings[i] = None

        data[indexings[0]:indexings[1]:indexings[2]].sort()
        print("data[{}:{}:{}].sort() = {}".format(indexings[0], indexings[1], indexings[2], data))

        # 데이터를 비교하기
        def compare():
            if change_num > 0:
                data = data2
                indexing_sorting()

    # 인덱싱 바꾸고 싶은 횟수
    change_num = int(input('인덱싱을 몇 번 바꾸고 싶나요?\n'))

    while(change_num != -1):
        if change_num == 0:
            listing()
            indexing_sorting()
        elif change_num > 0:
            listing()
            for i in range(change_num):
                indexing_sorting()
                print('\n')
# 특정 상황일 때 특정 수만큼 반복하는것 짜기!!!    

list_indexing()

type(data)

data = [ 5,  1,  4,  3, 10,  9,  6,  2,  7,  8]
data

a = data[::3]
type(a)

data[::-3]

data[::3].sort()
data

data[::-35].sort()
data
