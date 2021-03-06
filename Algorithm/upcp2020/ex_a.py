'''
 문제 A. 수학은 비대면강의입니다
 시간 제한 1 초
 메모리 제한 1024 MB
 
 다음 연립방정식에서 x와 y의 값을 계산하시오.
    ax+by = c
    dx+ey = f

버추얼 강의의 숙제 제출은 인터넷 창의 빈 칸에 수들을
입력하는 식이다. 각 칸에는 −999 이상 999 이하의 정수만 입력할 수 있다

입력

정수 a, b, c, d, e, f 가 공백으로 구분되어 차례대로 주어진다. (−999 ≤ a,b, c,d, e, f ≤ 999)
문제에서 언급한 방정식을 만족하는 (x, y)가 유일하게 존재하고, 이 때 x와 y가 각각 −999 이상 999 이하의 정수인
경우만 입력으로 주어짐이 보장된다.


출력

문제의 답인 x와 y를 공백으로 구분해 출력한다
'''


import sys


a,b,c,d,e,f = list(map(int,sys.stdin.readline().split()))

for x in range(-999,1000):
    for y in range(-999,1000):
        if(a*x + b*y == c and d * x + e * y == f):
            print(x,y)
