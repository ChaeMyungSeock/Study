# 모듈화
import re as regex # re이라는 모듈을 regex라고 표현하겠다.
# re는 정규표현식(regular expression, regex)을 다룰 때 필요한 다양한 함수와 상수를 포함

from collections import defaultdict, Counter # 모듈의 필요한 특정 기능만 불어와서 사용하겠다


# 함수


def double(x):
    '''
    이곳은 함수에 대한 설명을 적음
    파이썬 함수들은 변수로 할당되거나 함수의 인자로 전달할 수 있다는 점에서 일급 함수의 특성을 지님
    '''
    return x*2 # return x(1) 인자가 1인 함수 x를 호출
def apply_to_one(f):
    return f(1)

my_double = double # 방금 정의한 함수를 나타냄
x = apply_to_one(my_double)
print(x)
print(my_double)


# 람다 함수
'''
람다함수의 경우 람다함수안에 새로운 변수를 만들 수 없다. 바깥의 변수를 사용할 순 있다
새로운 변수를 사용해야 할 경우 def를 이용하자. 람다함수는 일회용 함수식이라고 생각하자.
'''
y = apply_to_one(lambda x: x+4) #5
print(y)

# another_double = lambda x : x*2 # 이 방법은 최대한 피하도록 하자 

def another_double(x):
    ''' 대신 이렇게 작성하자  '''
    return 2*x

'''
함수의 인자에는 기본값을 할당할 수 있는데, 기본값 외의 값을 전달하고 싶을 때는 값을 직접 명시
'''
def my_print(message = "my default message"):
    print(message)

my_print()          # 디폴트 값인  "my default message"를 출력
my_print("hello")   #'hello'를 출력

def full_name(first = "What's-his-name", last = "Something"):
    return first + " " + last
    


print(full_name("MyungSeock","baptist"))

full_name(last= "baptist") # "what's-his-name batist" 출력

