
gradient = lambda x: 2*x-4

# 매개변수 : return할 값을 적어줍니다
# lambda 람다 정의에는 "return"문이 포함되어 있지 않습니다. 반환값을 만드는 표현식이 있습니다.
#  함수가 사용될 수 있는 곳에는 어디라도 람다 정의를 넣을 수 있으며, 위의 예 처럼 변수에 할당하여 사용할 필요는 없습니다


def gradient2(x):
    temp = 2*x -4
    return temp

x = 3

print(gradient(x))
print(gradient2(x))
