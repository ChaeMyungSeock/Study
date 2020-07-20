from math import gcd
def solution(w,h):
    answer = 1
    a = w*h
    b = gcd(w,h)
    # y = h/w(x)로 생각하면 나눠지는데 최대 공배수로 나눠짐
    c = w//b
    d = h//b

    answer = a - b*(c+d-1)
    # 결국 최대 공배수로 나눠논 칸에서는 서로소의 배수로 또 하나의 사각형이 만들어짐
    # 계산해보니 c+d-1의 규칙을 보임

    
    return int(answer)