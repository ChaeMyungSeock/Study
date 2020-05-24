# 정렬
'''
파이썬의 모든 리스트에는 리스트를 자동으로 정렬해 주는 sort 매서드가 있다. 
'''

x = [4, 1, 2, 3]
y = sorted(x)       # y는 [1, 2, 3, 4]
x.sort()            # x는 [1, 2, 3, 4]

'''
기본적으로 sort 매서드와 sorted 함수는 리스트의 각 항목을 일일이 비교해서 오름차순으로 정렬
만약 리스트를 내림차수능로 정렬하고 싶다면 인자에 reverse=True를 추가해주면 된다. 그리고
리스트의 각 항목끼리 서로 비교하는 대신 key를 사용하면 지정한 함수의 결과값을 기준으로 
리스트를 정렬할 수 있다.
'''

# 절대값의 내림차순으로 리스트 정렬
x = sorted([-4, 1, -2, 3], key = abs, reverse=True)     # 결과는 [-4, 3, -2, 1]

# 빈도의 내림차순으로 단어와 빈도를 정렬
'''
wc = sorted(word_counts.items(), 
            key = lambda word_and_count: word_and_count[1],
            reverse = True)
'''

# 리스트 컴프리헨션
'''
기존의 리스트에서 특정 항목을 선택하거나 변환시킨 결과를 새로운 리스트에 저장해야 하는 
경우도 자주 발생한다.  => 리스트 컴프리헨션
'''

even_numbers = [x for x in range(5) if x % 2 == 0]              # [0, 2, 4]
squares = [x*x for x in range(5)]                               # [0, 1, 4, 9, 16]
even_squares = [x*x for x in even_numbers]                      # [0, 4, 16]

square_dict = {x:x*x for x in range(5)}                          # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
square_set = {x*x for x in [1, -1]}                             # {1}

# 보통 리스트에서 불필요한 값은 밑줄로 표기
zeros = [0 for _ in even_numbers]                               # even_numbers와 동일한 길이

# 리스트 컴프리헨션에는 여러 for를 포함할 수 있고

pairs = [(x,y)
         for x in range(10)
         for y in range(10)]                                    # (0,0) (0,1) ..... (9,8) (9,9) 총 100개

increasing_pairs = [(x, y)                                      # x < y인 경우만 해당
                    for x in range(10)                          # range(lo, hi)는
                    for y in range(x+1, 10)]                    # [lo, lo + 1, ..., hi - 1 ]을 의미한다.

# 자동테스트와 assert
'''
타입(type)이나 자동 테스트를 통해 코드가 제대로 작성되었는지 확인. assert는 지정된 조건이
충족되지 않는다면 ASSErtionError을 반환한다.
'''

assert 1 + 1 == 2
assert 1 + 1 == 2, "1 + 1 should equal 2 but didn't"

def smallest_item(xs):
    return min(xs)

assert smallest_item([10, 20, 5, 40]) == 5
assert smallest_item([1, 0 , -1, 2]) == -1

# assert를 통해서 인자를 검증할 수도 있다.

def smallest1_item(xs):
    assert xs, "empty list has no smallest item"
    return min(xs)

