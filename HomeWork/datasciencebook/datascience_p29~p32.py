# Counter

'''
Counter는 연속된 값을 defaultdict(int)와 유사한 객체로 변환해 주며, 키와 값의 빈도를 연결시켜 준다.
'''

from collections import Counter
c = Counter([0, 1, 2, 0])           # c는 결국 {0 : 2, 1: 1, 2: 1}

# 특정 문서에서 단어의 개수를 셀 때도 유용하다
# word_counts = Counter(document)     # document가 단어의 리스트임을 상기


# Counter 객체에는 굉징히 유용하게 쓰이는 most_common 함수가 있다.

# 가장 자주 나오는 단어 10개와 이 단어들의 빈도수를 출력
'''
for word, count in word_counts.most_common(10)
    print(word, count)
'''
# Set
'''
집합(Set)은 파이썬의 데이터 구조 중 유일한 항목의 집합을 나타내는 구조다. 집합은 중괄호를 사용해서 정의한다.
'''

primes_below_10 = {2, 3, 5, 7}

# {}은 비어 있는 딕셔너리를 의미하기 때문에 set()을 사용해서 비어 있는 set을 생성할 수 있다.

s = set()
s.add(1)            # s는 이제 {1}
s.add(2)            # s는 이제 {1, 2}
s.add(2)            # s는 아직도 {1, 2}
x = len(s)          # 2
y = 2 in s          # True
z = 3 in s          # False

'''
in은 집합에서 굉장히 빠르게 작동한다. 수많은 항목 중에서 특정 항목의 존재 여부를 확인해 보기 위해서는
리스트를 사용하는 것보다 집합을 사용하는 것이 효율적
'''

stopwords_list = ["a", "an", "at"]  + hundreads_of_other_words + ["yet", "you"]
"zip" in stopwords_list                 # False, but 모든 항목을 확인해야함

stopwords_set = set(stopwords_list)
"zip" in stopwords_list                 # 굉장히 빠르게 확인 가능

# 중복된 원소 제거

item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list)              # 6
item_set = set(item_list)                # {1, 2, 3}
num_distinct_items = len(item_set)      # 3
distinct_item_list = list(item_set)     # [1, 2, 3]


# 흐름제어
'''
대부분의 프로그래밍 언어처럼 if를 사용하면 조건에 따라 코드를 제어할 수 있다.
'''
# 삼항 연산자인 if-then-else문을 한 줄로 표현할 수도 있다.

pairty = "even" if x % 2 == 0 else "odd"

# while 문
x = 0
while x<10:
    x +=1

# for 문
for x in range(10):
    if x == 3:
        continue        # 다음 경우로 넘어감
    
    if x==5:
        break           # for문을 전체끝냄
    print(x)

# True와 False
'''
True = 1
False = 0
'''


