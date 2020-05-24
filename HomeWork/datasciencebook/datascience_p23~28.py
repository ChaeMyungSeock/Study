# p23~28 리스트, 튜플, 딕셔너리 

# 리스트

'''
리스트는 순서가 있는 자료의 집합(collection)이라고 볼 수 있다. (다른 언어에서 보통 배열(array)이라고 하는 것과 유사하지만,
리스트의 기능이 조금 더 풍부하다.)
'''

integer_list = [1,2,3]
heterorgenous_list = ["string", 0.1, True]
list_of_lists = [integer_list, heterorgenous_list,[]]

list_length = len(integer_list)     # 결과는 3
list_sum = sum(integer_list)        # 결과는 6

print(list_length,list_sum )

'''
대괄호를 사용해 리스트의 n번째 값을 불러오거나 설정할 수 있다.
'''
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
zero = x[0]         # 결과는 0, 리스트의 index 순서는 0부터 시작
nine = x[-1]        # 결과는 9, 리스트의 마지막 항목을 불러옴
eight = x[-2]       # 결과는 8, 리스트의 항목중 뒤에서 두 번째 항목을 불러옴
x[0] = -1           # x = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]

'''
또한 리스트는 대괄호를 이용하여 슬라이싱할 수도 있다. i:j는 리스트를 i번째 값부터 j-1번째 값까지 분리하라는 의미이다.
만약 i를 따로 명시해주지 않는다면 리스트의 첫 번째 값부터 나눈다는 것을 의미한다. 반면 j를 명시해주지 않는다면 리스트의
마지막 값까지 나눈다는 것을 의미한다.
'''
first_three = x[:3]                     #[-1, 1, 2]
three_to_end = x[3:]                    #[3, 4, 5, 6, 7, 8, 9]
one_to_four = x[1:5]                    #[ 1, 2, 3, 4]
last_three = x[-3:]                     #[7, 8, 9]
without_first_and_last = x[1:-1]        #[1, 2, 3, 4, 5, 6, 7, 8]
copy_of_x = x[:]                        #[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]

'''
동일한 방식으로 리스트뿐 아니라 문자열 같은 순차형(sequen-tial) 변수를 나눌 수 있다.
또한 간격(stride)을 설정하여 리스트를 분리할 수도 있다. 참고로 간격은 음수로도 설정할 수 있다.
'''

every_third = x[::3]            # [-1, 3, 6, 9]
five_to_three = x[5:2:-1]       # [5, 4, 3]

'''
파이썬에서 제공하는 in 연산자를 사용하면 리스트 안에서 항모의 존재 여부를 확인할 수 있다.
이 방법은 리스트의 항목을 하나씩 모두 확인해 보기 때문에 리스트의 크기가 작을 때만(혹은 확인하는 데 걸리는 시간이 
상관없다면) 사용하도록 하자.
'''

1 in [1, 2, 3]      # True  (참)
0 in [1, 2, 3]      # False (거짓)

'''
주어진 리스트에 다른 리스트를 추가 해주기
'''
x = [1, 2, 3]
x.extend([4,5,6])       # x는 이제 [1, 2, 3, 4, 5, 6]
x = [1, 2, 3]
y = x + [1, 2, 3]       # y는 이제 [1, 2, 3, 4, 5, 6] x는 변하지 않음

'''
주로 리스트에 항목을 하나씩 추가하는 경우가 많은데 한번 보자
'''
x = [1, 2, 3]
x.append(0)             # x는 이제 [1, 2, 3, 0]
y = x[-1]               # y = 0
z = len(x)              # z = 4

# 만약 리스트 안에 몇 개의 항목이 존재하는지 알고 있다면 손쉽게 리스트를 풀 수 있다.
x, y = [1, 2]           # x=1, y=2 // 하지만 양쪽 항목의 개수가 다르다면 ValueError가 발생
_, y = [1, 2]           # y==2이고 첫 번째 항목은 신경쓰지 않는다.


#튜플

'''
튜플(tuple)은 변경할 수 없는 리스트이다. 리스트에서 수정을 제외한 모든 기능을 튜플에 적용할 수 있다. 
튜플은 대괄호 대신 괄호를 사용해서 (혹은 아무런 기호 없이) 정의한다.
'''

my_list = [1,2]
my_tuple = (1,2)
other_tuple = 3,4
my_list[1] = 3                  #이제 my_list = [1,3]


# 함수에서 여러 값을 반환할 때 튜플을 사용하면 편하다

def sum_and_product(x, y):
    return (x+y), (x*y)

sp = sum_and_product(2,3)       # 결과는 (5,6)
s, p = sum_and_product(5, 10)   # s는 15, p는 50

'''
튜플과 리스트는 다중 할당(multiple assignment)을 지원한다.
'''
x,y = 1,2                       # x는 1, y는 2
x,y = y,x                       # x는 2, y는 1

# 딕셔너리

'''
딕셔너리(dict, dictionary, 사전)는 파이썬의 또 다른 기본적인 데이터 구조이며, 특정 값(value)과 
연관된 키(key)를 연결해 주고 이를 사용해 값을 빠르게 검색할 수 있다.
'''

empty_dict = {}                    # 가장 파이썬스러운 딕셔너리 선언
empty_dict2 = dict()               # 같은 딕셔너리 선언
grades = {"baptist" : 80, "Seock" : 95}

# 대괄호를 사용해서 키의 값을 불러올 수 있다.
baptist_grade = grades["baptist"]  # baptist라는 키값으로 불러온 value값의 결과는 80

# 만약 딕셔너리 안에 존재하지 않는 키를 입력하면 KeyError가 발생, 연산자 in을 사용하면 키의 존재 여부를 확인할 수 있다.

bapitst_has_grade = "baptist" in grades     # True
hell_has_grade = "hell" in grades           # False

'''
크기가 굉장히 큰 딕셔너리에서도 키의 존재 여부를 빠르게 확인할 수 있다. 딕셔너리에서 get 매서드를 
사용하면 입력한 키가 딕션너리에 없어도 에러를 반환하지 않고 기본값을 반환해준다.
'''

baptist_grade = grades.get("baptist", 0)    # 결과는 80 가지고 있던 기본값을 반환
hell_grade = grades.get("hell", 0)          # 결과는 None

# 또한 대괄호를 사용해서 키와 값을 새로 지정해 줄 수 있다.

grades["baptist"] = 99                      # 기존의 값을 대체
grades["kate"] = 100                        # 세 번째 항목을 추가
num_students = len(grades)                  # 결과는 3

# 특정 키 대신 딕셔너리의 모든 키를 한번에 살펴볼 수 있다.

grades_key = grades.keys()                  # 키에 대한 리스트
grades_values = grades.values()             # 값에 대한 리스트
grades_items = grades.items()               # (key, value) 튜플에 대한 리스트

"baptist" in grades_key                     # True, 하지만 리스트에서 in을 사용하기 때문에 느림
"baptist" in grades                         # 딕셔너리에서 in을 사용하기 때문에 빠름


# defaultdict

from collections import defaultdict
'''
문서에서 단어의 빈도수를 세어 보는 주잉라고 상상해보자. 가장 직관적인 방법은 단어를 키로, 빈도수를 
값으로지정하는 딕셔너리를 생성하는 것이다. 이때, 각 단어가 딕셔너리에 이미 존재하면 값을 증가시키고
존재하지 않는다면 새로운 키와 값을 추가해주면 됨
'''

word_counts = {}                # word_counts를 딕셔너리라고 선언
for word in document:
    if word in word_counts:
        word_counts[word] += 1  # 이미 있는 단어면 1증가
    else:
        word_counts[word] = 1   # 새로운 단어라면 새로 1만들어줌

# 예외처리 방식으로 딕셔너리 생성하기'''
'''
word_counts = {}                # word_counts를 딕셔너리라고 선언
for word in document:
    try:
        word_counts[word] += 1  # 이미 있는 단어면 1증가
    except KeyError:
        word_counts[word] = 1   # 새로운 단어라면 새로 1만들어줌

'''
# 존재하지 않는 키를 적절하게 처리해 주는 get을 사용해서 딕셔너리를 생성하는 방법
'''
word_counts = {}                # word_counts를 딕셔너리라고 선언
for word in document:
    previous_count = word_counts.get(word, 0)
    word_counts[word] = previous_count + 1
'''

'''
위의 세가지 방법 모두 약간 복잡함. defaultdict와 평범한 딕셔너리의 유일한 차이점은 만약 존재하지
않는 키가 주어진다면 defaultdict는 이 키와 인자에서 주어진 값으로 dict에 새로운 항목을 추가해 준다는
것이다. defaultdict를 사용하기 위해서는 먼저 collections 모듈에서 defaultdict를 불러와야 한다
'''



word_counts = defaultdict(int)                # int()는 0을 생성
for word in document:
    word_counts[word] += 1

# 리스트, 딕셔너리 혹은 직접 만든 함수를 인자에 넣어 줄 수 있다.

dd_list = defaultdict(list)                   # list()는 빈 리스트를 생성
dd_list[2].append(1)                          # 이제 dd_list는 {2: [1]}를 포함

dd_dict = defaultdict(dict)                   # dict()는 빈 딕셔너리를 생성
dd_dict["Joel"]["City"] = "Seattle"           # {"Joel" : {"City" : "Seattle"}}

dd_pair = defaultdict(lambda : [0,0])
dd_pair[2][1] = 1                             # 이제 dd_pair는 {2: [0,1]}을 포함 => 기본적으로 [0, 0]인데 
                                              # dd_pair[2][1] = 1이면 dd_pair[2][0] = 0 (기본값이 0이므로)




