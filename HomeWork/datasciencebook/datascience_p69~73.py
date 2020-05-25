# 통계

# 중심경향성
'''
데이터의 중심이 어디 있는지를 나타내는 중심 경향성(central tendency) 지표는 매우 중요하다.
그리고 대부분의 경우, 데이터의 값을 데이터 포인트의 개수로 나눈 평균을 사용하게 된다.
'''

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

'''
만약 데이터 포인트가 두 개라면 평균은 두 데이터 포인트의 정중앙에 위치한 값일 것이다. 데이터의 개수를 추가할수록, 평균은 각 데이터 포인트의 값에
따라 이동하게 된다. 예를 들어 10개의 데이터 포인트 중 아무 데이터 하나만 1을 증가시켜도 평균은 0.1이 증가한다.
 가끔은 중앙값(median)도 필요할 것이다. 데이터 포인트의 개수가 홀수라면 중앙값은 전체 데이터에서 가장 중앙에 있는 데이터 포인트를 의미한다.
 반면 데이터 포인트의 개수가 짝수라면 중앙값은 전체 데이터에서 가장 중앙에 있는 두 데이터 포인트의 평균을 의미한다.
 재밌는 사실은, 평균과 달리 중앙값은 데이터 포인트 모든 값의 영향을 받지 않는다는 것. 예를들어 값이 가장 큰 값이 커져도
 중앙값은 변하지 않음
 '''

 # 밑줄 표시로 시작하는 함수는 프라이빗 함수를 의미하며,
 # median 함수를 사용하는 사람이 직접 호출하는것이 아닌
 # median 함수만 호출하도록 생성

def _median_odd(xs : List[float]) -> float:
     """len(xs)가 홀수면 중앙값을 반환"""
     return (sorted(xs)[len(xs) // 2] 
  
def _median_even(xs : List[float]) -> float:
     """len(xs)가 짝수면 두 중앙값의 평균을 반환"""
     sorted_xs = sorted(xs)
     hi_midpoint = len(xs) // 2                     # e.g. length 4 => hi_midpoint 2
     return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2

def _median(xs : List[float]) -> float:
     """v의 중앙값을 계산"""
     return _median_even(v) if len(v) % 2 == 0 else_median_odd(v)

assert median([1, 10, 2, 9, 5]) == 5
assert median([1, 9, 2, 10]) == (2 + 9) / 2


'''
중앙값은 값들을 정렬해야함, 평균은 이상치(outlier)에 매우 민감하다. 가령 이상치가 '나쁜' 데이터(이해라려는 현상을 제대로
나타내고 있지 않은 데이터)라면 평균은 데이터에 대한 잘못된 정보를 줄 수 있다.
 분위(quantile)는 중앙값을 포괄하는 개념인데, 즉정 백분위보다 낮은 분위에 속하는 데이터를 의미한다.
'''

def quantile(xs:List[float], p:float) -> float:
    """x의 p분위에 속하는 값을 반환"""
    p_index = int(p* len(xs))
    return sorted(xs)[p_index]
assert quantile(num_friends, 0.10) == 1

def mode(x: List[float]) -> List[float]:
    """최빈값이 하나보다 많을수도 있으니 결과를 리스트로 반환"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]
assert set(mode(num_friends)) == {1,6}