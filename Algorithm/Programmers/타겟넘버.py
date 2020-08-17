from itertools import product
def solution(numbers, target):
    number = [(x,-x) for x in numbers]
    answer = list(map(sum,product(*number)))
    return answer.count(target)

numbers = [1, 1, 1, 1, 1]	
target = 3
print(solution(numbers,target))