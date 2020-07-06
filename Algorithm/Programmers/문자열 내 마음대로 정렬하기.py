def solution(strings, n):
    
    answer = sorted(strings, key = x[n])
    return answer

# index = list(range(1,4))
# print(index)

strings = ['sun', 'bed', 'car']	

answer = sorted(strings, key = lambda x : x[2])
print(answer)