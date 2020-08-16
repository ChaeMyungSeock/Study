​
def solution(brown, yellow):
# yellow를 기준으로 생각함
    answer = []
    ans =[]
    sum = brown + yellow
    for i in range(1,yellow+1):
        box = []
        if(yellow%i==0):
            j = yellow//i
            if(i>=j):
                box.append(i+2)
                box.append(j+2)
                answer.append(box)
    for i in range(len(answer)):
        if(sum == answer[i][0]*answer[i][1]):
            ans = answer[i]
​
​
​
    return ans
​
​
brown = 10
yellow = 2
​
print(solution(brown,yellow))
​
​
brown = 8
yellow = 1
​
print(solution(brown,yellow))
​
brown = 24
yellow = 24
​
print(solution(brown,yellow))
​
brown = 50
yellow = 22
​
print(solution(brown,yellow))
