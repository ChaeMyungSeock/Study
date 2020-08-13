def solution(people, limit):
    answer = 0
    people.sort()
    # 일단 정렬후에 처음과 끝을 더해서 더 적은 보트로 판단하려고 함
    i =0
    j = len(people)-1
    while i<=j:
        answer +=1
        if people[i] + people[j] <=limit:
            i += 1
        j -=1
    return answer




people =[70, 50, 80, 50]
limit = 100


print(solution(people,limit))
people = [10,20,30,40,50,60,70,80,90]
limit = 100 # => 5

print(solution(people,limit))
