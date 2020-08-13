def solution(clothes):
    answer = {}
    for i in clothes:
        if i[1] in answer: 
            answer[i[1]] +=1
            # 만약에 그 원소가 있다면 value만 추가
        else: 
            answer[i[1]] = 1
            # 만약에 원소가 없다면 그 원소를 추가 key와 value로 추가함
    cnt =1
    print(answer)
    for i in answer.values():
    # value값만 필요하므로
        print(i)
        cnt *= (i+1)
        # 각각 하나씩 입었을 때를 고려한다면 +1를 해줘야함
    
    return cnt-1
    # -1를 해주는 이유는 여기서는 모두 안 입었을 때도 포함하고 있기 때문

clothes = [["yellow_hat", "headgear"], ["blue_sunglasses", "eyewear"], ["green_turban", "headgear"]]

# print(len(clothes))

# print(collections.Counter(clothes))

print(solution(clothes))