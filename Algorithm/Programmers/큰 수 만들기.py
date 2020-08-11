from itertools import combinations
from itertools import permutations
''' 
1. 하나의 값을 넣는다
2. 맨 위에 쌓인 값과 그 바로 밑을 비교
3. 맨 위의 값이 더 클 경우 두개의 값의 자리를 바꾸고 pop해서 빼낸다 맨위 값이 작으면 넘어감
4. 1~3을 k번 or number를 다 쌓으면 끝
'''
# def solution(number, k):
#     collect = [] # 숫자를 따로 모아서 큰 수 를 만들 리스트

#     for i, num in enumerate(number):
#         # k개 만큼의 숫자를 빼낼 때, i의 인덱스를 기억
#         while len(collect)>0 and collect[-1] < num and k >0:
#             # 1. 맨 마지막 문자만 비교해도 괜찮은 이유는 하나씩 쌓아서 내림차순으로
#             # 2. 알파벳의 대소관계를 이용
#             # 3. collect의 마지막 문자는 한 문자로 이루어져 있다. 
#             collect.pop() # 리스트의 맨 마지막 문자를 삭제
#             k -=1
#         if k == 0:
#             collect += list(number[i:])
#             break
#         collect.append(num)
#     collect = collect[:-k] if k > 0 else collect
#     # k가 0이 되면 빈 리스트가 되기 때문에 if를 이용해 조건을 걸어준다


#     return ''.join(collect)


def solution(number,k):
    number = list(number)
    # 문자열을 리스트형태로 받음
    answer = [number[0]]
    # 일단 첫버째 문자을 받고 시작
    toss = 0
    for i in range(1, len(number)):
        answer.append(number[i])
        # 문자열을 리스트로 하나씩 받음
        toss+=1
        # 넘기는 숫자 체크
        cor = 1
        # 연속된 숫자를 제거하기 위함
        while(cor and k !=0):
            # cor 과 k가 0이 아니라면 while문에서 문자열들의 크기 비교
            cor -=1
            # 일단 하나를 빼기 위함이니 1빼고 시작
            for j in range(toss, len(answer)):
                # 뒤에 문자와 비교해서 큰지 작은지 check
                if answer[j-1] >= answer[j]:
                    pass
                # 패스
                else:
                # 둘이 바꾸고 빼준다
                    answer[j-1], answer[j] = answer[j], answer[j-1]
                    answer.pop()
                    cor += 1
                # 하나 제거해줬으니 1 증가
                    toss -=1
                # 하나 제거했으니 1 감소
                    k -=1
                # 하나 제거했으니 1 감소
                    break
    answer = "".join(answer)


    return answer


number = "54321"
k = 2
print(solution(number,k))


# number = "1231234"
# k = 3
# print(solution(number,k))
