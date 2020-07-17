def solution(priorities, location):
    answer = 0
    while len(priorities) != 0:
        if priorities[0] == max(priorities):
        # 첫번 째 문서 중요도가 가장 높은 문서의 경우
            answer += 1
            priorities.pop(0)
            if location==0:
                return answer
            else:
                location -= 1
        else:
        # 문서의 중요도가 다 다를 때
            priorities.append(priorities.pop(0))
            if location == 0:
                location = len(priorities) - 1 
            # 앞에 문서를 빼줘서 맨 뒤로 보내줬기 때문에 location을 옮겨 줌
            else :
                location -= 1
            # 문서가 앞으로 당겨짐

    return answer


    # def solution(priorities, location):
#     box=[]
#     for index, prior in enumerate(priorities):
#         box.append([index,prior])
#     c = box[location]
#     print('c :', c )
#     for i in range(location+1):
#         a = box[i][1]
#         for j in range(i, len(box),1):
#             if(a < box[j][1]):
#                 b = box.pop(0)
#                 box.append(b)
#     answer = box.index(c)
#     print(box)
#     return answer

# priorities=[1, 1, 9, 1, 1, 1]
# location = 0
# # priorities = [2, 1, 3, 2]
# # location = 2

# print(solution(priorities, location))
