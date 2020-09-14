from itertools import combinations
import operator

def solution(orders, course):
    result = []
    key = []

    for i in range(len(orders)):
        orders[i] = ''.join(sorted(orders[i]))
    for i in orders:
        for k in range(2,len(i)+1):
            number = [''.join(j) for j in combinations(i,k)]
            key += number 

    count={}
    for i in key:
        try: count[i] += 1
        except: count[i]=1
    count = sorted(count.items(),key=operator.itemgetter(1))
    print(count)
    for i in course:
        max = 0
        for j in range(len(count)):
            if len(count[j][0]) == i:
            # 문자열의 조합 갯수
                if count[j][1] >=2 :
                    if count[j][1] > max:
                        max = count[j][1]
                        answer = count[j][0]

        for k in range(len(count)):
            if len(count[k][0]) == i:
                if count[k][1] == max:
                    result.append(count[k][0])
    
    return sorted(result)

# orders = ["ABCFG", "AC", "CDE", "ACDE", "BCFG", "ACDEH"]
# course = [2,3,4]	
# print(solution(orders, course))

orders = ["XYZ", "XWY", "WXA"]	
course = [2,3,4]	
print(solution(orders, course))




# course = [2,3,4]
# key = []
# result = [] 

# import operator
# for i in orders:
#     for k in range(2,len(i)+1):
#         number = [''.join(j) for j in combinations(i,k)]
#         # print(number)
#         key += number 
# # print("1211")       
# # print(key)
# count={}
# for i in key:
#     try: count[i] += 1
#     except: count[i]=1
# # print(count)
# count = sorted(count.items(),key=operator.itemgetter(1))
# # print(count.__class__)
# print(count)
# answer = ''
# for i in course:
#     max = 0
#     for j in range(len(count)):
#         if len(count[j][0]) == i:
#         # 문자열의 조합 갯수
#             if count[j][1] >=2 :
#                 if count[j][1] > max:
#                     max = count[j][1]
#                     answer = count[j][0]
#     print('i : ',i)
#     print(max)    
#     for k in range(len(count)):
#         if len(count[k][0]) == i:
#             if count[k][1] == max:
#                 result.append(count[k][0])





# print(result)

# print(solution(orders, course))

# orders = ["ABCDE", "AB", "CD", "ADE", "XYZ", "XYZ", "ACD"]	
# course = [2,3,5]	
# print(solution(orders, course))