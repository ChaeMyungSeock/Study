def solution(numbers):
    answer = ''
    numbers = list(map(str,numbers))
    numbers.sort(key=lambda x:x*3,reverse=True)
    # 문자열에서 숫자의 경우 [0] -> [1]순으로 비교함
    # 그리고 큰 숫자부터 기본적으로 내림차순으로 비교
    answer = str(int(''.join(numbers)))
   # answer = str(''.join(numbers))
     # 반드시 int를 붙여줘야함...  마지막 테스터 케이스가 000인데 string의 경우 print 해보면 000 그대로 나옴 int형으로 바꿔야 0으로 나옴
    return answer

# answer = ''
# numbers = list(map(str,numbers))
# numbers.sort(key=lambda x:x*3,reverse=True)
#     # 문자열에서 숫자의 경우 [0] -> [1]순으로 비교함
#     # 그리고 큰 숫자부터 기본적으로 내림차순으로 비교
# answer = str(int(''.join(numbers)))
#    # answer = str(''.join(numbers))
#      # 반드시 int를 붙여줘야함...  마지막 테스터 케이스가 000인데 string의 경우 print 해보면 000 그대로 나옴 int형으로 바꿔야 0으로 나옴


# print(answer)    

# def permute(arr):
#     result = [arr[:]]
#     c = [0] * len(arr)
#     i = 0
#     while i < len(arr):
#         if c[i] < i:
#             if i % 2 == 0:
#                 arr[0], arr[i] = arr[i], arr[0]
#             else:
#                 arr[c[i]], arr[i] = arr[i], arr[c[i]]
#             result.append(arr[:])
#             c[i] += 1
#             i = 0
#         else:
#             c[i] = 0
#             i += 1
#     return result
	
# answer = [0]*len(a)
# for i in range(len(a)):
#     b = list(map(str, a[i]))
#     b = ''.join(b)
#     b = int(b)
#     answer[i] = b
# print(max(answer))
# a = map(str, a)
# a = ''.join(a)
# b = list(map(int, a))
# print(b)

# n개중에 n개를 뽑을 때 순서를 고려해야 하기 때문에 순열 n!개임 일단

# def permute(arr):
#     result = [arr[:]]
#     c = [0] * len(arr)
#     i = 0
#     while i < len(arr):
#         if c[i] < i:

