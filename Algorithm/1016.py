import sys

min,max = map(int,sys.stdin.readline().split())

def prime_list(min, max):
    solve = [True]*(max - min +1)
    count = 0
    num = 1
    while num*num <= max:
        num += 1
        square = num*num
        i = min // square

        while i * square <= max:
            idx = i *square - min

            if idx>=0 and solve[idx]:
                count +=1
                solve[idx] = False
            i +=1
    print(len(solve) - count)

prime_list(min, max)





























#     while num*num <= max:
#         num += 1
#         square = num * num
#         i = min//square

#         while square*i <= max:
#             idx = square*i - min

#             if idx >=0 and solve[idx]:
#                 count +=1
#                 solve[idx] = False
#             i +=1
#     print(len(solve) - count)

# prime_list(min, max)







# 에라토스의 체
# def prime_list(n):
#     solve = [True]*n
#     # 소수로 간주

#     m = n**0.5
#     for i in range(2,m+1):
#         if solve[i] == True: # i가 소수라면
#             for j in range(i+i, n, i):
#                 solve[j] == False
#     return [i for i in range(2,n) if solve[i] == True]

