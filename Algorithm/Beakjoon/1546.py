# 백준1546
import sys

num = int(sys.stdin.readline())
num_list = list(map(int, sys.stdin.readline().split()))
# 입력값들을 나눠서 리스트로 받아주겠다
 
max = max(num_list)
# max함수 리스트의 맥스값 리턴
sum = sum(num_list)
# sum함수 리스트의 sum값 리턴
print(((sum/max)*100)/num)

