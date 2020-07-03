import sys

ham = [0]*3
ju = [0]*2
for i in range(3):
    ham[i] = int(sys.stdin.readline())

for j in range(2):
    ju[j] = int(sys.stdin.readline())

print(min(ham)+min(ju)-50)