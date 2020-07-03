import sys

student = [0]*5
for i in range(5):
    score = int(sys.stdin.readline())
    student[i] = score
    if student[i]<40:
        student[i] = 40

print(sum(student)//5)