def solution(nums):
    answer = 0
    ball = []
    for i in nums:
        if i not in ball: 
            ball.append(i)
    
    if (len(ball)<=len(nums)//2):
        return len(ball)
    else :
        return len(nums)//2


  




nums = [3,1,2,3]
ball = []
for i in nums:
    if i not in ball: 
        ball.append(i)
print(len(ball))