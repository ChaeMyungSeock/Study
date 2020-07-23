def solution(prices):
    a = len(prices)
    print('a : ',a)
    answer = [0]*a
    for i in range(0,a-1):
        reward = 0
        for j in range(i, a):
# 뒤로 바로 작은 값이 있어도 1초뒤에 가격이 떨어지므로 
# 같은 값을 비교해줘서 1을 보상
            if(prices[i]<=prices[j]):
                print('j :',j)
                reward += 1
                if(j== a-1):
                    reward -= 1
                    answer[i] = reward
                    break
            else:
                print('reward : ',reward)
                answer[i] = reward
                break
    answer[-1] = 0
    return answer

prices = [1, 2, 3, 2, 3]	
print(solution(prices))

'''
def solution(prices):
    a = len(prices)
    answer = []
    for i in range(0,a-1):
        reward = 0
        for j in range(i, a):

            if(prices[i]<=prices[j]):
                reward += 1
                if(j== a-1):
                    reward -=1
                    answer.append(reward)
                    break
            else:
                answer.append(reward)
                break
    answer.append(0)
    return answer
'''