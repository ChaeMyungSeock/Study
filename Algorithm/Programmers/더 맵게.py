import heapq
def solution(scoville, K):
    heapq.heapify(scoville)
    answer =0
    while len(scoville) >1:
        n1 = heapq.heappop(scoville)
        n2 = heapq.heappop(scoville)

        if n1 <K or n2< K:
            heapq.heappush(scoville, n1+(n2*2))
            answer +=1
        else:
            return answer
    if(scoville[0]>=K):
        return answer
    else:
        return -1

        

scoville = [1, 2, 3, 9, 10, 12]
K = 7
print(solution(scoville,K))
# print(solution(scoville,K))