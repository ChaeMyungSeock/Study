def solution(citations):
    answer = 0
    max_num = max(citations)
    citations.sort(reverse = True)
    # 문제대로 큰 수 대로 인용한 숫자를 세준다
    while(1):
        cnt = 0
        # 카운트한 논문이 없으므로 0
        for a in citations:
        # 논문의 인용갯수를 변수로 사용
            if a >= max_num:
            # 파악한 현 논문인용 갯수가 max_num이라는 변수의 인용갯수의 숫자를 넘어가면 cnt
                cnt +=1
        if cnt >= max_num and len(citations)-cnt <= max1:
            # 카운트의 숫자가 인용갯수보다 많거나 혹은 같을 때
            # len(citations)-cnt <= max1 는 당연히 max1과 같거나 작아짐
            # cnt == max_num의 경우도 틀린것은 아니나 시간 초과가 발생한다 
            # 아마도 중간에 발생하는 다른 변수들을 cnt되지 않는 변수들을 고려하는라 시간초과가 발생하는듯...
            return max1
        # print(answer)
        max_num -= 1
    return answer