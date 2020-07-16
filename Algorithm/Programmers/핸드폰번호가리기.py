def solution(phone_number):
    a = len(phone_number) - 4
    answer = '*'*a + phone_number[-4:]
    return answer