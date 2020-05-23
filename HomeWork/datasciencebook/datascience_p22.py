# 문자열

'''
문자열 (string)은 작은 따옴표(') 또는 큰 따옴표(")로 묶어 나타낸다
만약 역슬래시를 역슬래시로 보이는 문자로 사용하고 싶다면(특히 윈도우 디렉터리 이름이나 정규표현식에서 사용하고 싶을때)
문자열 앞에 r을 붙여 raw string(가공되지 않은 문자열)이라고 명시하면 된다.
'''
tab_string = "\t"
not_tab_string = r"\t" #문자 '\'와 't'를 나타내는 문자열


# 예외 처리
'''
코드가 뭔가 잘못됐을 때 파이썬은 예외(exception)가 발생했음을 알려준다. 예외를 제대로 처리해 주지 않으면 
프로그램이 죽는데, 이를 방지하기 위해 사용할 수 있는 것이 try와 except이다.
'''

try:
    print(0 / 0)
except  ZeroDivisionError:
    print("cannot divide by zero")
