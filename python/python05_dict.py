# 3. 딕셔너리 # 중복 x
# {키 : 벨류}
# {key : value}

a = {1 : 'hi', 2: 'hello'}
print(a)
print(a[1])

b = {'hi' : 1, 'hello' :2}
print(b)
print(b['hello'])

#딕셔너리 요소 삭제
del a[1]
print(a)
del a[2]
print(a)

a = { 1 :'a' , 1 : 'b',  1 : 'c'} # a라는 메모리에 값이 덮어 씌워짐
print(a)

b = {1:'a', 2:'a',3:'a'} # value가 같은건 크게 문제가 되지 않는다 다른 메모리 주소에 각 값들이 저장되므로
print(b)

a = {'name' : 'AI_class', 'phone' : '010', 'birth' : '0511'}
print(a.keys())
print(a.values())
print(type(a))
print(a.get('name'))
print(a['name'])
print(a.get('phone'))
print(a['phone'])
