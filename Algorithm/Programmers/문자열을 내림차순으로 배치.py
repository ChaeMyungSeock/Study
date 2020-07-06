print(ord('a'))
print(ord('A'))

s = 'Zbcdefg'

s_list = list(s)
s_list.sort(reverse=True)
answer = ''.join(s_list)
print(answer)