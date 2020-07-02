string_a ="babacvabba"
# string_a = list(string_a)
# string_a.reverse()
# print(string_a)

from collections import Counter


string_a = list(string_a)
cnt = 0
b = len(string_a)
print(b)
if(b%2==1):
    string_b = string_a[:b//2]
    string_b.reverse()
    string_c = string_a[b//2+1:]
    print(string_b)
    print(string_c)

else:
    string_b = string_a[:b//2]
    string_b.reverse()
    string_c = string_a[b//2:]

while 1:
    c = Counter(string_b)
    d = Counter(string_c)
    
    for j in range(b//2):
        if(string_b[j] != string_c[j]):
            cnt +=1
            string_b[j] = string_c[j]

    if(string_b == string_c):
        break

print(cnt)











# def palindrom(string_a):
    # string_a = list(string_a)
    # cnt = 0
    # b = len(string_a)
    # print(b)
    # if(b%2==1):
    #     string_b = string_a[:b//2]
    #     string_b.reverse()
    #     string_c = string_a[b//2+1:]
    #     print(string_b)
    #     print(string_c)

    # else:
    #     string_b = string_a[:b//2]
    #     string_b.reverse()
    #     string_c = string_a[b//2:]

    # while 1:
    #     for i in range(b//2-1):
    #         c = string_b[i]
    #         d = string_b[i+1]
    #         e = string_c[i]
    #         f = string_c[i+1]
    #         if(c==f and d == e):
    #             cnt +=1
    #             string_b[i] = d
    #             string_b[i+1] = c
    #             break
    #         for j in range(b//2):
    #             if(string_b[j] != string_c[j]):
    #                 cnt +=1
    #                 string_b[j] = string_c[j]

    #     if(string_b == string_c):
    #         break
    # return cnt

# print(palindrom(string_a))

        
