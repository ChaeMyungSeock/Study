def solution(phone_book):
    for i in range(1,len(phone_book)):
        for j in range(len(phone_book)):
            if((i-1)!=j and len(phone_book[j])>=len(phone_book[i-1])):
                print(i)
                if(phone_book[i-1]==phone_book[j][:len(phone_book[i-1])]):
                    print(phone_book[i-1])
                    print(phone_book[j])
                    return False
    return True


phone_book	= ["123", "456", "789"]

print(solution(phone_book))
