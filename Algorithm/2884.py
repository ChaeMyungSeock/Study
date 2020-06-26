H,M = input().split()
H = int(H)
M = int(M)


if(M>=45):
    print(H, M-45)
else:  
    if(H==0):
        print(23, 15+M)
    else:
        print(H-1, 15+M)