def solution(n, m):
    answer = []
    m_sub = m
    n_sub = n
    if(n < m):
            while 1:
                m = m % n
                print(m)
                if(m==0):
                    
                    answer.append(n)
                    answer.append((m_sub//n) * (n_sub//n) * n)
                    break
                else:
                    n = n % m
                    if(n==0):
                        
                        answer.append(m)
                        answer.append((m_sub//m) * (n_sub//m) * m)
                        break
                    
    else:
        while 1:
            n = n % m
            if(n==0):
                
                answer.append(m)
                answer.append((m_sub//m) * (n_sub//m) * m)
                break
            else:
                m = m % n
                if(m==0):
                    
                    answer.append(n)
                    answer.append((m_sub//n) * (n_sub//n) * n)
                    break


                
    return answer