# pymssqpl 패키지 import
import pymssql
'''
Python에서 MSSQL에 있는 데이타를 사용하는 일반적인 절차는 다음과 같다.

1. pymssql 모듈을 import 한다
2. pymssql.connect() 메소드를 사용하여 MSSQL에 Connect 한다. 
   호스트명, 로그인, 암호, 접속할 DB 등을 파라미터로 지정할 수 있다.
3. DB 접속이 성공하면, Connection 객체로부터 cursor() 메서드를 호출하여 Cursor 객체를 가져온다. 
   DB 커서는 Fetch 동작을 관리하는데 사용된다.
4. Cursor 객체의 execute() 메서드를 사용하여 SQL 문장을 DB 서버에 보낸다.
5. SQL 쿼리의 경우 Cursor 객체의 fetchall(), fetchone(), fetchmany() 등의 메서드를 사용하여 
   데이타를 서버로부터 가져온 후, Fetch 된 데이타를 사용한다.
6. 삽입, 갱신, 삭제 등의 DML(Data Manipulation Language) 문장을 실행하는 경우, 
   INSERT/UPDATE/DELETE 후 Connection 객체의 commit() 메서드를 사용하여 데이타를 확정 갱신한다.
7. Connection 객체의 close() 메서드를 사용하여 DB 연결을 닫는다.
'''
# MSSQL 접속
conn = pymssql.connect(server="127.0.0.1", user='Hexapod', password = '402402', database='Test01')

 
# Connection 으로부터 Cursor 생성
cursor = conn.cursor()
 
# SQL문 실행
cursor.execute("SELECT * FROM member;")
 
# 데이타 하나씩 Fetch하여 출력
row = cursor.fetchone()
while row:
    print(row[0], row[1])
    row = cursor.fetchone()
   
# 연결 끊기
conn.close()       
