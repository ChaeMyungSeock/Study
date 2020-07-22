# pymssqpl 패키지 import
import pymssql

# MSSQL 접속
conn = pymssql.connect(server="127.0.0.1", user='Hexapod', password = '402402', database='Test01')

 
# Connection 으로부터 Cursor 생성
cursor = conn.cursor()

sql_datainput = "insert into Product values (%d, %s,%d)"

# SQL문 실행

# cursor.execute(sql_datainput)
cursor.execute(sql_datainput, ('1','d','1'))
cursor.execute(sql_datainput, ('2','z','2'))
conn.commit()
 
# # 데이타 하나씩 Fetch하여 출력
# row = cursor.fetchone()
# while row:
#     print(row[0], row[1], row[2], row[3])
#     row = cursor.fetchone()
   
# # 연결 끊기
conn.close()       
