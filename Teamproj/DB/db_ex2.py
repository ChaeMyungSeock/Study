# pymssqpl 패키지 import
import pymssql

# MSSQL 접속
conn = pymssql.connect(server = '192.168.0.146', user = 'bit2', port = 1433,
            password = '1234',database = 'Hexapod_ex')

 
# Connection 으로부터 Cursor 생성


cursor = conn.cursor()

# SQL 문 만들기
sql = "INSERT INTO ex_data(ex_lidar, ex_servo, ex_picamera) values(%s,%d,%s);"

cursor.execute(sql, ('0.5' ,'32', '11.5'))
print('끗')

# DB에 Complete 하기
conn.commit()



# DB 연결 닫기
conn.close()



'''
sql_datainput = "insert into Product values (%d, %s,%d)"
# 데이터 부분에서 데이터 형식을 맞춰줘야함 

# SQL문 실행

# cursor.execute(sql_datainput)
cursor.execute(sql_datainput, ('1','d','1'))
cursor.execute(sql_datainput, ('2','z','2'))
# 데이터 삽입

conn.commit()
 
# # 데이타 하나씩 Fetch하여 출력
# row = cursor.fetchone()
# while row:
#     print(row[0], row[1], row[2], row[3])
#     row = cursor.fetchone()
   
# # 연결 끊기
conn.close()
'''