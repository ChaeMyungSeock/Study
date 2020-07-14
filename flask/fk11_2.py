import pymssql as ms

# print('완료')

conn = ms.connect(server = '127.0.0.1', user = 'bit2', 
            password = '1234',database = 'bitdb')

print('끗')
cursor = conn.cursor()
cursor1 = conn.cursor()

cursor.execute("SELECT * FROM sonar;")
cursor1.execute("SELECT * FROM sonar;")
row = cursor.fetchone()
row1 = cursor1.fetchone()

while row:
    print("첫 컬럼 : %s, 두 번째 컬럼 : %s "%(row[0], row[1]))
    print('===========================================')
    print("첫 컬럼 : %s, 두 번째 컬럼 : %s "%(row1[0], row1[1]))
    print('===========================================')
    
    row = cursor.fetchone()
    row1 = cursor1.fetchone()

conn.close()