import pymssql as ms

# print('완료')

conn = ms.connect(server = '127.0.0.1', user = 'bit2', 
            password = '1234',database = 'bitdb')

print('끗')
cursor = conn.cursor()

cursor.execute("SELECT * FROM iris2;")

row = cursor.fetchone()

while row:
    print("첫 컬럼 : %s, 두 번째 컬럼 : %s "%(row[0], row[1]))
    row = cursor.fetchone()

conn.close()