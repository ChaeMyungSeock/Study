import sqlite3

conn = sqlite3.connect("test.db")
# 만든적 없으면 자동으로 생성

cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS supermarket(Itemno INTEGER, Category TEXT,
                FoodName TEXT, Company TEXT, Price INTEGER)""")

sql = "DELETE FROM supermarket"
cursor.execute(sql)
# supermaket이라는 테이블 생성했는데 칼럼 이름에 데이터를 넣어 줬는데 나중에 다시 실행하게 되면 
# 데이터가 저장됨 따라서 나중에 다시 실행할 때 data가 겹치기 때문에 초기화

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) \
        values (?,?,?,?,?) "
cursor.execute(sql,(1, '과일', '자몽', '마트', 1500))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) \
        values (?,?,?,?,?) "
cursor.execute(sql,(2, '음료수', '망고주스', '편의점', 1000))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) \
        values (?,?,?,?,?) "
cursor.execute(sql,(3, '고기', '소고기', '하나로마트', 10000))

sql = "SELECT * FROM supermarket"
# sql = "SELECT Itemno, Category, FoodName, Company, Price FROM supermaket"

cursor.execute(sql)

rows = cursor.fetchall()

for row in rows:

    print(str(row[0]) + " " + str(row[1]) + " " + str(row[2])+  " " + str(row[3]) + " " + str(row[4]))
conn.commit()
conn.close()
