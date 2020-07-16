from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

# 데이터베이스
conn = sqlite3.connect("D:/Study/data/db/wanggun.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM general")
print(cursor.fetchall())


@app.route('/')
def run():
    conn = sqlite3.connect('./data/db/wanggun.db')
    c = conn.cursor()
    c.execute("SELECT * FROM general")
    rows = c.fetchall()
    return render_template("board_index.html", rows = rows)

@app.route('/modi')
def modi():
    id = request.args.get('id')
    conn = sqlite3.connect('./data/db/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general WHERE id ='+ str(id))
    rows = c.fetchall()
    return render_template("bord_modi.html", rows = rows)

@app.route('/addrec',methods=['POST','GET'])
def addrec():
    if request.method == 'POST':
        print(request.form['war'])
        print(request.form['id'])
        try:
            war = request.form['war']
            id = request.form['id']
            with sqlite3.connect("./data/db/wanggun.db") as con:
                print("con")
                cur = con.cursor()
                cur.execute("UPDATE general SET war="+str(war)+" WHERE id="+str(id))
                print("con2")
                con.commit()
                msg="정상적으로 입력되었습니다"

        except:
            conn.rollback()
            msg = '에러가 발생하였습니다.'
        finally:
            return render_template("board_result.html", msg=msg)
            conn.close()

app.run(host='127.0.0.1',port=5010,debug=False)


# SELECT * FROM general WHERE id = 5; where 어디냐
# SELECT * FROM general WHERE name = '왕건';



