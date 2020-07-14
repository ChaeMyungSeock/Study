from flask import Flask

app = Flask(__name__)

print(app)

@app.route('/')
# http://127.0.0.1:8888 + / 주소를 제외하고 /여기까지 쳐줘야 웹서버가 나옴
def hello333():
    return "<h1>hello myungseock world</h1>"


@app.route('/bit')

def hello334():
    return "<h1>hello bit computer world</h1>"

@app.route('/bit/bitcamp')

def hello335():
    return "<h1>hello bitcamp world</h1>"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8888, debug=True)

