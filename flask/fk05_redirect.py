from flask import Flask
from flask import redirect


app = Flask(__name__)

@app.route('/')

def index():
    return redirect('http://www.naver.com')
# redirect 바로 그 주소랑 연결

if __name__ == '__main__':
    app.run(host='127.0.0.1', port = 5001, debug=False)
