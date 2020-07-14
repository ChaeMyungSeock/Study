from flask import Flask

app = Flask(__name__)

@app.route('/')

def hello333():
    return "<h1>hello world</h1>"


@app.route('/ping', methods = ['GET'])
def ping():
    return "<h1>pong</h1>"


if __name__ == '__main__':
    app.run(host = '127.0.0.1', port=5000, debug=False)
# default debug => False


'''
127.0.0.1 남이 볼 때 나의 ip
내가 볼 때 나의 ip 192.168.0.146
'''

