from flask import Flask, render_template
app = Flask(__name__)

@app.route('/user/<name>')

def user(name):
    return render_template('user2.html', name = name)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port = 5000)
    

'''
<h3> capitalize : </h3>
<h1> Hello, {{name|capitalize}} !!! </h1>
Hello, Myungseock !!!  첫글자 대문자


<h3> upper : </h3>
<h1> Hello, {{name|upper}} !!! </h1>
Hello, MYUNGSEOCK !!!  모두 대문자


<h3> title : </h3>
<h1> Hello, {{name|title}} !!! </h1>
Hello, Myungseock !!!  첫글자 대문자
'''