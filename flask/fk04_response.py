from flask import Flask

app = Flask(__name__)

from flask import make_response, session

@app.route('/')
def index():
    response = make_response('<h1> 잘 따라 치시오!!! </h1>')
    response.set_cookie('answer', 42)
    return 'You are not logged in'

# def index():
#     if 'username' in session:
#         return 'Logged in as %s' % escape(session['username'])
#     # response = make_response('<h1> 잘 따라 치시오!!! </h1>')
#     # response.set_cookie('answer', 42)
#     return 'You are not logged in'

# @app.route('/login',methods = ['Get', 'POST'])
# def login():
#     if request.method == 'POST':
#         session['username'] = request.form['username']
#         return redirect(url_for('index'))
#     return '''
#         <form action = "" method= "post"?
#             <p><intput type =text name - username>
#             <p><input type = submit value = Login>
#         </form?
#         '''

if __name__ == '__main__':
    app.run(host='127.0.0.1', port = 5001, debug=False)



