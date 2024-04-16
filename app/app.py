from flask import Flask, render_template, request
import mysql.connector as connector


app = Flask(__name__)

def MySQL():
    '''
    Container DB
    '''
    connection = connector.connect(
        user = 'root',
        password = 'root',
        host = 'db',
        port = '3306',
        database = 'flaskdb'
    )
    '''
    Local DB
    '''
    # connection = connector.connect(
    #     user = 'chengyu',
    #     password = 'qwer1212',
    #     host = 'localhost',
    #     port = '3306',
    #     database = 'flaskapp'
    # )
    return connection

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select', methods=['POST'])
def select():
    name = request.form.get('name')
    password = request.form.get('password')
    connection = MySQL()
    cursor = connection.cursor()
    # cursor.execute('SELECT * FROM messages WHERE name = %s AND password = %s', (name, password))
    # Vulnerable query
    query = "SELECT * FROM messages WHERE name = '%s' AND password = '%s'" % (name, password)
    print(query)
    cursor.execute(query, multi=True)
    results = cursor.fetchall()
    print(results)
    cursor.close()
    return render_template('result.html', messages=results)

@app.route('/insert', methods=['POST'])
def insert():
    name = request.form.get('name')
    password = request.form.get('password')
    message = request.form.get('new_message')
    connection = MySQL()
    cursor = connection.cursor()
    cursor.execute("INSERT INTO messages (name, password, message) VALUES (%s, %s, %s)", (name, password, message))
    connection.commit()
    cursor.close()
    return render_template('back.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001', debug=True)
