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
    return connection

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select', methods=['POST'])
def select():
    # parse request
    name = request.form.get('name')
    password = request.form.get('password')
    # connect to MySQL server
    connection = MySQL()
    cursor = connection.cursor()
    # cursor.execute('SELECT * FROM messages WHERE name = %s AND password = %s', (name, password)) # for MySQL
    '''Vulnerable query format'''
    query = "SELECT * FROM messages WHERE name = '%s' AND password = '%s'" % (name, password)
    #cursor.execute(query, multi=True) # for MySQL
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    return render_template('result.html', messages=results)

@app.route('/insert', methods=['POST'])
def insert():
    # parse request
    name = request.form.get('name')
    password = request.form.get('password')
    message = request.form.get('new_message')
    # connect to MySQL server
    connection = MySQL()
    cursor = connection.cursor()
    query = "INSERT INTO messages (name, password, message) VALUES ('%s', '%s', '%s')" % (name, password, message)
    cursor.execute(query)
    connection.commit()
    cursor.close()
    return render_template('back.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001', debug=True)
