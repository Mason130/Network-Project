import os
import sqlite3
from flask import Flask, render_template, request
import module.classifier as clf

app = Flask(__name__)
# current_script_path = os.path.dirname(__file__)
# initializa waf
waf = clf.WAF()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select', methods=['POST'])
def select():
    # parse request
    name = request.form.get('name')
    password = request.form.get('password')
    # use waf to detect input 
    x, y = waf.predict(name), waf.predict(password)
    print(x, y)
    if x or y:
        error_page = """
        <!DOCTYPE html>
        <html>
        <body>
            <h1 style="color: red; text-align: center;">Suspicious Request!</h1>
            <p style="text-align: center;">Please enter your information in normal format</p>
        </body>
        </html>
        """
        return error_page
    # connect to Database
    # connection = sqlite3.connect(os.path.join(current_script_path, 'flaskapp.db'))
    connection = sqlite3.connect('./flaskapp.db')
    cursor = connection.cursor()
    # cursor.execute('SELECT * FROM messages WHERE name = %s AND password = %s', (name, password)) # for MySQL
    '''Vulnerable query format'''
    query = "SELECT * FROM messages WHERE name = '%s' AND password = '%s'" % (name, password)
    print(query)
    cursor.execute(query)
    results = cursor.fetchall()
    print(results)
    cursor.close()
    return render_template('result.html', messages=results)

@app.route('/insert', methods=['POST'])
def insert():
    # parse request
    name = request.form.get('name')
    password = request.form.get('password')
    message = request.form.get('new_message')
    # use waf to detect input 
    x, y, z = waf.predict(name), waf.predict(password), waf.predict(message)
    print(x, y, z)
    if x or y or z:
        error_page = """
        <!DOCTYPE html>
        <html>
        <body>
            <h1 style="color: red; text-align: center;">Suspicious Request!</h1>
            <p style="text-align: center;">Please enter your information in normal format</p>
        </body>
        </html>
        """
        return error_page
    # connect to Database
    # connection = sqlite3.connect(os.path.join(current_script_path, 'flaskapp.db'))
    connection = sqlite3.connect('./flaskapp.db')
    cursor = connection.cursor()
    try:
        query = "INSERT INTO messages (name, password, message) VALUES ('%s', '%s', '%s')" % (name, password, message)
        print(query)
        cursor.execute(query)
        connection.commit()
        cursor.close()
        return render_template('back.html')
    except sqlite3.Error as e:
        return(f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001', debug=True)
