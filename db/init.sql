CREATE DATABASE flaskdb;
USE flaskdb;


CREATE TABLE messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50),
    password VARCHAR(50),
    message TEXT
);


INSERT INTO messages
  (name, password, message)
VALUES
  ('Mark', 'qwer1212', 'Good Morning!'),
  ('Zach', 'asdf1212', 'Good Afternoon!'),
  ('Mason', 'zxcv1212', 'Good Evening!');