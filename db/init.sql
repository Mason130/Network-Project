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
  ('Emily', 'pass1234', 'Hello Morning!'),
  ('John', 'mypass4321', 'Hey Afternoon!'),
  ('Lily', 'abcd1234', 'Evening Greetings!'),
  ('Sophia', 'pass6789', 'Bright Morning!'),
  ('James', 'james456', 'Morning Sparkle!'),
  ('Isabella', 'bella234', 'Afternoon Glow!'),
  ('Oscar', 'oscar678', 'Evening Peace!'),
  ('Charlotte', 'char7890', 'Morning Light!'),
  ('Ben', 'ben12345', 'Afternoon Shine!'),
  ('Nora', 'nora5678', 'Evening Relax!'),
  ('Luke', 'luke1234', 'Morning Bliss!'),
  ('Hannah', 'hanna321', 'Afternoon Serenity!'),
  ('Sam', 'sam4321', 'Quiet Evening!'),
  ('Jack', 'jack1234', 'Morning Vibes!'),
  ('Noah', 'noah5678', 'Quiet Evening!'),
  ('Emma', 'xyz12345', 'Good Noon!'),
  ('Ethan', 'ethan000', 'Sunny Afternoon!'),
  ('Olivia', 'olivia11', 'Peaceful Evening!'),
  ('Mason', 'zxcv1212', 'Good Evening!');