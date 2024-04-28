# Project for UVA CS6501/ECE6502 (Network Security and Privacy), 24 Spring
## Run with Docker
- Build & start docker image: `docker compose up --build`
- The flask app will be available at **http://localhost:5001/**
- Optional: `docker compose exec app bash`

## Run locally
- Change directory: `cd app`
- Install dependencies: `pip install -r requirements.txt`
- Run: `python app.py` (unprotected version); `python app_protected.py` (protected version);
- The flask app will be available at **http://localhost:5001/**

## Demo Site Hosted by [pythonanywhere](https://www.pythonanywhere.com/)
Due to storage limitations of pythonanywhere, only the unprotected version was deployed **https://rogerluo233.pythonanywhere.com/**

## SQL injection datasets & examples
1. [Dataset1](https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset); [Dataset2](https://www.kaggle.com/datasets/sajid576/sql-injection-dataset)
2. `SELECT * FROM table1 WHERE name = '' OR 'a'='a';--';-- AND password = 'aseqweqw213'`
