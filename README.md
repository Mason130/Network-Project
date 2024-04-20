## Run with Docker
- Build docker image: `docker compose build`
- Start container: `docker compose up`
- The flask app will be available at **http://localhost:5001/**
- Optional: `docker compose exec app bash`

## Run locally
- Change directory: `cd app`
- Install dependencies: `pip install -r requirements.txt`
- Run: `python app_sqlite.py`
- The flask app will be available at **http://localhost:5001/**

## Demo Site Hosted by [pythonanywhere](https://www.pythonanywhere.com/)
**https://rogerluo233.pythonanywhere.com/**

## SQL injection examples
1. [Dataset1](https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset); [Dataset2](https://www.kaggle.com/datasets/sajid576/sql-injection-dataset)
2. `name = ' OR 'a'='a';--` and `password = any`
