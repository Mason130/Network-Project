## Build docker image
`docker compose build`

## Start container
`docker compose up`

The flask app will be available at **http://localhost:5001/**

Optional: `docker compose exec app bash`

## SQL injection examples
1. `name = ' OR 'a'='a';--` and `password = any`