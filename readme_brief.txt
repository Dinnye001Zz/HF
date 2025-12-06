CREATE VENV
python -m venv .venv

INTALL PACKAGES
pip install -r requirements.txt

Run tests: 'pytest tests/test_train_inference.py'

Run app.py/flask app: 'python app.py'
API ip: http://127.0.0.1:8080
mlflow server --host 127.0.0.1 --port 5102
MLFlow ip: http://127.0.0.1:5102

RUN DOCKER FILE (runs both mlflow and api)
docker run -p 8080:8080 -p 5102:5102 mlsecops

BUILD DOCKER IMAGE
docker build -t mlsecops .

START DOCKER COMPOSE
docker-compose -f docker-compose-airflow.yml up -d

Airflow: http://127.0.0.1:8090
Username: admin
Pw: admin

STOP DOCKER COMPOSE
docker-compose -f docker-compose-airflow.yml down

INFERENCE EXAMPLE:
{
  "inference_row": [
    "France",
    "Marseille",
    "Farmhouse",
    "Semi-Furnished",
    "991",
    "412935",
    "1989",
    "6",
    "6",
    "2",
    "1",
    "1",
    "1",
    "0",
    "10745",
    "193949",
    "15",
    "6545",
    "218986",
    "0.16",
    "1",
    "5",
    "6"
  ]
}