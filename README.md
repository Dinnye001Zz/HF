# House Purchase Approval ML Pipeline

A complete MLSecOps pipeline for predicting house purchase approval decisions using XGBoost, with MLflow experiment tracking, Airflow orchestration, Docker containerization, and Evidently AI monitoring.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Monitoring](#monitoring)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a machine learning pipeline that predicts whether a house purchase application will be approved based on various property and customer features. The system integrates:

- **Machine Learning**: XGBoost classifier for binary classification
- **Experiment Tracking**: MLflow for model versioning and artifact management
- **Orchestration**: Apache Airflow for automated training workflows
- **Containerization**: Docker for consistent deployment environments
- **Monitoring**: Evidently AI for data quality and model performance reports
- **Visualization**: Streamlit dashboard for interactive report viewing
- **REST API**: Flask-RESTX for model serving and inference

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Airflow       │────▶│   Flask API     │────▶│   MLflow        │
│   (Scheduler)   │     │   (Training)    │     │   (Tracking)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                         │
                               ▼                         ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   Evidently     │     │   Streamlit     │
                        │   (Reports)     │────▶│   (Dashboard)   │
                        └─────────────────┘     └─────────────────┘
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| Flask API | 8080 | Model training and inference endpoints |
| MLflow Server | 5102 | Experiment tracking and model registry |
| Airflow Webserver | 8090 | DAG orchestration UI |
| Streamlit | 8501 | Evidently report visualization |
| PostgreSQL | 5432 | Airflow metadata database |

## Features

- **Automated Training Pipeline**: Airflow DAG triggers model training and promotes models based on accuracy improvements
- **Model Versioning**: MLflow tracks experiments, parameters, metrics, and artifacts
- **Staging/Production Workflow**: Models are tagged as "Staging" and promoted to "Production" based on performance
- **Data Quality Monitoring**: Evidently generates comprehensive data summary reports
- **REST API**: 
  - `/model/train` - Train new models with CSV data
  - `/model/predict` - Make predictions on new data
  - `/model/register` - Register trained models in MLflow
- **Interactive Dashboard**: Streamlit app displays Evidently reports with data quality metrics
- **Containerized Deployment**: All services run in Docker containers with shared volumes

## Prerequisites

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 1.29 or higher
- **Python**: 3.12 (for local development)
- **Storage**: At least 5GB free disk space

## Installation

### 1. Clone the Repository

```bash
git clone <https://github.com/Dinnye001Zz/HF.git>
cd HF
```

### 2. Build Docker Image

```bash
docker build -t mlsecops .
```

This builds the main image containing:
- Python 3.12
- All required packages (XGBoost, MLflow, Evidently, Flask, Streamlit)
- Application code (app.py, MLModel.py, constants.py, streamlit_app.py)
- MLflow server configured to run on port 5102

### 3. Start All Services

```bash
docker-compose -f docker-compose-airflow.yml up -d
```

This starts:
- PostgreSQL database
- Airflow webserver, scheduler, and init container
- MLflow server and Flask API
- Streamlit dashboard

### 4. Verify Services

Wait 30-60 seconds for services to initialize, then check:

```bash
docker-compose -f docker-compose-airflow.yml ps
```

All services should show "Up" status.

### 5. Access Web Interfaces

- **Airflow UI**: http://localhost:8090 (admin/admin)
- **MLflow UI**: http://localhost:5102
- **Streamlit Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8080

## Usage

### Training a Model

#### Option 1: Via Airflow DAG (Recommended)

1. Navigate to http://localhost:8090
2. Login with `admin/admin`
3. Find the `train_model_dag` DAG
4. Click the play button to trigger the DAG
5. The DAG will:
   - Train a new model with the dataset in `/opt/airflow/data/`
   - Compare accuracy with the current Staging model
   - Promote the new model to Staging if it's better
   - Send email notifications on success/failure (if configured)

#### Option 2: Via REST API

```bash
curl -X POST http://localhost:8080/model/train \
  -F "file=@data/global_house_purchase_dataset.csv"
```

Response:
```json
{
  "message": "Model trained successfully!",
  "train_accuracy": 0.9234,
  "test_accuracy": 0.9156
}
```

### Making Predictions

```bash
curl -X POST http://localhost:8080/model/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inference_row": [
      "France", "Marseille", "Farmhouse", "Semi-Furnished",
      "991", "412935", "1989", "6", "6", "2", "1", "1", "1", "0",
      "10745", "193949", "15", "6545", "218986", "0.16", "1", "5", "6"
    ]
  }'
```

Response:
```json
{
  "prediction": 1,
  "message": "Prediction made successfully"
}
```

### Viewing Evidently Reports

1. Train a model (generates report automatically)
2. Navigate to http://localhost:8501
3. View the interactive data quality report
4. Report includes:
   - Dataset statistics
   - Missing values analysis
   - Feature distributions
   - Target/prediction comparison

### Registering Models in MLflow

```bash
curl -X POST http://localhost:8080/model/register \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "abc123...",
    "model_name": "House_Purchase_Approval_Model"
  }'
```

## API Documentation

### Interactive API Docs

Visit http://localhost:8080 for Swagger UI with interactive API documentation.

### Endpoints

#### `POST /model/train`

Train a new XGBoost model with provided CSV data.

**Request**: Multipart form data with CSV file
- `file`: CSV file with training data

**Response**:
```json
{
  "message": "Model trained successfully!",
  "train_accuracy": 0.92,
  "test_accuracy": 0.91
}
```

#### `POST /model/predict`

Make predictions on new data using the current Staging model.

**Request**: JSON
```json
{
  "inference_row": ["country", "city", "property_type", ...]
}
```

**Response**:
```json
{
  "prediction": 1,
  "message": "Prediction made successfully"
}
```

#### `POST /model/register`

Register a trained model in MLflow model registry.

**Request**: JSON
```json
{
  "run_id": "run_id_from_mlflow",
  "model_name": "House_Purchase_Approval_Model"
}
```

## Project Structure

```
HF/
├── app.py                              # Flask REST API application
├── MLModel.py                          # ML model class with training/inference
├── constants.py                        # Feature definitions and column names
├── streamlit_app.py                    # Streamlit dashboard for reports
├── Dockerfile                          # Docker image definition
├── docker-compose-airflow.yml          # Multi-container orchestration
├── requirements.txt                    # Python dependencies (local dev)
├── requirements_docker.txt             # Python dependencies (Docker)
├── dags/
│   └── train_model_dag.py             # Airflow DAG for automated training
├── data/
│   └── global_house_purchase_dataset.csv  # Training dataset
├── tests/
│   └── test_train_inference.py        # Unit tests
├── mlruns/                            # MLflow experiment data
├── mlartifacts/                       # MLflow artifact storage
└── logs/                              # Airflow logs
```

## Model Details

### Algorithm

**XGBoost Classifier** with the following hyperparameters:
- `max_depth`: 2
- `n_estimators`: 5

### Features

The model uses 23 features across several categories:

#### Categorical Features (4)
- `country`: Property location country
- `city`: Property location city
- `property_type`: Type of property (e.g., Farmhouse, Apartment)
- `furnishing_status`: Furnished, Semi-Furnished, Unfurnished

#### Numerical Features (6)
- `property_size_sqft`: Property size in square feet
- `price`: Property price
- `customer_salary`: Applicant's annual salary
- `loan_amount`: Requested loan amount
- `monthly_expenses`: Applicant's monthly expenses
- `down_payment`: Down payment amount

#### Ordinal Features (3)
- `constructed_year`: Year property was built
- `loan_tenure_years`: Loan duration in years
- `emi_to_income_ratio`: EMI to income ratio

#### Binary Features (3)
- `garage`: Has garage (0/1)
- `garden`: Has garden (0/1)
- `legal_cases_on_property`: Legal cases (0/1)

#### Discrete Features (7)
- `previous_owners`: Number of previous owners
- `rooms`: Number of rooms
- `bathrooms`: Number of bathrooms
- `crime_cases_reported`: Crime cases in area
- `nearby_schools`: Number of nearby schools
- `nearby_hospitals`: Number of nearby hospitals
- `nearby_malls`: Number of nearby malls

### Preprocessing Pipeline

1. **Drop ID Column**: Remove `property_id`
2. **Label Encoding**: Transform categorical features to integers
3. **Standard Scaling**: Normalize numerical and ordinal features
4. **Type Conversion**: Convert all features to float64

### Target Variable

- `decision`: Binary classification (0 = Rejected, 1 = Approved)

## Monitoring

### Evidently AI Reports

Automatically generated during training with:
- **DataSummaryPreset**: Comprehensive dataset statistics
  - Number of rows and columns
  - Missing values analysis
  - Column types and distributions
  - Target variable distribution
  - Prediction distribution

Reports are:
1. Saved as MLflow artifacts
2. Stored in shared Docker volume (`/app/reports/report.html`)
3. Displayed in Streamlit dashboard

### MLflow Tracking

Each training run logs:
- **Parameters**: Model hyperparameters
- **Metrics**: Train/test accuracy
- **Artifacts**: 
  - Trained model (XGBoost)
  - Label encoders (pickle)
  - Scaler (pickle)
  - Evidently report (HTML)

### Model Promotion Workflow

```
New Model Training
       ↓
Compare with Staging Model
       ↓
    Better?
    ↙    ↘
  Yes     No
   ↓       ↓
Promote  Keep Old
  ↓
Tag as "Staging"
```

## Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Tests

```bash
pytest tests/test_train_inference.py
```

### Running Flask API Locally

```bash
# Start MLflow server
mlflow server --host 127.0.0.1 --port 5102

# In another terminal, start Flask app
python app.py
```

API available at http://127.0.0.1:8080

### Rebuilding Docker Image

After code changes:

```bash
# Rebuild image
docker build -t mlsecops .

# Restart services
docker-compose -f docker-compose-airflow.yml down
docker-compose -f docker-compose-airflow.yml up -d
```

## Troubleshooting

### Issue: Streamlit shows "No report found"

**Solution**: Train a model first to generate the report
```bash
curl -X POST http://localhost:8080/model/train \
  -F "file=@data/global_house_purchase_dataset.csv"
```

### Issue: Docker containers won't start

**Solution**: Check logs
```bash
docker-compose -f docker-compose-airflow.yml logs <service-name>
```

Common causes:
- Port already in use: Change port mapping in docker-compose-airflow.yml
- Insufficient disk space: Clean up old images with `docker system prune`

### Issue: Airflow DAG not appearing

**Solution**: 
1. Check DAG file in `dags/` directory
2. Verify no syntax errors: `docker-compose -f docker-compose-airflow.yml exec airflow-webserver airflow dags list`
3. Restart scheduler: `docker-compose -f docker-compose-airflow.yml restart airflow-scheduler`

### Issue: Model prediction fails with "No staging model loaded"

**Solution**: Train and register a model first, or check MLflow UI to ensure a model is tagged with `stage=Staging`

### Issue: MLflow artifacts not accessible

**Solution**: 
- Check volume mounts in docker-compose-airflow.yml
- Verify `/app/mlruns` and `/app/mlartifacts` directories exist in container
- Ensure `--serve-artifacts` flag is set in MLflow server command

### Issue: Evidently import errors

**Solution**: 
1. Verify `evidently==0.7.17` is in requirements_docker.txt
2. Rebuild Docker image: `docker build -t mlsecops .`
3. Restart containers

## Configuration

### Environment Variables

Set in docker-compose-airflow.yml:

```yaml
environment:
  MLFLOW_TRACKING_URI: "file:///app/mlruns"
  AIRFLOW__CORE__EXECUTOR: LocalExecutor
  AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
```

### Volumes

- `shared-reports`: Shared volume for Evidently reports between mlsecops and streamlit containers
- `postgres-db-volume`: PostgreSQL database persistence
- `./dags:/opt/airflow/dags`: Airflow DAG files
- `./data:/opt/airflow/data`: Training data
- `./logs:/opt/airflow/logs`: Airflow execution logs

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## License

This project is part of an academic assignment for MLSecOps coursework.

## Authors

- https://github.com/Dinnye001Zz

**Last Updated**: December 2025
