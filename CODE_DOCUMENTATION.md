# Code Documentation

## Table of Contents

- [app.py - Flask REST API Application](#apppy---flask-rest-api-application)
- [MLModel.py - Machine Learning Model Class](#mlmodelpy---machine-learning-model-class)

---

## app.py - Flask REST API Application

### Overview

**Purpose**: Provides a RESTful API for training and inference of the house purchase approval prediction model.

**Framework**: Flask with Flask-RESTX for API documentation and request validation.

**Integration Points**:
- MLflow for experiment tracking and model registry
- MLModel class for ML operations
- File system for temporary CSV storage

---

### Module-Level Components

#### Imports

```python
from flask import Flask, request
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
import os, mlflow
import pandas as pd
from MLModel import MLModel
from mlflow import MlflowClient
from datetime import datetime
```

**Dependencies**:
- `flask`: Web framework for building REST APIs
- `flask_restx`: Extension providing Swagger UI and request parsing
- `werkzeug`: Utilities for file uploads
- `mlflow`: Experiment tracking and model registry
- `pandas`: Data manipulation for CSV processing
- `MLModel`: Custom class containing ML training/inference logic
- `datetime`: Timestamp generation for unique run names

---

#### MLflow Configuration

```python
mlflow.set_tracking_uri("http://0.0.0.0:5102")
```

**Purpose**: Configure MLflow tracking server endpoint

**Details**:
- Points to MLflow server running on port 5102
- Uses `0.0.0.0` to bind to all network interfaces (Docker compatibility)
- All experiment logs and artifacts will be sent to this server

---

```python
experiment_name = "default_experiment"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)
```

**Purpose**: Initialize default MLflow experiment

**Details**:
- Creates experiment if it doesn't exist (idempotent)
- All subsequent runs will be logged under this experiment
- Enables experiment-level organization of training runs

---

#### Flask Application Initialization

```python
app = Flask(__name__)
api = Api(app, version="1.0", title="API Documentation")
```

**Purpose**: Create Flask application with Swagger UI documentation

**Details**:
- `app`: Flask application instance
- `api`: Flask-RESTX API wrapper providing:
  - Automatic Swagger UI at root endpoint (`/`)
  - Request validation
  - Response serialization
  - API versioning

---

#### MLflow Client and Model Loading

```python
client = MlflowClient()

try:
    obj_mlmodel = MLModel(client=client)
    if obj_mlmodel.model is None:
        print("Warning: No model found in Staging stage. Training is still possible.")
except Exception as e:
    print(f"Warning: Could not load Staging model. Training is still possible. Error: {e}")
    obj_mlmodel = MLModel(client=client)
```

**Purpose**: Initialize MLflow client and load staging model if available

**Details**:
- `client`: MLflow client for interacting with model registry
- `obj_mlmodel`: Global MLModel instance used by all endpoints
- **Error Handling**:
  - If no staging model exists, creates empty MLModel instance
  - Prints warnings but doesn't fail - allows training from scratch
  - Model can be loaded later after first training run

**Design Decision**: Global instance allows model reuse across requests without reloading

---

#### API Models (Request Schemas)

```python
predict_model = api.model(
    "PredictModel",
    {
        "inference_row": fields.List(
            fields.String, required=True, description="A row of raw feature values for inference"
        )
    },
)
```

**Purpose**: Define JSON schema for prediction requests

**Schema**:
- **Field**: `inference_row`
- **Type**: List of strings
- **Required**: Yes
- **Description**: Array of 23 feature values in preprocessed order
- **Validation**: Automatic by Flask-RESTX
- **Swagger**: Appears in API documentation with interactive testing

---

```python
file_upload = api.parser()
file_upload.add_argument(
    "file",
    location="files",
    type=FileStorage,
    required=True,
    help="CSV file for training",
)
```

**Purpose**: Define multipart form schema for file uploads

**Schema**:
- **Field**: `file`
- **Location**: Multipart form files
- **Type**: `FileStorage` (Werkzeug file handler)
- **Required**: Yes
- **Validation**: Ensures file is present in request

---

#### API Namespace

```python
ns = api.namespace("model", description="Model operations")
```

**Purpose**: Group related endpoints under `/model` prefix

**Benefits**:
- Logical organization of endpoints
- Cleaner Swagger documentation
- Namespace-level configuration options

---

### API Endpoints

#### POST /model/train

```python
@ns.route("/train")
class Train(Resource):
    @ns.expect(file_upload)
    def post(self):
```

**Purpose**: Train a new XGBoost model with uploaded CSV data

**HTTP Method**: POST

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**: CSV file with training data
- **Required Fields**: 
  - `file`: CSV file containing features and target variable

**Response** (Success - 200):
```json
{
  "message": "Model Trained Successfully",
  "train_accuracy": 0.9234,
  "test_accuracy": 0.9156
}
```

**Response** (Error - 400):
```json
{
  "error": "Invalid file type"
}
```

**Response** (Error - 500):
```json
{
  "message": "MLFlow Error",
  "error": "error description"
}
```

---

##### Implementation Details

**Step 1: File Validation**

```python
args = file_upload.parse_args()
uploaded_file = args["file"]
if os.path.splitext(uploaded_file.filename)[1] != ".csv":
    return {"error": "Invalid file type"}, 400
```

- Parse multipart form data
- Extract uploaded file
- Validate file extension is `.csv`
- Return 400 if validation fails

---

**Step 2: File Persistence**

```python
data_path = "global_house_purchase_dataset.csv"
uploaded_file.save(data_path)
```

- Save uploaded file to local filesystem
- **File Location**: Current working directory (Docker container: `/app`)
- **Temporary Storage**: File deleted after training completes

---

**Step 3: MLflow Run Initialization**

```python
run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
model_name = "House_Purchase_Approval_Model"

with mlflow.start_run(run_name=run_name) as run:
```

- Generate unique run name with timestamp (e.g., `train_20250206_143052`)
- Define model registry name for versioning
- Start MLflow run context manager:
  - Automatically logs to configured experiment
  - Provides `run.info.run_id` for artifact linking
  - Auto-closes run on context exit or exception

---

**Step 4: Data Loading and Preprocessing**

```python
df = pd.read_csv(data_path)
df = obj_mlmodel.preprocess_pipeline(df)
input_example = df.drop(columns="decision").iloc[:1]
signature = mlflow.models.infer_signature(df.drop(columns="decision"), df["decision"])
```

- **Load**: Read CSV into pandas DataFrame
- **Preprocess**: 
  - Drop `property_id` column
  - Label encode categorical features
  - Standard scale numerical/ordinal features
  - Convert all to float64
  - Log preprocessing artifacts (encoders, scalers) to MLflow
- **Input Example**: Extract first row (without target) for MLflow model schema
- **Signature**: Infer input/output schema for model validation
  - Input: 23 features (float64)
  - Output: Binary classification (int)

---

**Step 5: Dataset Logging**

```python
mlflow.log_artifact(data_path, artifact_path="datasets")
```

- Upload CSV file to MLflow as artifact
- **Artifact Path**: `datasets/global_house_purchase_dataset.csv`
- **Purpose**: Reproducibility - track exact data used for training
- **Storage**: MLflow artifact store (configured backend)

---

**Step 6: Model Training**

```python
train_accuracy, test_accuracy, xgb = obj_mlmodel.train_and_save_model(df)
print("Model trained successfully.")
```

- Delegate training to MLModel class
- **Returns**:
  - `train_accuracy`: Accuracy on training set
  - `test_accuracy`: Accuracy on test set (50% split)
  - `xgb`: Trained XGBClassifier instance
- **Side Effects**:
  - Generates Evidently report
  - Logs report to MLflow
  - Saves report to shared volume

---

**Step 7: Metrics Logging**

```python
mlflow.log_metric("train_accuracy", train_accuracy)
mlflow.log_metric("test_accuracy", test_accuracy)
print("Metrics logged successfully.")
```

- Log accuracy metrics to MLflow run
- **Queryable**: Metrics can be compared across runs in MLflow UI
- **Visualization**: MLflow automatically generates metric plots

---

**Step 8: Model Logging**

```python
mlflow.sklearn.log_model(
    sk_model=xgb,
    name="model",
    input_example=input_example,
    signature=signature,
)
print("Model logged successfully.")
```

- Log trained model to MLflow
- **Parameters**:
  - `sk_model`: XGBoost model (scikit-learn compatible)
  - `name`: Artifact name (creates `model/` directory)
  - `input_example`: Sample input for schema validation
  - `signature`: Enforce input/output types
- **Output**: Model stored in MLflow artifact store
- **Format**: MLmodel format with:
  - Serialized model pickle
  - Conda environment
  - Requirements file
  - Model metadata

---

**Step 9: Model Registration**

```python
model_uri = f"runs:/{run.info.run_id}/model"
registered_model_version = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)
print("Model registered successfully.")
```

- Register model in MLflow Model Registry
- **URI Format**: `runs:/<run_id>/model` points to logged model
- **Versioning**: Automatically increments version number
- **Returns**: `ModelVersion` object with version number
- **Purpose**: Enable model lifecycle management (Staging → Production)

---

**Step 10: Model Staging**

```python
mlflow_client = mlflow.tracking.MlflowClient()
mlflow_client.set_model_version_tag(
    name=model_name,
    version=registered_model_version.version,
    key="stage",
    value="Staging"
)
print("Model transitioned to Staging successfully.")
```

- Tag model version with `stage=Staging`
- **Custom Tag Approach**: MLflow 2.x deprecated stages API
- **Purpose**: Mark model as current staging model
- **Workflow**: Airflow DAG promotes to Production if accuracy improves
- **Loading**: `MLModel.__init__` searches for models with `stage=Staging` tag

---

**Step 11: Cleanup**

```python
os.remove(data_path)
```

- Delete temporary CSV file
- **Reason**: File already logged to MLflow, local copy not needed
- **Disk Management**: Prevents accumulation of training data

---

**Step 12: Response**

```python
return {
    "message": "Model Trained Successfully",
    "train_accuracy": train_accuracy,
    "test_accuracy": test_accuracy,
}, 200
```

- Return success response with metrics
- **HTTP Status**: 200 OK
- **Content**: JSON with accuracy values

---

##### Error Handling

```python
except Exception as mfe:
    return {"message": "MLFlow Error", "error": str(mfe)}, 500
except Exception as e:
    return {"message": "Internal Server Error", "error": str(e)}, 500
```

**MLflow Errors** (500):
- Tracking server unavailable
- Artifact storage errors
- Model registration failures
- Tag update failures

**General Errors** (500):
- CSV parsing errors
- Preprocessing failures
- Training crashes
- Disk I/O errors

**Design Decision**: Broad exception handling ensures API returns JSON error (not HTML 500 page)

---

#### POST /model/predict

```python
@ns.route('/predict')
class Predict(Resource):
    @api.expect(predict_model)
    def post(self):
```

**Purpose**: Make predictions on new data using the current staging model

**HTTP Method**: POST

**Request**:
- **Content-Type**: `application/json`
- **Body**:
```json
{
  "inference_row": [
    "France", "Marseille", "Farmhouse", "Semi-Furnished",
    "991", "412935", "1989", "6", "6", "2", "1", "1", "1", "0",
    "10745", "193949", "15", "6545", "218986", "0.16", "1", "5", "6"
  ]
}
```

**Response** (Success - 200):
```json
{
  "message": "Inference Successful",
  "prediction": 1
}
```

**Response** (Error - 400):
```json
{
  "error": "No inference_row found"
}
```
or
```json
{
  "error": "No model is loaded. Train a model before inference."
}
```

**Response** (Error - 500):
```json
{
  "message": "Internal Server Error",
  "error": "error description"
}
```

---

##### Implementation Details

**Step 1: Request Validation**

```python
data = request.get_json()
if "inference_row" not in data:
    return {"error": "No inference_row found"}, 400
```

- Parse JSON request body
- Validate `inference_row` field exists
- Return 400 if missing

---

**Step 2: Model Check**

```python
infer_array = data["inference_row"]
if obj_mlmodel.model is None:
    return {"error": "No model is loaded. Train a model before inference."}, 400
```

- Extract feature array
- Check if staging model is loaded
- **Guard Clause**: Prevents inference with uninitialized model
- Return 400 if no model available

---

**Step 3: MLflow Run Initialization**

```python
run_name = f"infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
with mlflow.start_run(run_name=run_name) as run:
```

- Generate unique inference run name (e.g., `infer_20250206_143052`)
- Start MLflow run to log inference parameters
- **Purpose**: Track all predictions for monitoring and debugging

---

**Step 4: Prediction**

```python
y_pred = obj_mlmodel.predict(infer_array)
```

- Delegate prediction to MLModel class
- **Process**:
  1. Convert array to DataFrame
  2. Apply label encoding (categorical features)
  3. Apply standard scaling (numerical features)
  4. Ensure correct column order
  5. Run XGBoost prediction
- **Returns**: Integer prediction (0 or 1)

---

**Step 5: Logging**

```python
mlflow.log_param("inference_input", infer_array)
mlflow.log_param("inference_output", y_pred)
```

- Log input and output to MLflow
- **Purpose**: 
  - Track prediction distribution over time
  - Debug incorrect predictions
  - Monitor for data drift
- **Queryable**: Can search MLflow for specific input/output combinations

---

**Step 6: Response**

```python
return {"message": "Inference Successful", "prediction": y_pred}, 200
```

- Return prediction result
- **HTTP Status**: 200 OK
- **Prediction Values**:
  - `0`: Application rejected
  - `1`: Application approved

---

##### Error Handling

```python
except Exception as e:
    return {"message": "Internal Server Error", "error": str(e)}, 500
```

**Potential Errors** (500):
- Invalid feature values (e.g., unknown categorical value)
- Preprocessing failures (encoding/scaling errors)
- Model prediction errors
- MLflow logging failures

---

### Application Entry Point

```python
if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=8080, debug=False)
```

**Configuration**:
- **Host**: `0.0.0.0` - Bind to all network interfaces (required for Docker)
- **Port**: `8080` - Exposed port for API access
- **Debug**: `False` - Production mode (no auto-reload, better performance)

**Docker Deployment**:
- Runs inside `mlsecops` container
- Accessible via `http://localhost:8080` (host)
- Accessible via `http://mlsecops:8080` (Docker network)

---

## MLModel.py - Machine Learning Model Class

### Overview

**Purpose**: Encapsulates all machine learning operations including training, preprocessing, inference, and monitoring.

**Responsibilities**:
- Model lifecycle management (loading/saving)
- Data preprocessing (encoding, scaling)
- Model training with XGBoost
- Accuracy evaluation
- Evidently report generation
- MLflow artifact management

---

### Imports

```python
import pickle, mlflow, tempfile, os
from constants import (
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    ORDINAL_COLS,
    COLUMN_ORDER_AFTER_PREPROCESSING,
)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mlflow.artifacts import download_artifacts
from evidently import Report
from evidently.presets import DataSummaryPreset
import streamlit as st
import streamlit.components.v1 as components
```

**Dependencies**:
- `pickle`: Serialization of encoders/scalers
- `mlflow`: Experiment tracking and artifact storage
- `tempfile`: Temporary directory for artifact staging
- `os`: File system operations
- `constants`: Feature definitions from separate module
- `xgboost`: Gradient boosting classifier
- `sklearn`: Preprocessing, train/test split, metrics
- `pandas`: Data manipulation
- `evidently`: Data quality monitoring reports
- `streamlit`: UI components (imported but not used in class methods)

**Note**: Streamlit imports are present but unused - likely legacy code from UI integration attempts.

---

### Class Definition

```python
class MLModel:
```

**Design Pattern**: State-holding service class

**Instance Variables**:
- `client`: MLflow client for registry operations
- `model`: Trained XGBoost classifier (or None)
- `label_encoder`: Dictionary of LabelEncoder instances per categorical feature
- `scaler`: StandardScaler instance for numerical features

---

### Constructor

```python
def __init__(self, client):
    self.client = client
    self.model = None
    self.label_encoder = None
    self.scaler = None
    self.load_staging_model()
    if isinstance(self.label_encoder, dict):
        for col, le in self.label_encoder.items:
            print(f"LabelEncoder for '{col}': {le.classes_}")
    print("Loaded label encoders:", self.label_encoder)
    print("Loaded scaler:", self.scaler)
```

**Purpose**: Initialize model instance and load staging model if available

**Parameters**:
- `client` (MlflowClient): MLflow client for model registry access

**Process**:
1. Store MLflow client reference
2. Initialize instance variables to None
3. Attempt to load staging model from registry
4. Print loaded artifacts for debugging

**Design Decision**: Constructor handles model loading automatically, making the class ready for inference immediately after instantiation.

---

### load_staging_model Method

```python
def load_staging_model(self):
    try:
        print("Searching for registered models...")
        for model in self.client.search_registered_models():
            for version in model.latest_versions:
                tags = version.tags
                if tags.get("stage") == "Staging":
                    model_url = version.source
                    print("Loading model from:", model_url)
                    self.model = mlflow.sklearn.load_model(model_url)
                    run_id = version.run_id
                    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
                    print("Loading artifacts from:", artifact_uri)
                    self.load_artifacts(artifact_uri)
                    print("Staging model loaded successfully.")
                    return
        print("No model found in Staging stage.")
    except Exception as e:
        import traceback
        print("Error loading staging model:", e)
        traceback.print_exc()
```

**Purpose**: Search for and load the current staging model from MLflow registry

**Algorithm**:
1. Iterate through all registered models in MLflow
2. For each model, check latest versions
3. Find version with `stage=Staging` tag
4. Load model from source URI
5. Extract run_id to locate artifacts
6. Load preprocessing artifacts (encoders, scalers)
7. Return on first match (short-circuit)

**Return Behavior**:
- **Success**: `self.model`, `self.label_encoder`, `self.scaler` populated
- **No Staging Model**: Instance variables remain None
- **Error**: Prints traceback but doesn't raise exception (graceful degradation)

**Design Decision**: Non-failing initialization allows application startup even without trained models.

---

### load_artifacts Method

```python
def load_artifacts(self, artifact_uri):
    try:
        # Load label encoder
        print("Loading label encoders from:", artifact_uri)
        label_encoder_path = download_artifacts(artifact_uri=f"{artifact_uri}/label_encoder.pkl")
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Load scaler
        scaler_path = download_artifacts(artifact_uri=f"{artifact_uri}/scaler.pkl")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print("Artifacts loaded successfully.")

    except Exception as e:
        print(f"Error loading artifacts: {e}")
```

**Purpose**: Download and deserialize preprocessing artifacts from MLflow

**Parameters**:
- `artifact_uri` (str): Base URI for run artifacts (e.g., `file:///app/mlruns/0/abc123/artifacts`)

**Process**:
1. **Label Encoders**:
   - Download `label_encoder.pkl` from artifact URI
   - Deserialize to dictionary: `{column_name: LabelEncoder}`
   - Store in `self.label_encoder`

2. **Scaler**:
   - Download `scaler.pkl` from artifact URI
   - Deserialize to StandardScaler instance
   - Store in `self.scaler`

**Error Handling**:
- Prints error but doesn't raise exception
- **Consequence**: Inference will fail if artifacts missing
- **Design Tradeoff**: Allows partial initialization for debugging

**Critical Dependency**: These artifacts must be present for inference to work correctly.

---

### save_model Static Method

```python
@staticmethod
def save_model(model, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(model, file)
```

**Purpose**: Serialize model to disk using pickle

**Parameters**:
- `model` (Any): Object to serialize (model, encoder, scaler)
- `file_path` (str): Destination file path

**Usage**: Legacy method - currently unused (MLflow handles model persistence)

---

### load_model Static Method

```python
@staticmethod
def load_model(file_path):
    with open(file_path, "rb") as file:
        model = pickle.load(file)
    return model
```

**Purpose**: Deserialize model from disk using pickle

**Parameters**:
- `file_path` (str): Source file path

**Returns**: Deserialized object

**Usage**: Legacy method - currently unused (MLflow handles model loading)

---

### preprocess_pipeline Method

```python
def preprocess_pipeline(self, df):
```

**Purpose**: Transform raw training data into model-ready format

**Parameters**:
- `df` (DataFrame): Raw training data with all original columns

**Returns**: 
- `df` (DataFrame): Preprocessed data ready for training

**Side Effects**:
- Sets `self.label_encoder` dictionary
- Sets `self.scaler` instance
- Logs artifacts to MLflow

---

#### Implementation Steps

**Step 1: Remove ID Column**

```python
df = df.drop("property_id", axis=1)
```

- Remove non-predictive identifier
- **Reason**: `property_id` has no correlation with target variable

---

**Step 2: Label Encoding**

```python
label_encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

self.label_encoder = label_encoders
```

- Transform categorical features to integers
- **CATEGORICAL_COLS**: `['country', 'city', 'property_type', 'furnishing_status']`
- **Process**:
  - Create new LabelEncoder for each column
  - Fit on unique values in column
  - Transform values to integers (0, 1, 2, ...)
  - Store encoder in dictionary for inverse transform during inference
- **Example**: `'France'` → `0`, `'Germany'` → `1`

---

**Step 3: Type Conversion (Categorical)**

```python
for col in CATEGORICAL_COLS:
    if col in df.columns:
        df[col] = df[col].astype(int)
```

- Ensure encoded values are integers (not objects)
- **Reason**: XGBoost requires numerical dtypes

---

**Step 4: Standard Scaling**

```python
all_numerical_to_scale = NUMERICAL_COLS + ORDINAL_COLS
scaler = StandardScaler()
df[all_numerical_to_scale] = scaler.fit_transform(df[all_numerical_to_scale])
self.scaler = scaler
```

- Standardize continuous and ordinal features to zero mean and unit variance
- **Formula**: `(x - mean) / std_dev`
- **Columns**: 9 features (6 numerical + 3 ordinal)
- **Purpose**:
  - Improve gradient descent convergence
  - Prevent features with large ranges from dominating
  - XGBoost benefits from scaled features

---

**Step 5: Type Conversion (Float)**

```python
for col in df.select_dtypes(include=['int']).columns:
    df[col] = df[col].astype(float)

for col in df.columns:
    df[col] = df[col].astype(float)
```

- Convert all columns to float64
- **Reason**: 
  - MLflow schema inference requires consistent types
  - Prevents int/float mismatch errors during inference
  - XGBoost DMatrix construction expects float

---

**Step 6: Artifact Logging**

```python
with tempfile.TemporaryDirectory() as tmpdir:
    label_encoder_path = os.path.join(tmpdir, "label_encoder.pkl")
    with open(label_encoder_path, "wb") as f:
        pickle.dump(self.label_encoder, f)
    mlflow.log_artifact(label_encoder_path)

    scaler_path = os.path.join(tmpdir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(self.scaler, f)
    mlflow.log_artifact(scaler_path)
```

- Serialize and upload preprocessing artifacts to MLflow
- **Temporary Directory**: Auto-cleanup on context exit
- **Artifacts**:
  - `label_encoder.pkl`: Dictionary of LabelEncoders
  - `scaler.pkl`: StandardScaler instance
- **Purpose**: Enable inference with same preprocessing

---

### train_and_save_model Method

```python
def train_and_save_model(self, df):
    print("Starting train_and_save_model")

    X = df.drop("decision", axis=1)
    y = df["decision"]

    print("Split X and y")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    print("Split train/test")

    xgb = XGBClassifier(max_depth=2, n_estimators=5)
    xgb.fit(X_train, y_train)
    print("Model fit")

    self.model = xgb

    train_accuracy, test_accuracy = self.get_accuracy(
        X_train, X_test, y_train, y_test
    )
    print("Got accuracy")

    return train_accuracy, test_accuracy, xgb
```

**Purpose**: Train XGBoost classifier and evaluate performance

**Parameters**:
- `df` (DataFrame): Preprocessed training data

**Returns**:
- `train_accuracy` (float): Accuracy on training set
- `test_accuracy` (float): Accuracy on test set
- `xgb` (XGBClassifier): Trained model instance

**Side Effects**:
- Sets `self.model`
- Generates Evidently report (via `get_accuracy`)
- Logs report to MLflow

---

#### Implementation Details

**Feature/Target Split**:
```python
X = df.drop("decision", axis=1)
y = df["decision"]
```
- **X**: 23 features (all except target)
- **y**: Binary target (0 or 1)

**Train/Test Split**:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)
```
- **Split Ratio**: 50/50 (unusual but specified)
- **Random State**: 42 (reproducibility)
- **Stratification**: Not used (could improve for imbalanced data)

**Model Configuration**:
```python
xgb = XGBClassifier(max_depth=2, n_estimators=5)
```
- **Algorithm**: Gradient Boosted Decision Trees
- **Hyperparameters**:
  - `max_depth=2`: Shallow trees (prevent overfitting)
  - `n_estimators=5`: Only 5 boosting rounds (fast training)
- **Note**: Simple configuration for demo purposes - production would use hyperparameter tuning

**Training**:
```python
xgb.fit(X_train, y_train)
```
- Standard scikit-learn API
- **Process**: Builds 5 decision trees sequentially, each correcting errors of previous

**Evaluation**:
```python
train_accuracy, test_accuracy = self.get_accuracy(
    X_train, X_test, y_train, y_test
)
```
- Delegates to `get_accuracy` method
- **Side Effect**: Generates Evidently report

---

### preprocess_pipeline_inference Method

```python
def preprocess_pipeline_inference(self, sample_data):
```

**Purpose**: Transform raw inference data using fitted encoders/scalers

**Parameters**:
- `sample_data` (list): Raw feature values in order defined by `COLUMN_ORDER_AFTER_PREPROCESSING`

**Returns**:
- `sample_data` (DataFrame): Single-row DataFrame ready for prediction

**Critical Difference from Training**: Uses `transform` (not `fit_transform`) to apply existing encoders/scalers

---

#### Implementation Steps

**Step 1: DataFrame Creation**

```python
try:
    sample_data = pd.DataFrame([sample_data], columns=COLUMN_ORDER_AFTER_PREPROCESSING)
except Exception as e:
    print("Error creating DataFrame:", e)
    raise
```

- Convert list to DataFrame with column names
- **Columns**: 23 features in specific order (without `property_id` and `decision`)
- **Error Handling**: Re-raises exception with context

---

**Step 2: Label Encoding**

```python
if self.label_encoder:
    for col in CATEGORICAL_COLS:
        if col in sample_data.columns:
            sample_data[col] = self.label_encoder[col].transform(sample_data[col])
```

- Apply fitted LabelEncoders to categorical columns
- **Critical**: Uses `transform` (not `fit_transform`)
- **Preserves Training Mapping**: `'France'` → same integer as during training
- **Error if Unknown Category**: Will fail if new categorical value not seen during training

---

**Step 3: Type Conversion (Categorical)**

```python
for col in CATEGORICAL_COLS:
    if col in sample_data.columns:
        sample_data[col] = sample_data[col].astype(int)
```

- Convert encoded values to integers

---

**Step 4: Standard Scaling**

```python
if self.scaler:
    all_numerical_to_scale = NUMERICAL_COLS + ORDINAL_COLS
    sample_data[all_numerical_to_scale] = self.scaler.transform(
        sample_data[all_numerical_to_scale]
    )
```

- Apply fitted StandardScaler
- **Critical**: Uses `transform` (not `fit_transform`)
- **Uses Training Statistics**: Applies mean/std from training data

---

**Step 5: Type Conversion (Float)**

```python
for col in sample_data.select_dtypes(include=['int']).columns:
    sample_data[col] = sample_data[col].astype(float)

for col in sample_data.columns:
    sample_data[col] = sample_data[col].astype(float)
```

- Ensure all columns are float64 (match training schema)

---

**Step 6: Column Ordering**

```python
expected_column_order = COLUMN_ORDER_AFTER_PREPROCESSING
columns_to_use = [col for col in expected_column_order if col in sample_data.columns]
sample_data = sample_data[columns_to_use]
```

- Reorder columns to match training data
- **Critical**: XGBoost expects features in exact same order
- **Filter**: Only use columns that exist (defensive programming)

---

### get_accuracy_full Method

```python
def get_accuracy_full(self, X, y):
    y_pred = self.model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print("Accuracy: ", accuracy)
    return accuracy
```

**Purpose**: Calculate accuracy on any dataset

**Parameters**:
- `X` (DataFrame): Features
- `y` (Series): True labels

**Returns**: 
- `accuracy` (float): Accuracy score (0.0 to 1.0)

**Usage**: Utility method for ad-hoc evaluation

---

### get_accuracy Method

```python
def get_accuracy(self, X_train, X_test, y_train, y_test):
    y_train_pred = self.model.predict(X_train)
    y_test_pred = self.model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Train Accuracy: ", train_accuracy)
    print("Test Accuracy: ", test_accuracy)

    # Generate Evidently Data Summary Report
    test_results = X_test.copy()
    test_results['target'] = y_test.values
    test_results['prediction'] = y_test_pred
    
    report = Report(metrics=[DataSummaryPreset()])
    my_eval = report.run(current_data=test_results, reference_data=None)
    
    # Save report as MLflow artifact and in shared volume
    os.makedirs('/app/reports', exist_ok=True)
    report_path = '/app/reports/report.html'
    my_eval.save_html(report_path)
    mlflow.log_artifact(report_path)

    return train_accuracy, test_accuracy
```

**Purpose**: Evaluate model and generate monitoring report

**Parameters**:
- `X_train`, `X_test` (DataFrame): Train/test features
- `y_train`, `y_test` (Series): Train/test labels

**Returns**:
- `train_accuracy` (float): Training set accuracy
- `test_accuracy` (float): Test set accuracy

**Side Effects**:
- Prints accuracies
- Generates Evidently HTML report
- Saves report to shared Docker volume
- Logs report to MLflow

---

#### Evidently Report Generation

**Step 1: Combine Features, Target, and Predictions**

```python
test_results = X_test.copy()
test_results['target'] = y_test.values
test_results['prediction'] = y_test_pred
```

- Create DataFrame with all data for Evidently
- **Columns**: 23 features + `target` + `prediction`
- **Purpose**: Evidently analyzes relationship between features, targets, and predictions

---

**Step 2: Create Report**

```python
report = Report(metrics=[DataSummaryPreset()])
my_eval = report.run(current_data=test_results, reference_data=None)
```

- **Report Type**: DataSummaryPreset
- **Metrics Included**:
  - Dataset size
  - Missing values
  - Column types
  - Feature distributions
  - Target distribution
  - Prediction distribution
- **Reference Data**: None (not comparing to baseline)
- **Returns**: Evaluation object with `save_html()` method

---

**Step 3: Save Report**

```python
os.makedirs('/app/reports', exist_ok=True)
report_path = '/app/reports/report.html'
my_eval.save_html(report_path)
mlflow.log_artifact(report_path)
```

- **Directory**: `/app/reports` (shared Docker volume)
- **File**: `report.html` (interactive HTML report)
- **Shared Access**: 
  - `mlsecops` container generates report
  - `streamlit` container reads report
  - Both mount same volume
- **MLflow**: Also log to experiment for historical tracking

---

### predict Method

```python
def predict(self, inference_row):
    if self.model is None:
        return {'error': 'No staging model is loaded'}, 400

    processed_data = self.preprocess_pipeline_inference(inference_row)
    prediction = self.model.predict(processed_data)
    return int(prediction)
```

**Purpose**: Make prediction on single inference sample

**Parameters**:
- `inference_row` (list): Raw feature values (23 elements)

**Returns**:
- `int`: Prediction (0 or 1)
- OR `dict`: Error message with HTTP status code

**Process**:
1. **Validation**: Check if model is loaded
2. **Preprocessing**: Apply encoders and scalers
3. **Prediction**: XGBoost classification
4. **Type Conversion**: Convert numpy int to Python int

**Error Handling**:
- Returns tuple `(error_dict, status_code)` if model not loaded
- Caller (Flask endpoint) handles this return format

**Design Note**: Inconsistent return type (int vs tuple) - could be improved with custom exception

---

## constants.py - Feature Definitions

### Overview

**Purpose**: Centralized configuration for feature engineering and preprocessing

**Pattern**: Constants module for maintainability and DRY principle

---

### Feature Categories

#### CATEGORICAL_COLS
```python
CATEGORICAL_COLS = ['country', 'city', 'property_type', 'furnishing_status']
```
- **Count**: 4 features
- **Encoding**: LabelEncoder (string → integer)
- **Examples**: 'France' → 0, 'Marseille' → 1, 'Farmhouse' → 2

#### NUMERICAL_COLS
```python
NUMERICAL_COLS = [
    'property_size_sqft', 'price', 'customer_salary',
    'loan_amount', 'monthly_expenses', 'down_payment'
]
```
- **Count**: 6 features
- **Scaling**: StandardScaler (zero mean, unit variance)
- **Type**: Continuous values

#### ORDINAL_COLS
```python
ORDINAL_COLS = ['constructed_year', 'loan_tenure_years', 'emi_to_income_ratio']
```
- **Count**: 3 features
- **Scaling**: StandardScaler (treated as continuous)
- **Nature**: Ordered numerical values

#### BINARY_COLS
```python
BINARY_COLS = ['garage', 'garden', 'legal_cases_on_property']
```
- **Count**: 3 features
- **Values**: 0 or 1
- **Processing**: No encoding needed

#### DISCRETE_COLS
```python
DISCRETE_COLS = [
    'previous_owners', 'rooms', 'bathrooms', 'crime_cases_reported',
    'satisfaction_score', 'neighbourhood_rating', 'connectivity_score'
]
```
- **Count**: 7 features
- **Type**: Integer counts/ratings
- **Processing**: No special encoding

#### COLUMN_ORDER_AFTER_PREPROCESSING
```python
COLUMN_ORDER_AFTER_PREPROCESSING = [
    'country', 'city', 'property_type', 'furnishing_status',
    'property_size_sqft', 'price', 'constructed_year', 'previous_owners',
    'rooms', 'bathrooms', 'garage', 'garden', 'crime_cases_reported',
    'legal_cases_on_property', 'customer_salary', 'loan_amount',
    'loan_tenure_years', 'monthly_expenses', 'down_payment',
    'emi_to_income_ratio', 'satisfaction_score', 'neighbourhood_rating',
    'connectivity_score'
]
```
- **Count**: 23 features (excludes `property_id` and `decision`)
- **Purpose**: Enforce consistent column ordering for XGBoost
- **Critical**: Inference must match this exact order

---

## streamlit_app.py - Report Visualization

### Overview

**Purpose**: Display Evidently HTML reports in interactive web UI

**Framework**: Streamlit for rapid dashboard development

---

### Implementation

```python
import streamlit as st
import streamlit.components.v1 as components

st.title("Evidently Report Viewer")

report_file = "/app/reports/report.html"

try:
    with open(report_file, "r") as f:
        html_content = f.read()
    components.html(html_content, height=800, width=1000, scrolling=True)
except FileNotFoundError:
    st.warning("No report found. Please train a model first by calling the training API.")
    st.info("POST to http://localhost:8080/")
```

### Components

**Title**: Display app header

**Report Loading**:
- **Path**: `/app/reports/report.html` (shared Docker volume)
- **Read**: Load full HTML content into memory
- **Render**: Use `components.html()` to embed in Streamlit app

**Error Handling**:
- **FileNotFoundError**: Show user-friendly warning
- **Guidance**: Provide instructions for generating report

**Configuration**:
- **Height**: 800px (scrollable content)
- **Width**: 1000px (wide format for tables)
- **Scrolling**: Enabled for large reports

**Deployment**: Runs in separate Docker container on port 8501

---

## train_model_dag.py - Airflow Orchestration

### Overview

**Purpose**: Automated daily model training and promotion workflow

**Pattern**: Airflow DAG with branching logic for conditional notifications

---

### Configuration

```python
mlflow.set_tracking_uri("http://mlsecops:5102")
client = MlflowClient()
model_name = "House_Purchase_Approval_Model"
csv_file_path = "/opt/airflow/data/global_house_purchase_dataset.csv"
```

**MLflow URI**: Uses Docker service name for container-to-container communication

**Model Name**: Must match registration name in `app.py`

**Data Path**: Mounted volume from host to Airflow container

---

### Task Functions

#### train_model()

**Purpose**: Train model via API and compare with staging model

**Process**:
1. **API Call**: POST CSV to `/model/train` endpoint
2. **Error Check**: Raise exception if training fails
3. **Get New Accuracy**: Extract test accuracy from response
4. **Find Staging Model**: Search registry for model with `stage=Staging` tag
5. **Compare Accuracy**: Check if new model is better
6. **Promote**: Archive old staging, new model remains in staging
7. **Return**: `True` if new model better/equal, `False` otherwise

**Logic**:
```python
if new_accuracy >= staging_accuracy:
    # Archive old model
    client.set_model_version_tag(name=model_name, version=staging_model.version,
                                   key="stage", value="Archived")
    return True
else:
    return False
```

**Network Communication**: Uses `http://mlsecops:8080` (Docker internal)

---

#### branch_decision(**kwargs)

**Purpose**: Decide whether to send notification based on training result

**XCom**: Retrieves return value from `train_and_compare_model` task

**Logic**:
- Return `'send_notification'` if new model worse (False)
- Return `'skip_notification'` if new model better (True)

**Pattern**: BranchPythonOperator uses return value as next task ID

---

### DAG Definition

```python
with DAG(
    dag_id="daily_model_training",
    start_date=datetime(2023, 11, 1),
    schedule_interval="0 2 * * *",
    catchup=False,
) as dag:
```

**Schedule**: Runs daily at 2 AM (cron: `0 2 * * *`)

**Catchup**: Disabled (don't backfill missed runs)

**Start Date**: Historical start (won't execute for past dates due to catchup=False)

---

### Tasks

#### train_and_compare_task
```python
PythonOperator(
    task_id="train_and_compare_model",
    python_callable=train_model,
    retries=3,
    retry_delay=timedelta(minutes=5),
)
```
- **Retries**: 3 attempts on failure
- **Retry Delay**: 5 minutes between attempts
- **Failure Scenarios**: API down, training errors, MLflow issues

#### branch_task
```python
BranchPythonOperator(
    task_id="branch_decision",
    python_callable=branch_decision,
    provide_context=True,
)
```
- **Context**: Provides `ti` (TaskInstance) for XCom access
- **Branching**: Routes to one of two downstream tasks

#### notification_task
```python
PythonOperator(
    task_id="send_notification",
    python_callable=lambda: print("New model accuracy is LOWER than staging model!"),
)
```
- **Trigger**: New model worse than staging
- **Action**: Print warning (production would send email/Slack)

#### skip_notification
```python
PythonOperator(
    task_id="skip_notification",
    python_callable=lambda: print("New model is better or equal."),
)
```
- **Trigger**: New model better or equal
- **Action**: Success message

### Task Dependencies

```python
train_and_compare_task >> branch_task
branch_task >> [notification_task, skip_notification]
```

**Flow**:
1. Train and compare model
2. Branch based on result
3. Execute one of two notification tasks

**Pattern**: Fan-out after branching (only one branch executes)

---

## Dockerfile - Container Image

### Overview

**Purpose**: Build production container with all dependencies

**Base Image**: Python 3.12 official image

---

### Build Steps

```dockerfile
FROM python:3.12
WORKDIR /app

COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

COPY app.py .
COPY MLModel.py .
COPY constants.py .
COPY streamlit_app.py .
COPY mlruns ./mlruns
COPY mlartifacts ./mlartifacts

ENV MLFLOW_TRACKING_URI="file:///app/mlruns"

CMD ["bash", "-c", "mlflow server --host 0.0.0.0 --port 5102 --backend-store-uri file:/app/mlruns --default-artifact-root /app/mlartifacts --serve-artifacts & python app.py"]
```

**Working Directory**: `/app` (all files copied here)

**Dependencies**: Install from `requirements_docker.txt` without cache (smaller image)

**Application Files**: Copy Python modules

**MLflow Storage**: Copy existing experiment data (optional, usually mounted as volume)

**Environment Variable**: Set MLflow URI for application

**Startup Command**:
- Run MLflow server in background (`&`)
- Run Flask app in foreground
- **MLflow Flags**:
  - `--host 0.0.0.0`: Accept external connections
  - `--port 5102`: MLflow UI/API port
  - `--backend-store-uri`: Experiment metadata storage
  - `--default-artifact-root`: Artifact storage path
  - `--serve-artifacts`: Enable artifact serving

---

## docker-compose-airflow.yml - Multi-Container Orchestration

### Overview

**Purpose**: Define and configure all services for the ML pipeline

**Pattern**: Infrastructure as Code for reproducible deployments

---

### Services

#### postgres
- **Image**: PostgreSQL 13
- **Purpose**: Airflow metadata database
- **Credentials**: airflow/airflow
- **Volume**: `postgres-db-volume` for persistence
- **Health Check**: `pg_isready` command

#### airflow-webserver
- **Image**: apache/airflow:2.7.1
- **Port**: 8090 (host) → 8080 (container)
- **Purpose**: Airflow web UI
- **Dependencies**: postgres, airflow-init
- **Additional Packages**: mlflow, requests (via `_PIP_ADDITIONAL_REQUIREMENTS`)

#### airflow-scheduler
- **Image**: apache/airflow:2.7.1
- **Purpose**: Execute scheduled DAGs
- **Dependencies**: postgres, airflow-init

#### airflow-init
- **Purpose**: One-time initialization (database migration, user creation)
- **Creates**: Admin user (admin/admin)
- **Runs Once**: `service_completed_successfully` condition

#### mlsecops
- **Image**: mlsecops (custom built)
- **Ports**: 8080 (Flask API), 5102 (MLflow)
- **Volume**: `shared-reports` for Evidently reports
- **Purpose**: Run ML training/inference and MLflow server

#### streamlit
- **Image**: mlsecops (same image, different command)
- **Port**: 8501
- **Volume**: `shared-reports` (read Evidently reports)
- **Command**: `streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0`

### Volumes

**postgres-db-volume**: Database persistence across restarts

**shared-reports**: Shared storage for Evidently HTML reports between mlsecops and streamlit containers

### Networks

**default**: Bridge network for inter-container communication (e.g., `http://mlsecops:8080`)

---

