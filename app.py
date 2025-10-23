from flask import Flask, request
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
import os, mlflow
import pandas as pd
from MLModel import MLModel
from mlflow import MlflowClient
from datetime import datetime
from constants import CATEGORICAL_COLS

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5102")

# Set default experiment
experiment_name = "default_experiment"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Flask REST API setup
app = Flask(__name__)
api = Api(app, version="1.0", title="API Documentation")

# Initialize MLflow client
client = MlflowClient()

# Attempt to load the latest Staging model version if it exists
try:
    obj_mlmodel = MLModel(client=client)
    if obj_mlmodel.model is None:
        print("Warning: No model found in Staging stage. Training is still possible.")
except Exception as e:
    print(f"Warning: Could not load Staging model. Training is still possible. Error: {e}")
    obj_mlmodel = MLModel(client=client) # create an empty MlModel instance



# Define input model for prediction
predict_model = api.model(
    "PredictModel",
    {
        "inference_row": fields.List(
            fields.String, required=True, description="A row of raw feature values for inference"
        )
    },
)

# File upload parser
file_upload = api.parser()
file_upload.add_argument(
    "file",
    location="files",
    type=FileStorage,
    required=True,
    help="CSV file for training",
)

ns = api.namespace("model", description="Model operations")


# train input
@ns.route("/train")
class Train(Resource):
    @ns.expect(file_upload)
    def post(self):
        args = file_upload.parse_args()
        uploaded_file = args["file"]
        if os.path.splitext(uploaded_file.filename)[1] != ".csv":
            return {"error": "Invalid file type"}, 400

        data_path = "global_house_purchase_dataset.csv"
        uploaded_file.save(data_path)

        try:
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_name = "House_Purchase_Approval_Model"

            with mlflow.start_run(run_name=run_name) as run:
                #load and preprocess data
                df = pd.read_csv(data_path)
                df = obj_mlmodel.preprocess_pipeline(df)  # preprocess first!
                input_example = df.drop(columns="decision").iloc[:1]
                signature = mlflow.models.infer_signature(df.drop(columns="decision"), df["decision"])

                # load dataset to MLflow
                mlflow.log_artifact(data_path, artifact_path="datasets")
            
                # Train and get accuracies
                train_accuracy, test_accuracy, xgb = obj_mlmodel.train_and_save_model(df)
                print("Model trained successfully.")

                # Log model to MLflow
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)
                print("Metrics logged successfully.")

                # Log the trained model with input example and signature
                mlflow.sklearn.log_model(
                    sk_model=xgb,
                    name="model",
                    input_example=input_example,
                    signature=signature,
                )
                print("Model logged successfully.")

                # Register the model
                model_uri = f"runs:/{run.info.run_id}/model"
                registered_model_version = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name
                )
                print("Model registered successfully.")

                # Transition model to Staging
                mlflow_client = mlflow.tracking.MlflowClient()
                mlflow_client.set_model_version_tag(
                    name=model_name,
                    version=registered_model_version.version,
                    key="stage",
                    value="Staging"
                )
                print("Model transitioned to Staging successfully.")

                # Clean up the temporary data file
                os.remove(data_path)

                return {
                    "message": "Model Trained Successfully",
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                }, 200
            
        except Exception as mfe:
            return {"message": "MLFlow Error", "error": str(mfe)}, 500
        except Exception as e:
            return {"message": "Internal Server Error", "error": str(e)}, 500


@ns.route('/predict')
class Predict(Resource):
    @api.expect(predict_model)
    def post(self):
        try:
            data = request.get_json()
            if "inference_row" not in data:
                return {"error": "No inference_row found"}, 400

            infer_array = data["inference_row"]
            if obj_mlmodel.model is None:
                return {"error": "No model is loaded. Train a model before inference."}, 400

            # Set a unique name for the inference run based on timestamp
            run_name = f"infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                y_pred = obj_mlmodel.predict(infer_array)
            
                # log input and output of inference to MLFlow
                mlflow.log_param("inference_input", infer_array)
                mlflow.log_param("inference_output", y_pred)

            return {"message": "Inference Successful", "prediction": y_pred}, 200
        except Exception as e:
            return {"message": "Internal Server Error", "error": str(e)}, 500


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="127.0.0.1", port=8080, debug=False)
