from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import requests
import mlflow
from mlflow import MlflowClient
import os

# Define the MLflow client - use container name to reach services on same Docker network
mlflow.set_tracking_uri("http://mlsecops:5102")
client = MlflowClient()

# Set the default model name (must match the name in app.py)
model_name = "House_Purchase_Approval_Model"

# Path to the CSV file (mounted from host)
csv_file_path = "/opt/airflow/data/global_house_purchase_dataset.csv"

def train_model():
    # Call the train endpoint with the CSV file
    with open(csv_file_path, 'rb') as f:
        files = {'file': ('global_house_purchase_dataset.csv', f)}
        # Use container name to reach services on same Docker network
        response = requests.post("http://mlsecops:8080/model/train", files=files)
    
    # Check if the request was successful
    if response.status_code != 200:
        data = response.json()
        raise Exception(f"Training failed: {data.get('error', 'Unknown error')}")
    
    # Parse response data
    data = response.json()
    new_accuracy = data["test_accuracy"]

    # Retrieve the latest model with stage=Staging tag (not using deprecated stages API)
    try:
        # Search for registered models
        registered_models = client.search_registered_models(f"name='{model_name}'")
        staging_model = None
        
        for rm in registered_models:
            for version in rm.latest_versions:
                # Check if this version has the Staging tag
                if version.tags.get("stage") == "Staging":
                    staging_model = version
                    break
            if staging_model:
                break
        
        if staging_model:
            # Get the staging model's test accuracy
            staging_accuracy = client.get_metric_history(staging_model.run_id, "test_accuracy")[-1].value
            
            # If new model is better or equal, update the Staging tag
            if new_accuracy >= staging_accuracy:
                # Remove Staging tag from old model
                client.set_model_version_tag(
                    name=model_name,
                    version=staging_model.version,
                    key="stage",
                    value="Archived"
                )
                print(f"Archived old model version {staging_model.version}")
                
                # The new model already has the Staging tag (set by app.py during training)
                print("New model is already tagged as Staging")
            else:
                print(f"New model accuracy ({new_accuracy}) is lower than staging ({staging_accuracy})")
                return False  # Indicates that the new model's accuracy is lower
        else:
            print("No existing Staging model found. New model will remain as Staging.")
    
    except Exception as e:
        print(f"Error comparing with staging model: {e}")
        # If there's an error (e.g., no staging model exists), continue anyway
    
    print("Model training and evaluation completed successfully.")
    return True

def branch_decision(**kwargs):
    # Get the return value from train_and_compare_model
    result = kwargs['ti'].xcom_pull(task_ids='train_and_compare_model')

    if result is True:
        return 'send_notification'
    else:
        return 'skip_notification'



# Define the Airflow DAG
with DAG(
    dag_id="daily_model_training",
    start_date=datetime(2023, 11, 1),
    schedule_interval="0 2 * * *",  # Runs daily at 2 AM
    catchup=False,
) as dag:

    # Define a task that calls the train_model function
    train_and_compare_task = PythonOperator(
        task_id="train_and_compare_model",
        python_callable=train_model,
        retries=3,
        retry_delay=timedelta(minutes=5),
    )

    # Branch task decides whether to send email or skip
    branch_task = BranchPythonOperator(
        task_id="branch_decision",
        python_callable=branch_decision,
        provide_context=True,
    )

    # Define notification tasks (using PythonOperator instead of EmailOperator to avoid SMTP config)
    notification_task = PythonOperator(
        task_id="send_notification",
        python_callable=lambda: print("New model accuracy is LOWER than staging model!"),
    )
    
    skip_notification = PythonOperator(
        task_id="skip_notification",
        python_callable=lambda: print("New model is better or equal."),
    )

    # Set task dependencies
    train_and_compare_task >> branch_task
    branch_task >> [notification_task, skip_notification]
