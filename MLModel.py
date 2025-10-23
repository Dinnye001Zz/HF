import os, pickle, mlflow, json
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

pd.set_option("display.max_columns", None)


class MLModel:
    # This part always run when you create an object from MLModel class
    def __init__(self, client):
        """self.label_encoders = (
            MLModel.load_model("artifacts/encoders/label_encoder.pkl")
            if os.path.exists("artifacts/encoders/label_encoder.pkl")
            else print("label_encoder.pkl does not exist.")
        )

        if isinstance(self.label_encoders, dict):
            for col, le in self.label_encoders.items():
                print(f"LabelEncoder for '{col}': {le.classes_}")

        self.scaler = (
            MLModel.load_model("artifacts/scalers/scaler.pkl")
            if os.path.exists("artifacts/scalers/scaler.pkl")
            else print("scaler.pkl does not exist.")
        )

        self.model = (
            MLModel.load_model("artifacts/models/xgb_model.pkl")
            if os.path.exists("artifacts/models/xgb_model.pkl")
            else print("xgb_model.pkl does not exist.")
        )"""
        self.client = client
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.load_staging_model()
        if isinstance(self.label_encoder, dict):
            for col, le in self.label_encoder.items():
                print(f"LabelEncoder for '{col}': {le.classes_}")
        print("Loaded label encoders:", self.label_encoder)
        print("Loaded scaler:", self.scaler)
    
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


    @staticmethod
    def save_model(model, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(model, file)

    @staticmethod
    def load_model(file_path):
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        return model

    def preprocess_pipeline(self, df):
        # 1. Drop property_id.
        df = df.drop("property_id", axis=1)

        #df = df.copy()

        # 2. Categorical columns - Label encode to numerical values
        label_encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        self.label_encoder = label_encoders

        # convert cols to int so they are not object type
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # 3. Apply standardization to continuous and ordinal columns
        all_numerical_to_scale = NUMERICAL_COLS + ORDINAL_COLS
        scaler = StandardScaler()
        df[all_numerical_to_scale] = scaler.fit_transform(df[all_numerical_to_scale])
        self.scaler = scaler

        print("Data types after preprocessing:", df.dtypes)

        # convert int columns to float to avoid schema issues
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = df[col].astype(float)
        
        # convert all columns to float
        for col in df.columns:
            df[col] = df[col].astype(float)

        # SAVE ARTIFACTS
        # encoder
        """os.makedirs("artifacts/encoders", exist_ok=True)
        with open("artifacts/encoders/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoders, f)

        # scaler
        os.makedirs("artifacts/scalers", exist_ok=True)
        with open("artifacts/scalers/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)"""

        """NINCS LEMENTVE A MODEL (XGB)"""
        # Serialize and log scalers and encoders
        with open("artifacts/encoders/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        mlflow.log_artifact("artifacts/encoders/label_encoder.pkl")

        with open("artifacts/scalers/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        mlflow.log_artifact("artifacts/scalers/scaler.pkl")

        return df

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
        """Itt kellene elementeni a modellt?!?"""
        """# I added model save
        os.makedirs("artifacts/models", exist_ok=True)
        with open("artifacts/models/xgb_model.pkl", "wb") as f:
            pickle.dump(xgb, f)
        print("Model saved")"""
        return train_accuracy, test_accuracy, xgb

    def preprocess_pipeline_inference(self, sample_data):
        print(f"infer array: {sample_data}")
        print(f"Length of infer_array: {len(sample_data)}")
        print("COLUMN_ORDER_AFTER_PREPROCESSING:", COLUMN_ORDER_AFTER_PREPROCESSING)
        print("Length of COLUMN_ORDER_AFTER_PREPROCESSING:", len(COLUMN_ORDER_AFTER_PREPROCESSING))
        
        try:
            sample_data = pd.DataFrame([sample_data], columns=COLUMN_ORDER_AFTER_PREPROCESSING)
        except Exception as e:
            print("Error creating DataFrame:", e)
            raise
        print(f"MLMODEL: df.head(1): {sample_data.head(1)}")

        if self.label_encoder:
            for col in CATEGORICAL_COLS:
                if col in sample_data.columns:
                    sample_data[col] = self.label_encoder[col].transform(sample_data[col])

        # convert cols to int so they are not object type
        for col in CATEGORICAL_COLS:
            if col in sample_data.columns:
                sample_data[col] = sample_data[col].astype(int)

        if self.scaler:
            all_numerical_to_scale = NUMERICAL_COLS + ORDINAL_COLS
            sample_data[all_numerical_to_scale] = self.scaler.transform(
                sample_data[all_numerical_to_scale]
            )

        # convert int columns to float to avoid schema issues
        for col in sample_data.select_dtypes(include=['int']).columns:
            sample_data[col] = sample_data[col].astype(float)
        
        # convert all columns to float
        for col in sample_data.columns:
            sample_data[col] = sample_data[col].astype(float)

        expected_column_order = COLUMN_ORDER_AFTER_PREPROCESSING

        columns_to_use = [col for col in expected_column_order if col in sample_data.columns]
        sample_data = sample_data[columns_to_use]

        print(f"MLMODEL: df.head(1): {sample_data.head(1)}")

        return sample_data

    def get_accuracy_full(self, X, y):
        y_pred = self.model.predict(X)

        accuracy = accuracy_score(y, y_pred)

        print("Accuracy: ", accuracy)

        return accuracy

    def get_accuracy(self, X_train, X_test, y_train, y_test):
        y_train_pred = self.model.predict(X_train)

        y_test_pred = self.model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print("Train Accuracy: ", train_accuracy)
        print("Test Accuracy: ", test_accuracy)

        return train_accuracy, test_accuracy

    def predict(self, inference_row):
        if self.model is None:
            return {'error': 'No staging model is loaded'}, 400

        processed_data = self.preprocess_pipeline_inference(inference_row)
        prediction = self.model.predict(processed_data)
        return int(prediction)
