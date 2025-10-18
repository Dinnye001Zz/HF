import pickle
import os
from constants import NUMERICAL_COLS, CATEGORICAL_COLS, ORDINAL_COLS, BINARY_COLS, DISCRETE_COLS
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class MLModel:
    def __init__(self):
        self.label_encoders = (MLModel.load_model(
            'artifacts/label_encoders.pkl')
                if os.path.exists('artifacts/label_encoders.pkl')
                else print("label_encoders.pkl does not exist."))
            
        self.scaler = (MLModel.load_model(
            'artifacts/scaler.pkl')
                if os.path.exists('artifacts/scaler.pkl')
                else print("scaler.pkl does not exist."))

        self.model = (MLModel.load_model(
            'artifacts/xgb_model.pkl')
                if os.path.exists('artifacts/xgb_model.pkl')
                else print("xgb_model.pkl does not exist."))

    @staticmethod
    def save_model(model, path):
        with open(path, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as file:
            return pickle.load(file)
    
    def preprocess_pipeline(self, df):
        df = df.copy()
        if 'property_id' in df.columns:
            df = df.drop('property_id', axis=1)

        df_preprocessed = df.copy()
        
        label_encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df_preprocessed[col] = le.fit_transform(df_preprocessed[col])
            label_encoders[col] = le
        
        os.makedirs('artifacts', exist_ok=True)
        with open('artifacts/label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
        
        scaler = StandardScaler()
        all_numerical_to_scale = NUMERICAL_COLS + ORDINAL_COLS
        df_preprocessed[all_numerical_to_scale] = scaler.fit_transform(df_preprocessed[all_numerical_to_scale])
        
        with open('artifacts/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        self.label_encoders = label_encoders
        self.scaler = scaler
        return df_preprocessed

    def train_and_save_model(self, df_preprocessed):
        X = df_preprocessed.drop('decision', axis=1)
        y = df_preprocessed['decision']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        self.model = XGBClassifier(
                    max_depth=2,
                    n_estimators=5
        )
        self.model.fit(X_train, y_train)

        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print(f"Train Accuracy: {train_accuracy}")
        print(f"Test Accuracy: {test_accuracy}")

        return train_accuracy, test_accuracy
    
    def preprocess_pipeline_inference(self, infer_array):
        columns = CATEGORICAL_COLS + NUMERICAL_COLS + ORDINAL_COLS + BINARY_COLS + DISCRETE_COLS
        
        df = pd.DataFrame([infer_array], columns=columns)

        if self.label_encoders:
            for col in CATEGORICAL_COLS:
                if col in df.columns:
                    df[col] = self.label_encoders[col].transform(df[col])

        if self.scaler:
            all_numerical_to_scale = NUMERICAL_COLS + ORDINAL_COLS
            df[all_numerical_to_scale] = self.scaler.transform(df[all_numerical_to_scale])

        expected_column_order = (
            CATEGORICAL_COLS +
            NUMERICAL_COLS +
            ORDINAL_COLS +
            BINARY_COLS +
            DISCRETE_COLS
        )
        
        columns_to_use = [col for col in expected_column_order if col in df.columns]
        df = df[columns_to_use]

        return df
    
    def get_accuracy_full(self, X, y_true):
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy