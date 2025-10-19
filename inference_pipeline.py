from MLModel import MLModel
from constants import NUMERICAL_COLS, CATEGORICAL_COLS, ORDINAL_COLS, BINARY_COLS, DISCRETE_COLS
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

"""Already trained model and artifacts usage on new data"""

# Everything runs from the MLModel __init__ function
# loads the saved artifacts: label_encoders, scaler, model (xgb)
model_handler = MLModel()

label_encoders = model_handler.load_model('artifacts/encoders/label_encoder.pkl')
scaler = model_handler.load_model('artifacts/scalers/scaler.pkl')
xgb = model_handler.load_model('artifacts/models/xgb_model.pkl')

data_path = 'data/global_house_purchase_dataset.csv'
df = pd.read_csv(data_path)

# Take a sample data for prediction
sample_data = df.iloc[0]
sample_data = pd.DataFrame([sample_data])
#print(f"Sample data before preprocessing:\n{sample_data}")

# We have to preprocess the sample_data the same way as in training
#1. Drop property_id
sample_data = sample_data.drop('property_id', axis=1)

# 2. Label encode with saved artifact (instead fit_transform, only transform)
for col in CATEGORICAL_COLS:
    sample_data[col] = label_encoders[col].transform(sample_data[col])

# 3. Standardize with saved artifact
all_numerical_to_scale = NUMERICAL_COLS + ORDINAL_COLS
sample_data[all_numerical_to_scale] = scaler.transform(sample_data[all_numerical_to_scale])
#print(sample_data.columns)

# +1 Drop  target column if exists
if 'decision' in sample_data.columns:
    sample_data = sample_data.drop('decision', axis=1)

# Predict with the loaded model artifact
predicted_value = xgb.predict(sample_data)

print(f"Predicted value for the sample data: {predicted_value[0]}")