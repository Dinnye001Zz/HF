from MLModel import MLModel
from constants import NUMERICAL_COLS, CATEGORICAL_COLS, ORDINAL_COLS, BINARY_COLS, DISCRETE_COLS
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

"""Already trained model usage on new data"""

model_handler = MLModel()

label_encoders = model_handler.load_model('artifacts/label_encoders.pkl')
scaler = model_handler.load_model('artifacts/scaler.pkl')
model = model_handler.load_model('artifacts/xgb_model.pkl')

path = 'data/global_house_purchase_dataset.csv'
df = pd.read_csv(path)

sample_data = df.iloc[0]
sample_data = pd.DataFrame([sample_data])

sample_data = sample_data.drop('property_id', axis=1)

for col in CATEGORICAL_COLS:
    sample_data[col] = label_encoders[col].transform(sample_data[col])

all_numerical_to_scale = NUMERICAL_COLS + ORDINAL_COLS
sample_data[all_numerical_to_scale] = scaler.transform(sample_data[all_numerical_to_scale])

if 'decision' in sample_data.columns:
    sample_data = sample_data.drop('decision', axis=1)

predicted_value = model.predict(sample_data)
print(f"Predicted value for the sample data: {predicted_value[0]}")