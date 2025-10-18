import pandas as pd
from constants import NUMERICAL_COLS, CATEGORICAL_COLS, ORDINAL_COLS, BINARY_COLS, DISCRETE_COLS
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
pd.set_option('display.max_columns', None)

"""GOAL:
- Load the dataset, apply saved preprocessing steps
- Train and evaluate a model"""

df = pd.read_csv('data/global_house_purchase_dataset.csv')

df = df.drop('property_id', axis=1)

with open('artifacts/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('artifacts/scaler.pkl', 'rb') as f:  
    scaler = pickle.load(f)

df_encoded = df.copy()

for col in CATEGORICAL_COLS:
    df_encoded[col] = label_encoders[col].transform(df_encoded[col])

all_numerical_to_scale = NUMERICAL_COLS + ORDINAL_COLS
df_encoded[all_numerical_to_scale] = scaler.transform(df_encoded[all_numerical_to_scale])
#print(df_encoded.head())

#TRAINING A MODEL
X = df_encoded.drop('decision', axis=1)
y = df_encoded['decision']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = XGBClassifier(
            max_depth=2,
            n_estimators=5
)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

with open('artifacts/xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)