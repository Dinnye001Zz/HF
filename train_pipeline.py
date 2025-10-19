import pandas as pd
from constants import NUMERICAL_COLS, CATEGORICAL_COLS, ORDINAL_COLS, BINARY_COLS, DISCRETE_COLS
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import LabelEncoder, StandardScaler

"""GOAL: reproduce the preprocessing steps and train a model as per the given preprocessing code. (data_analysis_and_preprocessing.py)"""

df = pd.read_csv('data/global_house_purchase_dataset.csv')

#1. Drop property_id.
df = df.drop('property_id', axis=1)

df_encoded = df.copy()

# 2. Categorical columns - Label encode to numerical values
label_encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# 3. Apply standardization to continuous and ordinal columns
all_numerical_to_scale = NUMERICAL_COLS + ORDINAL_COLS
scaler = StandardScaler()
df_encoded[all_numerical_to_scale] = scaler.fit_transform(df_encoded[all_numerical_to_scale])

#TRAINING A MODEL
X = df_encoded.drop('decision', axis=1)
y = df_encoded['decision']

print(X.head(1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

xgb = XGBClassifier(
            max_depth=2,
            n_estimators=5
)
xgb.fit(X_train, y_train)

y_train_pred = xgb.predict(X_train)
y_test_pred = xgb.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

"""with open('artifacts/xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)"""