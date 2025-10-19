import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle
pd.set_option('display.max_columns', None)

"""This file: Data Scientist gives this type of preprocessed python files to the ML Engineer.
The ML Enginieer uses this code to understand what preprocessing steps were applied and tries to reproduce it.
- property_id was dropped as it's just an identifier, not useful for ML.
- "decision" (y) is not preprocessed and is not part of the preprocessed df as it was already binary.
"""

df = pd.read_csv('data/global_house_purchase_dataset.csv')

# BASIC EDA
print(df.head())
print(df.describe())
print(df.info())
print(df['decision'].value_counts())

# PREPROCESSING STEPS

# 1. Drop property_id - it's just an identifier, not useful for ML
df = df.drop('property_id', axis=1)

# 2. Categorical columns - Label encode to numerical values
df_encoded = df.copy()
CATEGORICAL_COLS = ['country', 'city', 'property_type', 'furnishing_status']

# Dictionary to store all encoders
label_encoders = {}

for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

"""# Save the label encoders for later use
with open('artifacts/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)"""

# 3. Numerical columns with wide ranges - Standardize (mean=0, std=1)
NUMERICAL_COLS = [
    'property_size_sqft',    # Range: 400-6000, wide range
    'price',                 # Range: 56k-4.2M, very wide range  
    'customer_salary',       # Range: 2k-100k, wide range
    'loan_amount',          # Range: 23k-3.5M, very wide range
    'monthly_expenses',     # Range: 500-20k, wide range
    'down_payment'          # Range: 8k-2.5M, very wide range
]

# 4. Integer columns with meaningful ordinal relationships - Keep as is or standardize
ORDINAL_COLS = [
    'constructed_year',      # Year values, standardize to handle age effect
    'loan_tenure_years',     # Range: 10-30, relatively small range but standardize for consistency
    'emi_to_income_ratio'    # Already a ratio (0-3.46), but standardize due to wide range
]

# 5. Binary/categorical columns that are already encoded - Keep as is
BINARY_COLS = [
    'garage',               # Binary: 0 or 1
    'garden',               # Binary: 0 or 1  
    'legal_cases_on_property'  # Binary: 0 or 1
]

# 6. Count/discrete columns - Keep as is (they're already on reasonable scales)
DISCRETE_COLS = [
    'previous_owners',       # Range: 0-6, small discrete values
    'rooms',                # Range: 1-8, small discrete values
    'bathrooms',            # Range: 1-8, small discrete values
    'crime_cases_reported', # Range: 0-10, small discrete values
    'satisfaction_score',   # Range: 1-10, ordinal scale
    'neighbourhood_rating', # Range: 1-10, ordinal scale  
    'connectivity_score'    # Range: 1-10, ordinal scale
]

# Apply standardization to continuous and ordinal columns
all_numerical_to_scale = NUMERICAL_COLS + ORDINAL_COLS
scaler = StandardScaler()
df_encoded[all_numerical_to_scale] = scaler.fit_transform(df_encoded[all_numerical_to_scale])

#print(f'df_encoded.columns: {df_encoded.columns}')
#print(df_encoded.head())

# Save the scaler
"""with open('artifacts/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)"""

"""# Save the preprocessed dataframe
df_encoded.to_csv('artifacts/preprocessed_data.csv', index=False)"""