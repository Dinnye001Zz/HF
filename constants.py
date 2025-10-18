"""GOAL: Define columns to be used later in the pipeline."""

CATEGORICAL_COLS = ['country', 'city', 'property_type', 'furnishing_status']

NUMERICAL_COLS = [
    'property_size_sqft',
    'price',
    'customer_salary',
    'loan_amount',
    'monthly_expenses',
    'down_payment'
]

ORDINAL_COLS = [
    'constructed_year',
    'loan_tenure_years',
    'emi_to_income_ratio'
]

BINARY_COLS = [
    'garage',
    'garden',
    'legal_cases_on_property'
]

DISCRETE_COLS = [
    'previous_owners',
    'rooms',
    'bathrooms',
    'crime_cases_reported',
    'satisfaction_score',
    'neighbourhood_rating',
    'connectivity_score'
]