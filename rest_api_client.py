import requests
import json
import pandas as pd
import numpy as np

url = "http://127.0.0.1:8080/model/predict"

data_path = "data/global_house_purchase_dataset.csv"
df = pd.read_csv(data_path)
df.drop("decision", axis=1, inplace=True)
df.drop("property_id", axis=1, inplace=True)

index = 0

inference_row = df.iloc[index].apply(lambda x: x.item() if isinstance(x, np.generic) else x).tolist()

data = {
    "inference_row": pd.Series(inference_row).astype(str).to_list()
}

response = requests.post(url, json=data)

print('Status Code:', response.status_code)
print('Response Body:', response.text)
print('Data sent for prediction:', data)


# Create a valid JSON for the Swagger API
# Convert to JSON string with proper formatting
json_data = json.dumps(data, indent=2)

print("JSON for Swagger API:")
print(json_data)

# For direct copy-paste into Swagger UI
print("\nRaw JSON (for copy-paste):")
print(json_data.replace('\n', '').replace(' ', ''))

# You can also make a direct API call with this data
swagger_response = requests.post(url, json=data)
print("\nSwagger API Response:")
print(f"Status Code: {swagger_response.status_code}")
print(f"Response: {swagger_response.text}")
