import requests
import json
import pandas as pd
import numpy as np

url = 'http://127.0.0.1:8080/model/predict'

data_path = 'data/global_house_purchase_dataset.csv'
df = pd.read_csv(data_path)

index = 2

inference_row = df.iloc[index].apply(lambda x: x.item() if isinstance(x, np.generic) else x).tolist()

data = {
    "inference_row": pd.Series(inference_row).astype(str).tolist()
}

response = requests.post(url, json=data)

print("Response Status Code:", response.status_code)
print("Response body:", response.text)