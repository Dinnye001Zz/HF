import requests
import json
import pandas as pd
import numpy as np

url = "http://127.0.0.1:8080/model/predict"

data_path = "artifacts/preprocessed/saved_dataframe_new.csv"
df = pd.read_csv(data_path)
df.drop("decision", axis=1, inplace=True)

index = 1

inference_row = (
    df.iloc[index]
    .apply(lambda x: x.item() if isinstance(x, np.generic) else x)
    .tolist()
)

data = {"inference_row": pd.Series(inference_row).astype(str).tolist()}

response = requests.post(url, json=data)

print("Response Status Code:", response.status_code)
print("Response body:", response.text)
