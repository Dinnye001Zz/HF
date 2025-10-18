import pandas as pd
import numpy as np
import sys
import pathlib as Path

current_file_path = Path.Path(__file__).resolve()
parent_directory = current_file_path.parent.parent
sys.path.append(str(parent_directory))

from MLModel import MLModel

def test_prediction_accuracy():
    obj_mlmodel = MLModel()
    data_path = 'data/global_house_purchase_dataset.csv'
    df = pd.read_csv(data_path).head(1000)

    #train pipeline
    df_preprocessed = obj_mlmodel.preprocess_pipeline(df)
    y_expected = df_preprocessed['decision']
    
    train_features = df_preprocessed.drop('decision', axis=1).columns.tolist()
    
    accuracy_train_pipeline_full = obj_mlmodel.get_accuracy_full(
        df_preprocessed.drop('decision', axis=1),
        y_expected
    )
    accuracy_train_pipeline_full = np.round(accuracy_train_pipeline_full, 2)

    #inference pipeline
    obj_mlmodel = MLModel()
    df = pd.read_csv(data_path).head(1000)

    preprocessed_list = []
    for i in df.iterrows():
        preprocessed_list.append(
            obj_mlmodel.preprocess_pipeline_inference(i[1])
        )

    df_preprocessed_inference = pd.concat(preprocessed_list)
    
    df_preprocessed_inference = df_preprocessed_inference[train_features]
    
    accuracy_inference_pipeline_full = obj_mlmodel.get_accuracy_full(
        df_preprocessed_inference,
        y_expected
    )
    accuracy_inference_pipeline_full = np.round(accuracy_inference_pipeline_full, 2)
    print(f"Accuracy from train pipeline: {accuracy_train_pipeline_full}")
    print(f"Accuracy from inference pipeline: {accuracy_inference_pipeline_full}")

    assert accuracy_train_pipeline_full == accuracy_inference_pipeline_full, "Inference prediction accuracy is not as expected"