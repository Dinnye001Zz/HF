from flask import Flask, request
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
import os
import pandas as pd
from MLModel import MLModel

app = Flask(__name__)
api = Api(app, version='1.0', title='API Documentation')

obj_mlmodel = MLModel()

predict_model = api.model('PredictModel', {
    'inference_row': fields.List(fields.Raw, required=True,
                                 description='A row of data for inference')
})

file_upload = api.parser()
file_upload.add_argument('file', location='files',
                         type=FileStorage, required=True,
                         help='CSV file for training')

ns = api.namespace('model', description='Model operations')

@ns.route('/train')
class Train(Resource):
    @ns.expect(file_upload)
    def post(self):
        args = file_upload.parse_args()
        uploaded_file = args['file']
        if os.path.splitext(uploaded_file.filename)[1] != '.csv':
            return {'error': 'Only CSV files are allowed'}, 400
        
        data_path = 'data/temp_uploaded_dataset.csv'
        uploaded_file.save(data_path)

        try:
            df = pd.read_csv(data_path)
            df_preprocessed = obj_mlmodel.preprocess_pipeline(df)
            print(df_preprocessed.head())
            train_accuracy, test_accuracy = obj_mlmodel.train_and_save_model(df_preprocessed)
            obj_mlmodel.save_model(obj_mlmodel.model, 'artifacts/xgb_model.pkl')
            df_preprocessed.to_csv('artifacts/saved_dataframe_new.csv', index=False)
            os.remove(data_path)

            return {'message': 'File processed and saved successfully',
                    'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}, 200
        except Exception as e:
            return {'message': 'Internal server error', 'error': str(e)}, 500

@ns.route('/predict')
class Predict(Resource):
    @ns.expect(predict_model)
    def post(self):
        try:
            data = request.json()
            if 'inference_row' not in data:
                return {'error': 'Missing inference_row in request'}, 400
            
            infer_array = data['inference_row']
            df = obj_mlmodel.preprocess_pipeline_inference(infer_array)
            y_pred = obj_mlmodel.model.predict(df)

            return {'message': 'Inference Successful', 'predictions': int(y_pred)}, 200
        except Exception as e:
            return {'message': 'Internal server error', 'error': str(e)}, 500
