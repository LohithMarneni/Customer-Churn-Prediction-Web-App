from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd  # Import pandas to create a DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Custom transformer for LabelEncoding multiple columns
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        # Fit label encoder for each specified column
        for column in self.columns:
            encoder = LabelEncoder()
            encoder.fit(X[column])
            self.encoders[column] = encoder
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Apply label encoding to each specified column
        for column, encoder in self.encoders.items():
            X_copy[column] = encoder.transform(X_copy[column])
        return X_copy

# Load the trained model pipeline (including custom transformer) from the file
with open('model_pipeline.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])

def predict():
    try:
        # Get data from the POST request
        data = request.get_json()

        # Check if all required features are present
        required_features = [
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ]
        
        for feature in required_features:
            if feature not in data:
                return jsonify({
                    'error': f'Missing feature: {feature}',
                    'message': 'Failed to process request'
                })

        # Extract features from the JSON data
        features = [
            data['CreditScore'],
            data['Geography'],
            data['Gender'],
            data['Age'],
            data['Tenure'],
            data['Balance'],
            data['NumOfProducts'],
            data['HasCrCard'],
            data['IsActiveMember'],
            data['EstimatedSalary']
        ]

        # Convert features into numpy array and scale it
        # Convert the input features into a DataFrame with the correct column names
        feature_names = [
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ]
        
        features_df = pd.DataFrame([features], columns=feature_names)

        # Predict using the loaded pipeline (which includes preprocessing and classification)
        prediction = model_pipeline.predict(features_df)

        # Return the prediction result as a JSON response
        return jsonify({
            'churn_prediction': int(prediction[0]),
            'message': 'Prediction successful'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to process request'
        })

if __name__ == '__main__':
    app.run(debug=True)
