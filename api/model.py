import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score

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
            print(f"Classes for '{column}': {encoder.classes_}")  # Print out classes for verification
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Apply label encoding to each specified column
        for column, encoder in self.encoders.items():
            # Transform the column with a fallback for unseen labels
            X_copy[column] = X_copy[column].apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )
        return X_copy


# Step 1: Load the data and prepare it
df = pd.read_csv('Churn_Modelling.csv')

# Drop unnecessary columns
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Feature Selection
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Step 2: Set up a ColumnTransformer for Label Encoding (for categorical columns) and Standard Scaling (for numerical columns)
categorical_features = ['Geography', 'Gender']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Preprocessing steps: Custom LabelEncoder for categorical features, Standard Scaling for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', LabelEncoderTransformer(categorical_features), categorical_features),  # Custom Label Encoding
        ('num', StandardScaler(), numerical_features)   # Standard scaling for numerical features
    ]
)

# Step 3: Set up the pipeline including the preprocessor and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='poly'))  # SVM classifier
])

# Step 4: Train the model using the pipeline
pipeline.fit(X_train, y_train)

# **SAVE THE MODEL AND PIPELINE**
with open('model_pipeline.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)

print("Pipeline model has been saved.")

# Step 5: Evaluate the model
y_pred = pipeline.predict(X_test)
print(f"Accuracy of the SVM model is {accuracy_score(y_test, y_pred) * 100:.2f}%")
def make_prediction(input_data):
    # Convert the input data dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make a prediction using the loaded pipeline
    prediction = pipeline.predict(input_df)
    
    # Return the prediction result (1 for "Exited", 0 for "Not Exited")
    return "Exited" if prediction[0] == 1 else "Not Exited"

# Example input data (similar to the data format from Form.jsx)
input_data = {
    'CreditScore': 619,
    'Geography': 'France',
    'Gender': 'Female',
    'Age': 42,
    'Tenure': 2,
    'Balance': 0,
    'NumOfProducts': 1,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 101348.9
}

# Run the prediction
result = make_prediction(input_data)
print(f"Prediction for input data: {result}")
