from fastapi import FastAPI, Query
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and preprocess the dataset
heart = pd.read_csv('heartv1.csv')

# Encode the 'sex' column
label_encoder = LabelEncoder()
heart['sex'] = label_encoder.fit_transform(heart['sex'])

# Define features and target
X = heart.drop(columns=['target'])
y = heart['target']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Gradient Boosting model
gradient_boosting_model = GradientBoostingClassifier(random_state=42)
gradient_boosting_model.fit(X_scaled, y)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease Prediction API!"}

# Prediction endpoint using query parameters
@app.get("/predict_heart_disease/")
def predict_heart_disease(
    sex: int = Query(..., description="Sex (male=0, female=1)"),
    age: int = Query(..., description="Age in years"),
    cp: int = Query(..., description="Chest Pain Type (0/1/2/3)"),
    resting_BP: float = Query(..., description="Resting Blood Pressure in mm Hg"),
    chol: float = Query(..., description="Cholesterol level in mg/dL"),
    fbs: int = Query(..., description="Fasting Blood Sugar (>120 mg/dL, 1: True, 0: False)"),
    restecg: int = Query(..., description="Resting ECG Results (0/1/2)"),
    thalach: float = Query(..., description="Maximum Heart Rate Achieved"),
    exang: int = Query(..., description="Exercise Induced Angina (1: Yes, 0: No)"),
    oldpeak: float = Query(..., description="ST Depression"),
    slope: int = Query(..., description="Slope of Peak Exercise ST Segment (0/1/2)"),
    ca: int = Query(..., description="Number of Major Vessels Colored (0-3)"),
    thal: int = Query(..., description="Thalassemia (1: Normal, 2: Fixed Defect, 3: Reversible Defect)"),
    max_heart_rate_reserve: float = Query(..., description="Max Heart Rate Reserve"),
    heart_disease_risk_score: float = Query(..., description="Heart Disease Risk Score")
):
    # Combine inputs into a feature array
    input_features = [[
        sex, age, cp, resting_BP, chol, fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal, max_heart_rate_reserve, heart_disease_risk_score
    ]]

    # Scale the input data
    input_features_scaled = scaler.transform(input_features)

    # Make a prediction
    prediction = gradient_boosting_model.predict(input_features_scaled)[0]
    result = "Heart Disease Present (1)" if prediction == 1 else "No Heart Disease (0)"

    return {"Predicted Risk": result}
