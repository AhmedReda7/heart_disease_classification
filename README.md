Heart Disease Classification and Prediction

This project applies machine learning and web technologies to predict the likelihood of heart disease in patients based on clinical and lifestyle features.

Project Overview

Built a Heart Disease Prediction Model using multiple machine learning algorithms such as:

Gradient Boosting

Bagging

Random Forest

Extra Trees

K-Nearest Neighbors (KNN)

Logistic Regression

Support Vector Machines (SVM – Linear, RBF, Polynomial)

Conducted Exploratory Data Analysis (EDA) to identify correlations and feature importance.

Achieved up to 100% accuracy with ensemble models (Random Forest & Extra Trees).

Developed a FastAPI backend that serves predictions via REST API.

Built a responsive frontend (HTML/CSS/JS) where users can input patient details and receive real-time predictions.

Dataset

The dataset (heartv1.csv) contains 1,035 patient records with features such as:

age – Patient’s age

sex – Gender (male/female)

cp – Chest pain type (0–3)

resting_BP – Resting blood pressure (mm Hg)

chol – Serum cholesterol (mg/dL)

fbs – Fasting blood sugar (>120 mg/dL)

restecg – Resting ECG results (0–2)

thalach – Maximum heart rate achieved

exang – Exercise-induced angina (1: Yes, 0: No)

oldpeak – ST depression

slope, ca, thal – ECG-related features

Max Heart Rate Reserve & Heart Disease Risk Score (derived features)

target – Heart disease presence (1: Yes, 0: No)

Technical Workflow

Data Preprocessing

Encoded categorical variables (e.g., sex).

Scaled features using StandardScaler.

Split data into training & test sets (80/20).

Model Training & Evaluation

Compared multiple models.

Achieved top results with Random Forest and Extra Trees (100% accuracy).

Gradient Boosting and Bagging also performed well (>97%).

API Deployment (FastAPI)

Created REST endpoints for heart disease prediction.

Input parameters are passed as query values, scaled, and fed into the trained model.

Returns a prediction result (Heart Disease Present / No Heart Disease).

Frontend (index.html)

Clean, responsive UI built with HTML/CSS.

Users enter medical details into a form.

Fetches results from the FastAPI backend.

Tech Stack

Languages & Libraries: Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

Backend: FastAPI, CORS middleware

Frontend: HTML, CSS (with animations), JavaScript

ML Models: Gradient Boosting, Random Forest, Extra Trees, Bagging, Logistic Regression, KNN, SVM

Results

Random Forest & Extra Trees: 100% accuracy

Gradient Boosting: 98.5% accuracy

Bagging: 97% accuracy

KNN: 96% accuracy

Logistic Regression: 86% accuracy

SVM (RBF): 92% accuracy

Usage

Run the FastAPI server:

uvicorn app:app --reload


Open index.html in a browser.

Enter patient details → get instant heart disease prediction.
