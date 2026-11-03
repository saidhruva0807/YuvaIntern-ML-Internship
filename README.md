
# Breast Cancer Classification – Week 4 Capstone

## 1. Project Overview
This project demonstrates an end-to-end Machine Learning workflow
using the Breast Cancer Wisconsin dataset and Logistic Regression.

## 2. Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Flask
- Joblib

## 3. Project Workflow
1. Loaded the Breast Cancer Wisconsin dataset.
2. Split the data into 80% training and 20% testing.
3. Applied StandardScaler for feature scaling.
4. Trained a Logistic Regression classification model.
5. Evaluated the model using classification metrics.
6. Saved the trained model using Joblib.
7. Built a Flask API for predictions.
8. Tested the API using a sample input.

## 4. Model Performance
- Accuracy: 98.25%
- Precision: 98.61%
- Recall: 98.61%
- F1 Score: 98.61%
- ROC-AUC: 99.60%

## 5. Confusion Matrix
[[41, 1],
 [1, 71]]

## 6. API Endpoints
- GET / : Home page
- GET /health : API health check
- POST /predict : Returns a classification prediction

## 7. Model Serialization
The trained model is stored in breast_cancer_model.joblib.
Feature names are stored in feature_names.json.

## 8. Testing
The Flask API returned HTTP status 200 for a sample prediction.
The returned result included a prediction and confidence score.

## 9. Limitation
The application was tested locally within Google Colab.
It has not been deployed to permanent public hosting.
This project is an educational demonstration and not a medical diagnostic tool.
