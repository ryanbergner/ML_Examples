# Importing necessary libraries for data manipulation and model creation
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Loading dataset and defining features (X) and target (y)
data = load_breast_cancer()
X = data.data
y = data.target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features for normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Defining and training the SVM model
model = SVC(C=0.1, kernel='rbf', gamma=1, class_weight="balanced")
model.fit(X_train_scaled, y_train)  # Fitting the model to the training data

# Cross-Validation to evaluate model performance
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10)

# GridSearchCV for hyperparameter tuning
param_grid = {'C': [0.1, 1.0, 10.0], 'gamma': [0.01, 0.1, 1], 'kernel': ['rbf', 'linear']}
grid_search = GridSearchCV(SVC(), param_grid, cv=10)
grid_search.fit(X_train_scaled, y_train)

# Evaluation of the best model found by GridSearchCV
best_model = grid_search.best_estimator_
best_score = best_model.score(X_test_scaled, y_test)

# Model Evaluation Metrics Functions
def calculate_classification_metrics(df):
    TP = df.loc['Actual (1/P)', 'Predicted (1/P)']
    TN = df.loc['Actual (0/N)', 'Predicted (0/N)']
    FP = df.loc['Actual (0/N)', 'Predicted (1/P)']
    FN = df.loc['Actual (1/P)', 'Predicted (0/N)']
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1_score, 3)

def calculate_regression_metrics(df):
    actual = df['Actual']
    predicted = df['Predicted']
    
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean(np.square(actual - predicted))
    rmse = np.sqrt(mse)
    
    return round(mae, 2), round(mse, 2), round(rmse, 2)
